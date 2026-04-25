"""
EvaluationEngine: The central orchestrator.

Coordinates:
  1. Loading models from the registry
  2. Loading datasets for each scenario
  3. Running inference (locally or via Triton)
  4. Computing metrics
  5. Saving results and generating reports
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from vlm_eval.core.config import EvalConfig, ScenarioConfig
from vlm_eval.core.scenario import EvalResult, EvalSample, ScenarioResult
from vlm_eval.datasets.loader import DatasetLoader
from vlm_eval.metrics.registry import MetricRegistry
from vlm_eval.models.registry import ModelRegistry
from vlm_eval.reporters.html_reporter import HTMLReporter


class EvaluationEngine:
    """
    Main evaluation orchestrator.

    Usage
    -----
    >>> from vlm_eval import EvaluationEngine, EvalConfig
    >>> config = EvalConfig.from_yaml("configs/my_suite.yaml")
    >>> engine = EvaluationEngine(config)
    >>> results = engine.run()
    >>> engine.export_report(results)
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.output_path = config.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._model_registry = ModelRegistry()
        self._metric_registry = MetricRegistry()
        self._dataset_loader = DatasetLoader()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, List[ScenarioResult]]:
        """
        Run the full evaluation suite.

        Returns
        -------
        dict mapping model_name → list of ScenarioResult
        """
        logger.info(
            f"Starting evaluation suite: {self.config.suite_name!r} | "
            f"Models: {[m.name for m in self.config.models]} | "
            f"Scenarios: {[s.name for s in self.config.scenarios]}"
        )

        all_results: Dict[str, List[ScenarioResult]] = {}

        for model_cfg in self.config.models:
            logger.info(f"Loading model: {model_cfg.name} ({model_cfg.hf_model_id})")
            model = self._model_registry.load(model_cfg)
            all_results[model_cfg.name] = []

            for scenario_cfg in self.config.scenarios:
                logger.info(
                    f"  Running scenario: {scenario_cfg.name} "
                    f"[{scenario_cfg.task_type}]"
                )
                result = self._run_scenario(model, model_cfg, scenario_cfg)
                all_results[model_cfg.name].append(result)

                agg = result.aggregate_scores()
                logger.info(
                    f"  ✓ {scenario_cfg.name} | {result.n_samples} samples | "
                    f"scores: { {k: f'{v:.3f}' for k,v in agg.items()} }"
                )

            # Unload model to free memory before loading next one
            model.unload()
            logger.info(f"Model {model_cfg.name!r} unloaded.")

        self._save_results(all_results)

        if self.config.export_html_report:
            reporter = HTMLReporter(self.config)
            report_path = reporter.generate(all_results)
            logger.info(f"HTML report saved to: {report_path}")

        logger.info("Evaluation complete ✓")
        return all_results

    def run_single_scenario(
        self,
        model_name: str,
        scenario_name: str,
    ) -> ScenarioResult:
        """
        Run one scenario with one model — useful for quick testing.
        """
        model_cfg = next(
            (m for m in self.config.models if m.name == model_name), None
        )
        scenario_cfg = self.config.get_scenario(scenario_name)

        if model_cfg is None:
            raise ValueError(f"Model {model_name!r} not found in config.")
        if scenario_cfg is None:
            raise ValueError(f"Scenario {scenario_name!r} not found in config.")

        model = self._model_registry.load(model_cfg)
        result = self._run_scenario(model, model_cfg, scenario_cfg)
        model.unload()
        return result

    def get_leaderboard(
        self,
        results: Dict[str, List[ScenarioResult]],
        metric: str = "clip_score",
    ) -> pd.DataFrame:
        """
        Build a leaderboard DataFrame ranked by a chosen metric.

        Returns
        -------
        pd.DataFrame with columns: model, scenario, metric_score
        """
        rows = []
        for model_name, scenario_results in results.items():
            for sr in scenario_results:
                scores = sr.aggregate_scores()
                rows.append({
                    "model": model_name,
                    "scenario": sr.scenario_name,
                    "task_type": sr.task_type,
                    "n_samples": sr.n_samples,
                    **{k: round(v, 4) for k, v in scores.items()},
                })
        df = pd.DataFrame(rows)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        return df

    # ─────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────

    def _run_scenario(
        self,
        model,
        model_cfg,
        scenario_cfg: ScenarioConfig,
    ) -> ScenarioResult:
        """Run inference + metric computation for one scenario."""
        # Load dataset samples
        samples: List[EvalSample] = self._dataset_loader.load(
            dataset_cfg=scenario_cfg.dataset,
            scenario_cfg=scenario_cfg,
            max_samples=scenario_cfg.dataset.max_samples,
        )

        scenario_result = ScenarioResult(
            scenario_name=scenario_cfg.name,
            model_name=model_cfg.name,
            task_type=scenario_cfg.task_type,
        )

        # Run inference in batches
        batch_size = self.config.batch_size
        for i in tqdm(
            range(0, len(samples), batch_size),
            desc=f"{model_cfg.name} / {scenario_cfg.name}",
            leave=False,
        ):
            batch = samples[i: i + batch_size]
            batch_results = self._run_batch(model, model_cfg, scenario_cfg, batch)
            scenario_result.sample_results.extend(batch_results)

        # Save per-sample predictions if requested
        if self.config.save_predictions:
            self._save_predictions(scenario_result)

        return scenario_result

    def _run_batch(
        self,
        model,
        model_cfg,
        scenario_cfg: ScenarioConfig,
        samples: List[EvalSample],
    ) -> List[EvalResult]:
        """Run inference on a batch of samples and compute metrics."""
        results = []
        for sample in samples:
            try:
                t0 = time.perf_counter()
                prediction = model.generate(
                    image=sample.image,
                    prompt=sample.prompt,
                    max_new_tokens=model_cfg.max_new_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                # Compute metrics
                scores = self._metric_registry.compute(
                    prediction=prediction,
                    references=sample.references,
                    image=sample.image,
                    metric_cfg=scenario_cfg.metrics,
                )

                results.append(EvalResult(
                    sample_id=sample.sample_id,
                    model_name=model_cfg.name,
                    prediction=prediction,
                    scores=scores,
                    latency_ms=latency_ms,
                ))

            except Exception as e:
                logger.warning(f"Inference failed for sample {sample.sample_id}: {e}")
                results.append(EvalResult(
                    sample_id=sample.sample_id,
                    model_name=model_cfg.name,
                    prediction="",
                    error=str(e),
                ))

        return results

    def _save_results(self, all_results: Dict[str, List[ScenarioResult]]):
        """Save results as JSON for later analysis."""
        records = []
        for model_name, scenario_results in all_results.items():
            for sr in scenario_results:
                records.append(sr.to_dict())

        out = self.output_path / "results_summary.json"
        with open(out, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Results saved to {out}")

        # Also save as CSV
        df = pd.DataFrame(records)
        df.to_csv(self.output_path / "results_summary.csv", index=False)

    def _save_predictions(self, sr: ScenarioResult):
        """Save per-sample predictions to JSONL."""
        out_dir = self.output_path / "predictions"
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / f"{sr.model_name}_{sr.scenario_name}.jsonl".replace(" ", "_")
        with open(fname, "w") as f:
            for r in sr.sample_results:
                f.write(json.dumps({
                    "sample_id": r.sample_id,
                    "prediction": r.prediction,
                    "scores": r.scores,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }) + "\n")
