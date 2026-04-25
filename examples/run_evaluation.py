"""
examples/run_evaluation.py
==========================
End-to-end demonstration of the VLM Evaluation Framework.

Runs three mock VLMs across five scenario categories on synthetic data —
no GPU or model download required.

Run with:
    python examples/run_evaluation.py
    python examples/run_evaluation.py --n-samples 20 --output ./my_results
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vlm_eval.core.config import (
    EvalConfig, ModelConfig, ScenarioConfig,
    DatasetConfig, MetricConfig,
)
from vlm_eval.core.engine import EvaluationEngine

console = Console()


def build_config(n_samples: int, output_dir: str) -> EvalConfig:
    """Build the evaluation configuration programmatically."""

    # ── Models ──────────────────────────────────────────────────
    models = [
        ModelConfig(
            name="MockVLM-General",
            hf_model_id="mock-general",
            model_type="mock",
        ),
        ModelConfig(
            name="MockVLM-Safety",
            hf_model_id="mock-safety",
            model_type="mock",
        ),
        ModelConfig(
            name="MockVLM-Scene",
            hf_model_id="mock-scene",
            model_type="mock",
        ),
    ]

    # ── Shared metric config ─────────────────────────────────────
    metrics = MetricConfig(
        bleu=True,
        rouge=True,
        bert_score=False,     # slow without GPU
        clip_score=False,     # needs CLIP model download
        semantic_similarity=True,
        exact_match=False,
        accuracy=True,
    )

    # ── Scenarios ────────────────────────────────────────────────
    scenarios = [
        ScenarioConfig(
            name="Anomaly Detection",
            task_type="anomaly_detection",
            description="Detect unusual/unexpected elements in road scenes.",
            prompt_template=(
                "Carefully examine this image. "
                "Describe any unusual, unexpected, or anomalous elements you observe. "
                "If the scene appears normal, explicitly state that."
            ),
            dataset=DatasetConfig(name="synthetic_anomaly", source="synthetic", max_samples=n_samples),
            metrics=metrics,
            tags=["anomaly", "safety", "autonomous"],
        ),
        ScenarioConfig(
            name="Scene Understanding",
            task_type="scene_understanding",
            description="Holistic scene comprehension: type, layout, context.",
            prompt_template=(
                "Describe this scene comprehensively. "
                "What type of environment is shown? "
                "What are the main elements and how are they spatially arranged?"
            ),
            dataset=DatasetConfig(name="synthetic_scene", source="synthetic", max_samples=n_samples),
            metrics=metrics,
            tags=["scene", "captioning"],
        ),
        ScenarioConfig(
            name="Image Captioning",
            task_type="image_captioning",
            description="Standard single-sentence image description.",
            prompt_template="Describe this image in one to two sentences.",
            dataset=DatasetConfig(name="synthetic_caption", source="synthetic", max_samples=n_samples),
            metrics=metrics,
            tags=["captioning", "general"],
        ),
        ScenarioConfig(
            name="Visual QA",
            task_type="visual_question_answering",
            description="Direct factual questions about visual content.",
            prompt_template=(
                "Answer this question about the image: "
                "What is the main colored object, and what color is it?"
            ),
            dataset=DatasetConfig(name="synthetic_vqa", source="synthetic", max_samples=n_samples),
            metrics=MetricConfig(
                bleu=True, rouge=True, semantic_similarity=True,
                accuracy=True, clip_score=False,
            ),
            tags=["vqa", "factual"],
        ),
        ScenarioConfig(
            name="Safety Critical",
            task_type="safety_critical",
            description="Safety-critical road scenarios for autonomous systems.",
            prompt_template=(
                "You are an AI assistant for an autonomous vehicle system. "
                "Analyze this scene and identify any safety-critical elements "
                "that require immediate attention."
            ),
            dataset=DatasetConfig(name="synthetic_safety", source="synthetic", max_samples=n_samples),
            metrics=MetricConfig(
                bleu=True, rouge=True, semantic_similarity=True,
                accuracy=True, clip_score=False,
            ),
            tags=["safety", "autonomous", "critical"],
        ),
    ]

    return EvalConfig(
        suite_name="VLM Benchmark — AiLiveSim Evaluation Suite",
        description=(
            "Comprehensive evaluation of VLMs across 5 scenario categories. "
            "Directly aligned with multimodal evaluation work at AiLiveSim, Turku."
        ),
        models=models,
        scenarios=scenarios,
        output_dir=output_dir,
        batch_size=4,
        seed=42,
        save_predictions=True,
        export_html_report=True,
    )


def print_summary(engine: EvaluationEngine, results: dict):
    """Print a rich summary table to the console."""
    lb = engine.get_leaderboard(results, metric="semantic_similarity")

    table = Table(
        title="📊 Evaluation Results",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Model", style="bold")
    table.add_column("Scenario")
    table.add_column("Task Type", style="dim")
    table.add_column("Samples", justify="right")
    table.add_column("BLEU-4", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("Sem. Sim.", justify="right")

    for _, row in lb.iterrows():
        table.add_row(
            str(row["model"]),
            str(row["scenario"]),
            str(row.get("task_type", "")),
            str(int(row["n_samples"])),
            f"{row.get('bleu_4', 0):.3f}",
            f"{row.get('rouge_l', 0):.3f}",
            f"{row.get('semantic_similarity', 0):.3f}",
        )

    console.print(table)

    # Per-scenario best model
    console.print("\n[bold]Best model per scenario (by Semantic Similarity):[/bold]")
    best_table = Table(show_header=True, header_style="bold green")
    best_table.add_column("Scenario")
    best_table.add_column("Best Model")
    best_table.add_column("Sem. Sim.", justify="right")

    for scenario in lb["scenario"].unique():
        sub = lb[lb["scenario"] == scenario]
        if "semantic_similarity" in sub.columns:
            best = sub.loc[sub["semantic_similarity"].idxmax()]
            best_table.add_row(
                scenario,
                str(best["model"]),
                f"{best['semantic_similarity']:.3f}",
            )

    console.print(best_table)


def main():
    parser = argparse.ArgumentParser(description="VLM Eval Framework Demo")
    parser.add_argument("--n-samples", type=int, default=10, help="Samples per scenario")
    parser.add_argument("--output", default="./eval_results/demo", help="Output directory")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold blue]🔬 VLM Evaluation Framework[/bold blue]\n"
        "Scenario-driven benchmarking for Vision-Language Models\n"
        "[dim]github.com/bashiralam185/vlm-eval-framework[/dim]",
        border_style="blue",
    ))

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Models:      3 mock VLMs")
    console.print(f"  Scenarios:   5 task categories")
    console.print(f"  Samples:     {args.n_samples} per scenario")
    console.print(f"  Output:      {args.output}")

    # Build config
    console.print("\n[bold]Step 1: Building evaluation config...[/bold]")
    config = build_config(n_samples=args.n_samples, output_dir=args.output)
    console.print(f"  ✓ {len(config.models)} models × {len(config.scenarios)} scenarios "
                  f"= {len(config.models) * len(config.scenarios)} evaluation runs")

    # Run
    console.print("\n[bold]Step 2: Running evaluation engine...[/bold]")
    engine = EvaluationEngine(config)
    results = engine.run()

    # Summary
    console.print("\n[bold]Step 3: Results[/bold]")
    print_summary(engine, results)

    # Coherence / correctness check
    console.print("\n[bold]Step 4: Verification[/bold]")
    total_samples = sum(
        sr.n_samples
        for model_results in results.values()
        for sr in model_results
    )
    total_failures = sum(
        sr.n_failed
        for model_results in results.values()
        for sr in model_results
    )
    console.print(f"  Total samples evaluated: {total_samples}")
    console.print(f"  Inference failures: {total_failures}")
    console.print(f"  Success rate: {(1 - total_failures/max(total_samples,1))*100:.1f}%")

    # Output locations
    output_path = config.output_path
    console.print(Panel.fit(
        f"[bold green]✓ Evaluation complete![/bold green]\n\n"
        f"[bold]Outputs:[/bold]\n"
        f"  📊 HTML Report:    [cyan]{output_path}/eval_report.html[/cyan]\n"
        f"  📄 JSON Results:   [cyan]{output_path}/results_summary.json[/cyan]\n"
        f"  📋 CSV Results:    [cyan]{output_path}/results_summary.csv[/cyan]\n"
        f"  🔍 Predictions:    [cyan]{output_path}/predictions/[/cyan]\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"  • Open the HTML report in your browser\n"
        f"  • Replace 'mock' models with real HuggingFace VLMs\n"
        f"  • Use COCO or custom datasets via YAML config\n"
        f"  • Launch the API: [cyan]uvicorn api.main:app --reload[/cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
