"""
VLM Eval CLI

Commands:
  vlm-eval run      Run a full evaluation suite from a YAML config
  vlm-eval validate Validate a YAML config without running
  vlm-eval list     List available model backends and metrics
  vlm-eval report   Re-generate HTML report from saved JSON results
  vlm-eval demo     Run a quick demo with mock models
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="vlm-eval",
    help="🔬 VLM Evaluation Framework — Benchmark Vision-Language Models",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to YAML evaluation config"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Override output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
    scenario: Optional[str] = typer.Option(None, "--scenario", "-s", help="Run only this scenario"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Run only this model"),
):
    """Run a full VLM evaluation suite from a YAML config file."""
    from vlm_eval.core.config import EvalConfig
    from vlm_eval.core.engine import EvaluationEngine

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Loading config:[/bold blue] {config}")
    cfg = EvalConfig.from_yaml(config)

    if output_dir:
        cfg.output_dir = output_dir

    # Filter if requested
    if model:
        cfg.models = [m for m in cfg.models if m.name == model]
        if not cfg.models:
            console.print(f"[red]Model '{model}' not found in config.[/red]")
            raise typer.Exit(1)

    if scenario:
        cfg.scenarios = [s for s in cfg.scenarios if s.name == scenario]
        if not cfg.scenarios:
            console.print(f"[red]Scenario '{scenario}' not found in config.[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Suite:[/green] {cfg.suite_name}")
    console.print(f"[green]Models:[/green] {[m.name for m in cfg.models]}")
    console.print(f"[green]Scenarios:[/green] {[s.name for s in cfg.scenarios]}")

    if dry_run:
        console.print("[yellow]Dry run — config valid, exiting.[/yellow]")
        return

    engine = EvaluationEngine(cfg)
    results = engine.run()

    # Print summary
    table = Table(title="Results Summary")
    table.add_column("Model", style="bold")
    table.add_column("Scenario")
    table.add_column("Samples", justify="right")
    table.add_column("BLEU-4", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("CLIPScore", justify="right")

    for m_name, scenario_results in results.items():
        for sr in scenario_results:
            agg = sr.aggregate_scores()
            table.add_row(
                m_name,
                sr.scenario_name,
                str(sr.n_samples),
                f"{agg.get('bleu_4', 0):.3f}",
                f"{agg.get('rouge_l', 0):.3f}",
                f"{agg.get('clip_score', 0):.1f}",
            )

    console.print(table)
    console.print(f"\n[green]✓ Results saved to:[/green] {cfg.output_path}")


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to YAML config to validate"),
):
    """Validate a YAML config file without running any models."""
    from vlm_eval.core.config import EvalConfig

    try:
        cfg = EvalConfig.from_yaml(config)
        console.print(f"[green]✓ Config valid:[/green] {cfg.suite_name}")
        console.print(f"  Models: {[m.name for m in cfg.models]}")
        console.print(f"  Scenarios: {[s.name for s in cfg.scenarios]}")
    except Exception as e:
        console.print(f"[red]✗ Config invalid: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_backends():
    """List available model backends and supported metrics."""
    from vlm_eval.models.registry import ModelRegistry
    from vlm_eval.metrics.registry import MetricRegistry

    reg = ModelRegistry()
    m_reg = MetricRegistry()

    console.print("\n[bold]Available model backends:[/bold]")
    for backend in reg.list_backends():
        console.print(f"  • {backend}")

    console.print("\n[bold]Available metrics:[/bold]")
    for metric in m_reg.list_metrics():
        console.print(f"  • {metric}")


@app.command()
def demo(
    output_dir: str = typer.Option("./demo_results", "--output", "-o"),
    n_samples: int = typer.Option(10, "--samples", "-n"),
    n_models: int = typer.Option(2, "--models", help="Number of mock models to compare"),
):
    """
    Run a quick demo evaluation with mock models and synthetic data.
    No GPU or model download required.
    """
    from vlm_eval.core.config import (
        EvalConfig, ModelConfig, ScenarioConfig,
        DatasetConfig, MetricConfig,
    )
    from vlm_eval.core.engine import EvaluationEngine

    console.print("[bold blue]🚀 Running VLM Eval Demo (mock models, synthetic data)[/bold blue]")

    models = [
        ModelConfig(
            name=f"MockVLM-{chr(65+i)}",
            hf_model_id=f"mock-model-{i}",
            model_type="mock",
        )
        for i in range(n_models)
    ]

    scenarios = [
        ScenarioConfig(
            name="Anomaly Detection",
            task_type="anomaly_detection",
            prompt_template="Describe any unusual or anomalous elements you observe in this image.",
            description="Evaluate ability to detect anomalies in road/traffic scenes.",
            dataset=DatasetConfig(
                name="synthetic_anomaly",
                source="synthetic",
                max_samples=n_samples,
            ),
            metrics=MetricConfig(bleu=True, rouge=True, semantic_similarity=True, clip_score=False),
            tags=["anomaly", "safety"],
        ),
        ScenarioConfig(
            name="Scene Understanding",
            task_type="scene_understanding",
            prompt_template="Describe the scene in this image, including environment type and key elements.",
            description="Evaluate scene-level understanding across diverse environments.",
            dataset=DatasetConfig(
                name="synthetic_scene",
                source="synthetic",
                max_samples=n_samples,
            ),
            metrics=MetricConfig(bleu=True, rouge=True, semantic_similarity=True, clip_score=False),
            tags=["scene", "captioning"],
        ),
        ScenarioConfig(
            name="Object Recognition",
            task_type="object_recognition",
            prompt_template="What objects do you see in this image? List the main objects.",
            description="Evaluate object identification accuracy.",
            dataset=DatasetConfig(
                name="synthetic_objects",
                source="synthetic",
                max_samples=n_samples,
            ),
            metrics=MetricConfig(bleu=True, rouge=True, semantic_similarity=True, clip_score=False),
            tags=["objects"],
        ),
    ]

    cfg = EvalConfig(
        suite_name="VLM Eval Demo",
        description="Quick demo using mock models and synthetic data",
        models=models,
        scenarios=scenarios,
        output_dir=output_dir,
        batch_size=4,
        save_predictions=True,
        export_html_report=True,
    )

    engine = EvaluationEngine(cfg)
    results = engine.run()

    # Leaderboard
    lb = engine.get_leaderboard(results, metric="semantic_similarity")
    table = Table(title="Demo Leaderboard")
    for col in lb.columns:
        table.add_column(str(col))
    for _, row in lb.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)

    console.print(f"\n[green]✓ Demo complete! Open:[/green] {output_dir}/eval_report.html")


if __name__ == "__main__":
    app()
