"""
Scenario: The fundamental unit of evaluation.

A Scenario = one task type (e.g. anomaly_detection) + dataset + prompts + metrics.
A ScenarioSuite = multiple scenarios bundled together for a complete benchmark run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class EvalSample:
    """
    A single evaluation sample: one image + prompt + optional reference answers.

    Attributes
    ----------
    sample_id : str
        Unique identifier (e.g. COCO image_id or file stem).
    image_path : Path | None
        Path to the image file (None if image is loaded from bytes/URL).
    image : Any
        PIL Image object (populated by the dataset loader).
    prompt : str
        The text prompt sent to the VLM.
    references : list of str
        Ground-truth reference answers (can be multiple).
    metadata : dict
        Extra fields: category, split, tags, annotation IDs, etc.
    """

    sample_id: str
    image_path: Optional[Path]
    image: Any  # PIL.Image.Image
    prompt: str
    references: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvalSample(id={self.sample_id!r}, "
            f"prompt={self.prompt[:40]!r}..., "
            f"refs={len(self.references)})"
        )


@dataclass
class EvalResult:
    """
    The output of running one model on one sample.

    Attributes
    ----------
    sample_id : str
    model_name : str
    prediction : str
        The VLM's raw text output.
    scores : dict
        Computed metric scores, e.g. {"bleu": 0.42, "clip_score": 72.3}.
    latency_ms : float
        Inference latency in milliseconds.
    error : str | None
        Error message if inference failed.
    """

    sample_id: str
    model_name: str
    prediction: str
    scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class ScenarioResult:
    """
    Aggregated results for one (scenario, model) pair.
    """

    scenario_name: str
    model_name: str
    task_type: str
    sample_results: List[EvalResult] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.sample_results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.sample_results if r.failed)

    def aggregate_scores(self) -> Dict[str, float]:
        """Compute mean score for each metric across all samples."""
        if not self.sample_results:
            return {}
        all_metrics: Dict[str, List[float]] = {}
        for r in self.sample_results:
            if r.failed:
                continue
            for k, v in r.scores.items():
                all_metrics.setdefault(k, []).append(v)
        return {k: sum(vals) / len(vals) for k, vals in all_metrics.items() if vals}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "model": self.model_name,
            "task_type": self.task_type,
            "n_samples": self.n_samples,
            "n_failed": self.n_failed,
            "scores": self.aggregate_scores(),
        }


class Scenario:
    """
    A single evaluation scenario: wraps config + provides iteration over samples.

    Can be defined via YAML or programmatically:

    >>> from vlm_eval.core.scenario import Scenario
    >>> scenario = Scenario.from_yaml("configs/scenarios/anomaly_detection.yaml")
    """

    def __init__(
        self,
        name: str,
        task_type: str,
        prompt_template: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.task_type = task_type
        self.prompt_template = prompt_template
        self.description = description
        self.tags = tags or []

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            task_type=data.get("task_type", "image_captioning"),
            prompt_template=data.get("prompt_template", "Describe this image."),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )

    def build_prompt(self, sample_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Fill prompt template with sample-level metadata if needed."""
        if sample_metadata is None:
            return self.prompt_template
        try:
            return self.prompt_template.format(**sample_metadata)
        except KeyError:
            return self.prompt_template

    def __repr__(self) -> str:
        return f"Scenario(name={self.name!r}, task={self.task_type!r})"


class ScenarioSuite:
    """
    A collection of scenarios forming a complete benchmark.

    Load from YAML:
        suite = ScenarioSuite.from_yaml("configs/scenarios/full_suite.yaml")

    Or build programmatically:
        suite = ScenarioSuite(name="My Benchmark", scenarios=[...])
    """

    def __init__(self, name: str, scenarios: List[Scenario], description: str = ""):
        self.name = name
        self.scenarios = scenarios
        self.description = description

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioSuite":
        with open(path) as f:
            data = yaml.safe_load(f)
        scenarios = [
            Scenario(
                name=s["name"],
                task_type=s.get("task_type", "image_captioning"),
                prompt_template=s.get("prompt_template", "Describe this image."),
                description=s.get("description", ""),
                tags=s.get("tags", []),
            )
            for s in data.get("scenarios", [])
        ]
        return cls(
            name=data.get("suite_name", "Unnamed Suite"),
            scenarios=scenarios,
            description=data.get("description", ""),
        )

    def filter_by_tag(self, tag: str) -> "ScenarioSuite":
        filtered = [s for s in self.scenarios if tag in s.tags]
        return ScenarioSuite(
            name=f"{self.name} [{tag}]",
            scenarios=filtered,
            description=self.description,
        )

    def filter_by_task(self, task_type: str) -> "ScenarioSuite":
        filtered = [s for s in self.scenarios if s.task_type == task_type]
        return ScenarioSuite(name=self.name, scenarios=filtered)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def __repr__(self) -> str:
        tasks = list({s.task_type for s in self.scenarios})
        return f"ScenarioSuite(name={self.name!r}, n={len(self)}, tasks={tasks})"
