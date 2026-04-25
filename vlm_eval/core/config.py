"""
EvalConfig: Central configuration system.

Supports loading from YAML files, env vars, and direct construction.
All evaluation runs are fully reproducible via a single config file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for a single VLM to evaluate."""
    name: str = Field(..., description="Display name, e.g. 'LLaVA-1.5-7B'")
    hf_model_id: str = Field(..., description="HuggingFace model ID")
    model_type: Literal["hf_vlm", "openai", "mock"] = Field(default="hf_vlm")
    device: str = Field(default="auto")
    dtype: str = Field(default="float16")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.0)
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid = {"auto", "cpu", "cuda", "mps"}
        if v not in valid and not v.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid} or 'cuda:N'")
        return v


class MetricConfig(BaseModel):
    """Which metrics to compute per scenario."""
    bleu: bool = True
    rouge: bool = True
    bert_score: bool = True
    clip_score: bool = True
    semantic_similarity: bool = True
    exact_match: bool = False
    accuracy: bool = False


class DatasetConfig(BaseModel):
    """Dataset / scenario-level configuration."""
    name: str
    source: Literal["coco", "custom", "synthetic", "huggingface"] = "synthetic"
    split: str = "val"
    max_samples: Optional[int] = Field(default=None)
    image_dir: Optional[str] = None
    annotation_file: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    cache_dir: str = Field(default="~/.cache/vlm_eval/datasets")


class ScenarioConfig(BaseModel):
    """Full configuration for one evaluation scenario."""
    name: str
    description: str = ""
    task_type: Literal[
        "image_captioning",
        "visual_question_answering",
        "scene_understanding",
        "anomaly_detection",
        "object_recognition",
        "safety_critical",
    ] = "image_captioning"
    prompt_template: str = Field(
        default="Describe this image in detail.",
        description="Prompt sent to the VLM for each image.",
    )
    reference_field: str = Field(
        default="captions",
        description="Which field in the dataset provides ground-truth references.",
    )
    dataset: DatasetConfig
    metrics: MetricConfig = Field(default_factory=MetricConfig)
    tags: List[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    """
    Top-level evaluation configuration.

    Can be loaded from a YAML file:
        config = EvalConfig.from_yaml("configs/scenarios/my_suite.yaml")

    Or constructed directly:
        config = EvalConfig(
            suite_name="My Benchmark",
            models=[ModelConfig(name="LLaVA", hf_model_id="llava-hf/llava-1.5-7b-hf")],
            scenarios=[...],
        )
    """

    suite_name: str = Field(..., description="Human-readable name for this evaluation suite.")
    description: str = ""
    version: str = "0.1.0"

    models: List[ModelConfig] = Field(..., min_length=1)
    scenarios: List[ScenarioConfig] = Field(..., min_length=1)

    output_dir: str = Field(default="./eval_results")
    batch_size: int = Field(default=4)
    num_workers: int = Field(default=2)
    seed: int = Field(default=42)
    use_triton: bool = Field(default=False)
    triton_url: str = Field(default="localhost:8001")
    save_predictions: bool = Field(default=True)
    export_html_report: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path):
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @property
    def output_path(self) -> Path:
        return Path(os.path.expanduser(self.output_dir))

    def get_scenario(self, name: str) -> Optional[ScenarioConfig]:
        for s in self.scenarios:
            if s.name == name:
                return s
        return None
