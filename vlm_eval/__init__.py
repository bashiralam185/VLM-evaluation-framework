"""
VLM Eval — Vision-Language Model Evaluation Framework
======================================================
A modular, scenario-driven benchmarking toolkit for evaluating
Vision-Language Models across structured task categories.

Author : Bashir Alam <bashir.alam@abo.fi>
License: MIT

Built from experience evaluating large multimodal models at AiLiveSim,
Turku, Finland (Nov 2024 – May 2025).
"""

from vlm_eval.core.engine import EvaluationEngine
from vlm_eval.core.scenario import Scenario, ScenarioSuite
from vlm_eval.core.config import EvalConfig
from vlm_eval.models.registry import ModelRegistry
from vlm_eval.metrics.registry import MetricRegistry

__version__ = "0.1.0"
__author__ = "Bashir Alam"
__email__ = "bashir.alam@abo.fi"

__all__ = [
    "EvaluationEngine",
    "Scenario",
    "ScenarioSuite",
    "EvalConfig",
    "ModelRegistry",
    "MetricRegistry",
]
