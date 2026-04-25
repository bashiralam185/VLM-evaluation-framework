"""Tests for config, scenario, and dataset loading."""

import pytest
from pathlib import Path
from PIL import Image

from vlm_eval.core.config import (
    EvalConfig, ModelConfig, ScenarioConfig, DatasetConfig, MetricConfig,
)
from vlm_eval.core.scenario import (
    Scenario, ScenarioSuite, EvalSample, EvalResult, ScenarioResult,
)
from vlm_eval.datasets.loader import DatasetLoader


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

def make_model_cfg(name="TestModel", model_type="mock"):
    return ModelConfig(name=name, hf_model_id="mock-model", model_type=model_type)

def make_dataset_cfg(source="synthetic", max_samples=5):
    return DatasetConfig(name="test_ds", source=source, max_samples=max_samples)

def make_scenario_cfg(task_type="image_captioning"):
    return ScenarioConfig(
        name="Test Scenario",
        task_type=task_type,
        prompt_template="Describe this image.",
        dataset=make_dataset_cfg(),
        metrics=MetricConfig(bleu=True, rouge=True, clip_score=False, semantic_similarity=True),
    )

def make_eval_config():
    return EvalConfig(
        suite_name="Test Suite",
        models=[make_model_cfg()],
        scenarios=[make_scenario_cfg()],
        output_dir="/tmp/vlm_eval_test",
        export_html_report=False,
        save_predictions=False,
    )


# ─────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────

class TestEvalConfig:

    def test_basic_construction(self):
        cfg = make_eval_config()
        assert cfg.suite_name == "Test Suite"
        assert len(cfg.models) == 1
        assert len(cfg.scenarios) == 1

    def test_to_and_from_yaml(self, tmp_path):
        cfg = make_eval_config()
        path = tmp_path / "test_config.yaml"
        cfg.to_yaml(path)
        assert path.exists()
        loaded = EvalConfig.from_yaml(path)
        assert loaded.suite_name == cfg.suite_name
        assert len(loaded.models) == len(cfg.models)

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EvalConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_get_scenario_by_name(self):
        cfg = make_eval_config()
        s = cfg.get_scenario("Test Scenario")
        assert s is not None
        assert s.name == "Test Scenario"

    def test_get_scenario_missing(self):
        cfg = make_eval_config()
        s = cfg.get_scenario("NonExistent")
        assert s is None

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            ModelConfig(name="X", hf_model_id="x", model_type="mock", device="invalid_device")

    def test_valid_cuda_device(self):
        cfg = ModelConfig(name="X", hf_model_id="x", model_type="mock", device="cuda:0")
        assert cfg.device == "cuda:0"

    def test_output_path_property(self):
        cfg = make_eval_config()
        assert isinstance(cfg.output_path, Path)


# ─────────────────────────────────────────────────────────────────
# Scenario tests
# ─────────────────────────────────────────────────────────────────

class TestScenario:

    def test_prompt_template(self):
        s = Scenario(
            name="Test",
            task_type="image_captioning",
            prompt_template="Describe the {obj} in this image.",
        )
        result = s.build_prompt({"obj": "car"})
        assert result == "Describe the car in this image."

    def test_prompt_template_no_vars(self):
        s = Scenario(name="Test", task_type="vqa", prompt_template="What do you see?")
        assert s.build_prompt() == "What do you see?"

    def test_repr(self):
        s = Scenario(name="Demo", task_type="anomaly_detection", prompt_template="")
        assert "Demo" in repr(s)

    def test_from_yaml(self, tmp_path):
        yaml_content = """
name: Test Scenario
task_type: anomaly_detection
prompt_template: Describe any anomalies.
description: Test desc
tags: [anomaly, test]
"""
        p = tmp_path / "scenario.yaml"
        p.write_text(yaml_content)
        s = Scenario.from_yaml(p)
        assert s.name == "Test Scenario"
        assert s.task_type == "anomaly_detection"
        assert "anomaly" in s.tags


class TestScenarioSuite:

    def _make_suite(self):
        scenarios = [
            Scenario("A", "anomaly_detection", "", tags=["anomaly"]),
            Scenario("B", "scene_understanding", "", tags=["scene"]),
            Scenario("C", "anomaly_detection", "", tags=["anomaly", "safety"]),
        ]
        return ScenarioSuite("Test Suite", scenarios)

    def test_len(self):
        suite = self._make_suite()
        assert len(suite) == 3

    def test_filter_by_tag(self):
        suite = self._make_suite()
        filtered = suite.filter_by_tag("anomaly")
        assert len(filtered) == 2

    def test_filter_by_task(self):
        suite = self._make_suite()
        filtered = suite.filter_by_task("scene_understanding")
        assert len(filtered) == 1

    def test_iteration(self):
        suite = self._make_suite()
        names = [s.name for s in suite]
        assert names == ["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────
# Scenario result tests
# ─────────────────────────────────────────────────────────────────

class TestScenarioResult:

    def _make_result(self):
        sr = ScenarioResult("Test", "MockVLM", "image_captioning")
        for i in range(5):
            sr.sample_results.append(EvalResult(
                sample_id=f"s{i}",
                model_name="MockVLM",
                prediction="A test image.",
                scores={"bleu_1": 0.5 + i * 0.05, "rouge_l": 0.4 + i * 0.03},
            ))
        return sr

    def test_n_samples(self):
        sr = self._make_result()
        assert sr.n_samples == 5

    def test_n_failed(self):
        sr = self._make_result()
        sr.sample_results[0].error = "timeout"
        assert sr.n_failed == 1

    def test_aggregate_scores(self):
        sr = self._make_result()
        agg = sr.aggregate_scores()
        assert "bleu_1" in agg
        assert "rouge_l" in agg
        assert 0 < agg["bleu_1"] < 1

    def test_to_dict(self):
        sr = self._make_result()
        d = sr.to_dict()
        assert d["scenario"] == "Test"
        assert d["model"] == "MockVLM"
        assert "scores" in d


# ─────────────────────────────────────────────────────────────────
# Dataset loader tests
# ─────────────────────────────────────────────────────────────────

class TestDatasetLoader:

    @pytest.fixture
    def loader(self):
        return DatasetLoader()

    @pytest.fixture
    def scenario_cfg(self):
        return make_scenario_cfg()

    def test_load_synthetic_captioning(self, loader, scenario_cfg):
        cfg = make_dataset_cfg(source="synthetic", max_samples=8)
        samples = loader.load(cfg, scenario_cfg, max_samples=8)
        assert len(samples) == 8
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_load_synthetic_anomaly(self, loader):
        s_cfg = make_scenario_cfg(task_type="anomaly_detection")
        cfg = make_dataset_cfg(source="synthetic", max_samples=5)
        samples = loader.load(cfg, s_cfg)
        assert len(samples) == 5
        for s in samples:
            assert isinstance(s.image, Image.Image)
            assert len(s.references) > 0
            assert s.prompt == "Describe this image."

    @pytest.mark.parametrize("task_type", [
        "anomaly_detection",
        "scene_understanding",
        "image_captioning",
        "visual_question_answering",
        "object_recognition",
        "safety_critical",
    ])
    def test_all_task_types(self, loader, task_type):
        s_cfg = make_scenario_cfg(task_type=task_type)
        cfg = make_dataset_cfg(source="synthetic", max_samples=3)
        samples = loader.load(cfg, s_cfg)
        assert len(samples) == 3

    def test_max_samples_respected(self, loader, scenario_cfg):
        cfg = make_dataset_cfg(source="synthetic", max_samples=50)
        samples = loader.load(cfg, scenario_cfg, max_samples=7)
        assert len(samples) == 7

    def test_sample_has_required_fields(self, loader, scenario_cfg):
        cfg = make_dataset_cfg(source="synthetic", max_samples=1)
        samples = loader.load(cfg, scenario_cfg)
        s = samples[0]
        assert s.sample_id
        assert s.image is not None
        assert s.prompt
        assert isinstance(s.references, list)
        assert isinstance(s.metadata, dict)

    def test_custom_source_fallback(self, loader, scenario_cfg):
        """Custom source with missing dir should fall back to synthetic."""
        cfg = DatasetConfig(
            name="custom_test", source="custom",
            image_dir="/nonexistent/path/images",
            max_samples=3,
        )
        samples = loader.load(cfg, scenario_cfg)
        # Falls back to synthetic — should still return samples
        assert isinstance(samples, list)

    def test_unknown_source_raises(self, loader, scenario_cfg):
        cfg = DatasetConfig(name="bad", source="nonexistent_source")
        with pytest.raises(ValueError, match="Unknown dataset source"):
            loader.load(cfg, scenario_cfg)
