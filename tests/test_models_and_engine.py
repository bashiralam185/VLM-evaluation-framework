"""Tests for model registry and full pipeline integration."""

import pytest
from PIL import Image
import numpy as np

from vlm_eval.models.registry import ModelRegistry, MockVLM
from vlm_eval.core.config import (
    EvalConfig, ModelConfig, ScenarioConfig, DatasetConfig, MetricConfig,
)
from vlm_eval.core.engine import EvaluationEngine


def dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_mock_model_cfg(name="MockVLM-Test", seed=42):
    return ModelConfig(
        name=name,
        hf_model_id="mock-model",
        model_type="mock",
    )


def make_full_config(output_dir="/tmp/vlm_eval_test", n_samples=5):
    return EvalConfig(
        suite_name="Integration Test Suite",
        models=[
            ModelConfig(name="MockA", hf_model_id="mock-a", model_type="mock"),
            ModelConfig(name="MockB", hf_model_id="mock-b", model_type="mock"),
        ],
        scenarios=[
            ScenarioConfig(
                name="Anomaly Detection",
                task_type="anomaly_detection",
                prompt_template="Describe any anomalies in this image.",
                dataset=DatasetConfig(name="ds1", source="synthetic", max_samples=n_samples),
                metrics=MetricConfig(
                    bleu=True, rouge=True, clip_score=False,
                    bert_score=False, semantic_similarity=True,
                ),
            ),
            ScenarioConfig(
                name="Scene Understanding",
                task_type="scene_understanding",
                prompt_template="Describe this scene.",
                dataset=DatasetConfig(name="ds2", source="synthetic", max_samples=n_samples),
                metrics=MetricConfig(
                    bleu=True, rouge=True, clip_score=False,
                    bert_score=False, semantic_similarity=True,
                ),
            ),
        ],
        output_dir=output_dir,
        batch_size=4,
        save_predictions=True,
        export_html_report=True,
    )


# ─────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────

class TestMockVLM:

    def test_generate_returns_string(self):
        cfg = make_mock_model_cfg()
        model = MockVLM(cfg)
        result = model.generate(dummy_image(), "Describe this image.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_anomaly_prompt(self):
        cfg = make_mock_model_cfg()
        model = MockVLM(cfg)
        result = model.generate(dummy_image(), "Is there an anomaly in this image?")
        assert isinstance(result, str)

    def test_generate_vqa_prompt(self):
        cfg = make_mock_model_cfg()
        model = MockVLM(cfg)
        result = model.generate(dummy_image(), "What color is the main object?")
        assert isinstance(result, str)

    def test_deterministic_with_same_seed(self):
        cfg = make_mock_model_cfg()
        model1 = MockVLM(cfg, seed=0)
        model2 = MockVLM(cfg, seed=0)
        img = dummy_image()
        r1 = model1.generate(img, "Describe this image.")
        r2 = model2.generate(img, "Describe this image.")
        assert r1 == r2

    def test_unload(self):
        cfg = make_mock_model_cfg()
        model = MockVLM(cfg)
        model.unload()
        assert not model._loaded


class TestModelRegistry:

    def test_load_mock_model(self):
        registry = ModelRegistry()
        cfg = make_mock_model_cfg()
        model = registry.load(cfg)
        assert model is not None
        assert model.name == "MockVLM-Test"

    def test_returns_cached_instance(self):
        registry = ModelRegistry()
        cfg = make_mock_model_cfg()
        m1 = registry.load(cfg)
        m2 = registry.load(cfg)
        assert m1 is m2  # Same instance

    def test_invalid_model_type_raises(self):
        registry = ModelRegistry()
        cfg = ModelConfig(name="X", hf_model_id="x", model_type="nonexistent_backend")
        with pytest.raises(ValueError, match="Unknown model_type"):
            registry.load(cfg)

    def test_list_backends(self):
        registry = ModelRegistry()
        backends = registry.list_backends()
        assert "mock" in backends
        assert "hf_vlm" in backends
        assert "triton" in backends

    def test_register_custom_backend(self):
        registry = ModelRegistry()

        class MyCustomModel(MockVLM):
            pass

        registry.register("custom", MyCustomModel)
        assert "custom" in registry.list_backends()

    def test_clear_cache(self):
        registry = ModelRegistry()
        cfg = make_mock_model_cfg()
        registry.load(cfg)
        assert len(registry._cache) == 1
        registry.clear_cache()
        assert len(registry._cache) == 0


# ─────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────

class TestEvaluationEngine:

    @pytest.fixture
    def config(self, tmp_path):
        cfg = make_full_config(output_dir=str(tmp_path / "results"), n_samples=5)
        return cfg

    def test_engine_runs_without_error(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        assert isinstance(results, dict)

    def test_results_have_all_models(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        assert "MockA" in results
        assert "MockB" in results

    def test_results_have_all_scenarios(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        for model_name in results:
            scenario_names = [sr.scenario_name for sr in results[model_name]]
            assert "Anomaly Detection" in scenario_names
            assert "Scene Understanding" in scenario_names

    def test_correct_sample_count(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        for model_name, sr_list in results.items():
            for sr in sr_list:
                assert sr.n_samples == 5

    def test_scores_computed(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        for model_name, sr_list in results.items():
            for sr in sr_list:
                agg = sr.aggregate_scores()
                assert len(agg) > 0
                assert "bleu_1" in agg or "rouge_l" in agg or "semantic_similarity" in agg

    def test_output_files_created(self, config):
        engine = EvaluationEngine(config)
        engine.run()
        output = config.output_path
        assert (output / "results_summary.json").exists()
        assert (output / "results_summary.csv").exists()

    def test_predictions_saved(self, config):
        config.save_predictions = True
        engine = EvaluationEngine(config)
        engine.run()
        pred_dir = config.output_path / "predictions"
        assert pred_dir.exists()
        assert len(list(pred_dir.glob("*.jsonl"))) > 0

    def test_html_report_generated(self, config):
        config.export_html_report = True
        engine = EvaluationEngine(config)
        engine.run()
        report = config.output_path / "eval_report.html"
        assert report.exists()
        content = report.read_text()
        assert "VLM Eval" in content or "vlm" in content.lower()

    def test_leaderboard(self, config):
        engine = EvaluationEngine(config)
        results = engine.run()
        lb = engine.get_leaderboard(results, metric="semantic_similarity")
        assert len(lb) > 0
        assert "model" in lb.columns
        assert "scenario" in lb.columns

    def test_run_single_scenario(self, config):
        engine = EvaluationEngine(config)
        result = engine.run_single_scenario("MockA", "Anomaly Detection")
        assert result.scenario_name == "Anomaly Detection"
        assert result.model_name == "MockA"
        assert result.n_samples > 0

    def test_run_single_missing_model_raises(self, config):
        engine = EvaluationEngine(config)
        with pytest.raises(ValueError, match="not found"):
            engine.run_single_scenario("NonExistent", "Anomaly Detection")

    def test_run_single_missing_scenario_raises(self, config):
        engine = EvaluationEngine(config)
        with pytest.raises(ValueError, match="not found"):
            engine.run_single_scenario("MockA", "NonExistent Scenario")
