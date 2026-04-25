"""Tests for metric computation functions."""

import pytest
from PIL import Image
import numpy as np

from vlm_eval.core.config import MetricConfig
from vlm_eval.metrics.registry import (
    MetricRegistry,
    compute_bleu,
    compute_rouge,
    compute_exact_match,
    compute_accuracy,
    compute_semantic_similarity,
)


def dummy_image(w=224, h=224) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestBLEU:

    def test_perfect_match(self):
        scores = compute_bleu("a cat sat on the mat", ["a cat sat on the mat"])
        assert scores["bleu_1"] > 0.9
        assert scores["bleu_4"] > 0.5

    def test_empty_prediction(self):
        scores = compute_bleu("", ["a cat sat on the mat"])
        assert scores["bleu_1"] == 0.0

    def test_no_overlap(self):
        scores = compute_bleu("dog barked loudly", ["cat sat quietly"])
        assert scores["bleu_1"] < 0.3

    def test_multiple_references(self):
        scores = compute_bleu(
            "a cat on the mat",
            ["a cat sat on the mat", "feline resting on a rug"],
        )
        assert scores["bleu_1"] > 0.0

    def test_returns_all_ngrams(self):
        scores = compute_bleu("test prediction", ["test reference"])
        assert "bleu_1" in scores
        assert "bleu_2" in scores
        assert "bleu_4" in scores


class TestROUGE:

    def test_perfect_match(self):
        scores = compute_rouge("a cat sat on the mat", ["a cat sat on the mat"])
        assert scores["rouge_1"] > 0.9

    def test_empty_prediction(self):
        scores = compute_rouge("", ["some reference text"])
        assert scores["rouge_1"] == 0.0

    def test_partial_overlap(self):
        scores = compute_rouge("cat on the mat", ["a cat sat on the mat"])
        assert 0.0 < scores["rouge_1"] < 1.0

    def test_returns_all_variants(self):
        scores = compute_rouge("test", ["test"])
        assert "rouge_1" in scores
        assert "rouge_2" in scores
        assert "rouge_l" in scores


class TestExactMatch:

    def test_exact_match_true(self):
        assert compute_exact_match("yes", ["yes"]) == 1.0

    def test_exact_match_false(self):
        assert compute_exact_match("no", ["yes"]) == 0.0

    def test_case_insensitive(self):
        assert compute_exact_match("YES", ["yes"]) == 1.0

    def test_whitespace_stripped(self):
        assert compute_exact_match("  yes  ", ["yes"]) == 1.0

    def test_multiple_references(self):
        assert compute_exact_match("cat", ["dog", "cat", "bird"]) == 1.0


class TestAccuracy:

    def test_contains_answer(self):
        assert compute_accuracy("There is a cat in the image.", ["cat"]) == 1.0

    def test_does_not_contain(self):
        assert compute_accuracy("There is a dog.", ["cat"]) == 0.0

    def test_case_insensitive(self):
        assert compute_accuracy("A CAT is visible.", ["cat"]) == 1.0


class TestSemanticSimilarity:

    def test_identical_strings(self):
        sim = compute_semantic_similarity("a cat on the mat", ["a cat on the mat"])
        assert sim > 0.9

    def test_similar_strings(self):
        sim = compute_semantic_similarity(
            "a cat sitting on the floor",
            ["a feline resting on the ground"],
        )
        # Should be moderately similar
        assert sim > 0.0

    def test_empty_prediction(self):
        sim = compute_semantic_similarity("", ["reference text"])
        assert isinstance(sim, float)

    def test_returns_float(self):
        sim = compute_semantic_similarity("test", ["test reference"])
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0


class TestMetricRegistry:

    @pytest.fixture
    def registry(self):
        return MetricRegistry()

    @pytest.fixture
    def image(self):
        return dummy_image()

    def test_compute_bleu_rouge(self, registry, image):
        cfg = MetricConfig(
            bleu=True, rouge=True,
            clip_score=False, bert_score=False, semantic_similarity=False,
        )
        scores = registry.compute(
            prediction="A cat sitting on a mat.",
            references=["A cat on a mat."],
            image=image,
            metric_cfg=cfg,
        )
        assert "bleu_1" in scores
        assert "rouge_l" in scores
        assert "clip_score" not in scores

    def test_compute_semantic_similarity(self, registry, image):
        cfg = MetricConfig(
            bleu=False, rouge=False,
            clip_score=False, bert_score=False, semantic_similarity=True,
        )
        scores = registry.compute(
            prediction="Test prediction.",
            references=["Test reference."],
            image=image,
            metric_cfg=cfg,
        )
        assert "semantic_similarity" in scores
        assert isinstance(scores["semantic_similarity"], float)

    def test_compute_no_metrics(self, registry, image):
        cfg = MetricConfig(
            bleu=False, rouge=False, clip_score=False,
            bert_score=False, semantic_similarity=False,
        )
        scores = registry.compute("test", ["ref"], image, cfg)
        assert scores == {}

    def test_list_metrics(self, registry):
        metrics = registry.list_metrics()
        assert "bleu_1" in metrics
        assert "rouge_l" in metrics
        assert "clip_score" in metrics
        assert "semantic_similarity" in metrics

    def test_empty_references(self, registry, image):
        cfg = MetricConfig(bleu=True, rouge=True, semantic_similarity=True)
        # Should not raise
        scores = registry.compute("test prediction", [], image, cfg)
        assert isinstance(scores, dict)
