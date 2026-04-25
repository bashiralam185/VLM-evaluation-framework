"""Tests for the FastAPI REST API."""

import base64
import io
import json
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def make_base64_image(w=64, h=64) -> str:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def make_eval_request(n_samples=3):
    return {
        "suite_name": "API Test Suite",
        "models": [
            {"name": "MockA", "hf_model_id": "mock-a", "model_type": "mock"},
            {"name": "MockB", "hf_model_id": "mock-b", "model_type": "mock"},
        ],
        "scenarios": [
            {
                "name": "Anomaly Detection",
                "task_type": "anomaly_detection",
                "prompt_template": "Describe any anomalies.",
                "dataset": {"name": "ds1", "source": "synthetic", "max_samples": n_samples},
                "metrics": {"bleu": True, "rouge": True, "semantic_similarity": True, "clip_score": False},
            },
            {
                "name": "Scene Understanding",
                "task_type": "scene_understanding",
                "prompt_template": "Describe this scene.",
                "dataset": {"name": "ds2", "source": "synthetic", "max_samples": n_samples},
                "metrics": {"bleu": True, "rouge": True, "semantic_similarity": True, "clip_score": False},
            },
        ],
        "batch_size": 4,
        "save_predictions": False,
        "export_html_report": True,
    }


class TestSystemEndpoints:

    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_list_models(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "backends" in data
        assert "mock" in data["backends"]

    def test_list_metrics(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "bleu_1" in data["metrics"]
        assert "semantic_similarity" in data["metrics"]


class TestEvaluationEndpoints:

    def test_evaluate_basic(self):
        resp = client.post("/evaluate", json=make_eval_request(n_samples=3))
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "complete"
        assert "results" in data

    def test_evaluate_returns_all_models(self):
        resp = client.post("/evaluate", json=make_eval_request(n_samples=3))
        data = resp.json()
        results = data["results"]
        assert "MockA" in results
        assert "MockB" in results

    def test_evaluate_returns_all_scenarios(self):
        resp = client.post("/evaluate", json=make_eval_request(n_samples=3))
        data = resp.json()
        for model_name, scenario_results in data["results"].items():
            scenario_names = [sr["scenario"] for sr in scenario_results]
            assert "Anomaly Detection" in scenario_names
            assert "Scene Understanding" in scenario_names

    def test_list_jobs_after_eval(self):
        client.post("/evaluate", json=make_eval_request(n_samples=2))
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert "jobs" in resp.json()
        assert len(resp.json()["jobs"]) > 0

    def test_get_job_by_id(self):
        resp = client.post("/evaluate", json=make_eval_request(n_samples=2))
        job_id = resp.json()["job_id"]
        resp2 = client.get(f"/jobs/{job_id}")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "complete"

    def test_get_nonexistent_job(self):
        resp = client.get("/jobs/nonexistent_job_xyz")
        assert resp.status_code == 404


class TestPredictEndpoint:

    def test_predict_mock_model(self):
        img_b64 = make_base64_image()
        resp = client.post("/predict", json={
            "model_name": "MockVLM",
            "hf_model_id": "mock-model",
            "model_type": "mock",
            "image_base64": img_b64,
            "prompt": "Describe this image.",
            "max_new_tokens": 128,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "latency_ms" in data
        assert isinstance(data["prediction"], str)
        assert len(data["prediction"]) > 0

    def test_predict_invalid_image(self):
        resp = client.post("/predict", json={
            "model_name": "MockVLM",
            "hf_model_id": "mock-model",
            "model_type": "mock",
            "image_base64": "not_valid_base64!!!",
            "prompt": "Describe this.",
        })
        assert resp.status_code == 400

    def test_predict_returns_latency(self):
        img_b64 = make_base64_image()
        resp = client.post("/predict", json={
            "model_name": "MockVLM",
            "hf_model_id": "mock-model",
            "model_type": "mock",
            "image_base64": img_b64,
            "prompt": "What do you see?",
        })
        assert resp.status_code == 200
        assert resp.json()["latency_ms"] >= 0
