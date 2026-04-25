"""
VLM Eval REST API

Endpoints:
  POST /evaluate        Run evaluation from a config dict
  POST /evaluate/file   Run evaluation from an uploaded YAML config
  GET  /results/{id}    Retrieve saved results
  GET  /health          Health check
  GET  /models          List available model backends
  GET  /metrics         List available metrics
  POST /predict         Run a single model on a single image (base64)
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────

class ModelRequest(BaseModel):
    name: str
    hf_model_id: str
    model_type: str = "mock"
    device: str = "auto"
    dtype: str = "float16"
    max_new_tokens: int = 256


class MetricsRequest(BaseModel):
    bleu: bool = True
    rouge: bool = True
    bert_score: bool = False
    clip_score: bool = False
    semantic_similarity: bool = True
    exact_match: bool = False
    accuracy: bool = False


class DatasetRequest(BaseModel):
    name: str
    source: str = "synthetic"
    max_samples: int = 10
    image_dir: Optional[str] = None
    annotation_file: Optional[str] = None


class ScenarioRequest(BaseModel):
    name: str
    task_type: str = "image_captioning"
    prompt_template: str = "Describe this image."
    dataset: DatasetRequest
    metrics: MetricsRequest = Field(default_factory=MetricsRequest)


class EvalRequest(BaseModel):
    suite_name: str = "API Eval"
    models: List[ModelRequest]
    scenarios: List[ScenarioRequest]
    batch_size: int = 4
    save_predictions: bool = True
    export_html_report: bool = True


class PredictRequest(BaseModel):
    model_name: str
    hf_model_id: str
    model_type: str = "mock"
    image_base64: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    prompt: str
    max_new_tokens: int = 256


class JobStatus(BaseModel):
    job_id: str
    status: str
    output_dir: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VLM Eval API",
    description=(
        "REST API for the VLM Evaluation Framework. "
        "Benchmark Vision-Language Models across structured scenario categories."
    ),
    version="0.1.0",
    contact={
        "name": "Bashir Alam",
        "email": "bashir.alam@abo.fi",
        "url": "https://github.com/bashiralam185/vlm-eval-framework",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry (use Redis/DB in production)
_jobs: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _build_eval_config(req: EvalRequest, output_dir: str):
    """Convert API request into EvalConfig."""
    from vlm_eval.core.config import (
        EvalConfig, ModelConfig, ScenarioConfig,
        DatasetConfig, MetricConfig,
    )

    models = [
        ModelConfig(
            name=m.name,
            hf_model_id=m.hf_model_id,
            model_type=m.model_type,
            device=m.device,
            dtype=m.dtype,
            max_new_tokens=m.max_new_tokens,
        )
        for m in req.models
    ]

    scenarios = [
        ScenarioConfig(
            name=s.name,
            task_type=s.task_type,
            prompt_template=s.prompt_template,
            dataset=DatasetConfig(
                name=s.dataset.name,
                source=s.dataset.source,
                max_samples=s.dataset.max_samples,
                image_dir=s.dataset.image_dir,
                annotation_file=s.dataset.annotation_file,
            ),
            metrics=MetricConfig(
                bleu=s.metrics.bleu,
                rouge=s.metrics.rouge,
                bert_score=s.metrics.bert_score,
                clip_score=s.metrics.clip_score,
                semantic_similarity=s.metrics.semantic_similarity,
                exact_match=s.metrics.exact_match,
                accuracy=s.metrics.accuracy,
            ),
        )
        for s in req.scenarios
    ]

    return EvalConfig(
        suite_name=req.suite_name,
        models=models,
        scenarios=scenarios,
        output_dir=output_dir,
        batch_size=req.batch_size,
        save_predictions=req.save_predictions,
        export_html_report=req.export_html_report,
    )


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "healthy", "version": "0.1.0", "active_jobs": len(_jobs)}


@app.get("/models", tags=["System"])
async def list_models():
    from vlm_eval.models.registry import ModelRegistry
    reg = ModelRegistry()
    return {"backends": reg.list_backends()}


@app.get("/metrics", tags=["System"])
async def list_metrics():
    from vlm_eval.metrics.registry import MetricRegistry
    reg = MetricRegistry()
    return {"metrics": reg.list_metrics()}


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate(req: EvalRequest):
    """
    Start a full evaluation run.

    Returns a job_id — poll GET /jobs/{job_id} for status.
    For small evaluations with mock models, runs synchronously.
    """
    from vlm_eval.core.engine import EvaluationEngine

    job_id = str(uuid.uuid4())[:8]
    output_dir = f"./eval_results/api_{job_id}"
    _jobs[job_id] = {"status": "running", "output_dir": output_dir}

    try:
        cfg = _build_eval_config(req, output_dir)
        engine = EvaluationEngine(cfg)
        results = engine.run()

        # Serialize results summary
        summary = {}
        for model_name, sr_list in results.items():
            summary[model_name] = [sr.to_dict() for sr in sr_list]

        _jobs[job_id].update({
            "status": "complete",
            "results": summary,
            "report_path": str(Path(output_dir) / "eval_report.html"),
        })

        return {
            "job_id": job_id,
            "status": "complete",
            "output_dir": output_dir,
            "results": summary,
        }

    except Exception as e:
        _jobs[job_id].update({"status": "failed", "error": str(e)})
        logger.exception(f"Evaluation job {job_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@app.post("/evaluate/file", tags=["Evaluation"])
async def evaluate_from_file(config_file: UploadFile = File(...)):
    """
    Start evaluation from an uploaded YAML config file.
    """
    import tempfile
    from vlm_eval.core.config import EvalConfig
    from vlm_eval.core.engine import EvaluationEngine

    job_id = str(uuid.uuid4())[:8]

    try:
        content = await config_file.read()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        cfg = EvalConfig.from_yaml(tmp_path)
        cfg.output_dir = f"./eval_results/api_{job_id}"

        _jobs[job_id] = {"status": "running", "output_dir": cfg.output_dir}
        engine = EvaluationEngine(cfg)
        results = engine.run()

        summary = {m: [sr.to_dict() for sr in srs] for m, srs in results.items()}
        _jobs[job_id].update({"status": "complete", "results": summary})

        return {"job_id": job_id, "status": "complete", "results": summary}

    except Exception as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", tags=["Evaluation"])
async def get_job(job_id: str):
    """Get status/results of an evaluation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return _jobs[job_id]


@app.get("/jobs/{job_id}/report", tags=["Evaluation"], response_class=HTMLResponse)
async def get_report(job_id: str):
    """Serve the HTML report for a completed job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = _jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    report_path = Path(job.get("report_path", ""))
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return HTMLResponse(content=report_path.read_text(), status_code=200)


@app.post("/predict", tags=["Inference"])
async def predict_single(req: PredictRequest):
    """
    Run a single VLM inference on one image.

    Accepts a base64-encoded image and returns the model's text output.
    """
    from vlm_eval.core.config import ModelConfig
    from vlm_eval.models.registry import ModelRegistry

    try:
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        model_cfg = ModelConfig(
            name=req.model_name,
            hf_model_id=req.hf_model_id,
            model_type=req.model_type,
            max_new_tokens=req.max_new_tokens,
        )
        registry = ModelRegistry()
        model = registry.load(model_cfg)

        import time
        t0 = time.perf_counter()
        prediction = model.generate(image=image, prompt=req.prompt, max_new_tokens=req.max_new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "model": req.model_name,
            "prediction": prediction,
            "latency_ms": round(latency_ms, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", tags=["Evaluation"])
async def list_jobs():
    """List all evaluation jobs."""
    return {
        "jobs": [
            {"job_id": jid, "status": j["status"]}
            for jid, j in _jobs.items()
        ]
    }


def run_server():
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
