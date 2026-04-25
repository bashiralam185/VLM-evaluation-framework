# Changelog

## [0.1.0] — 2025-05-01

### Added
- `EvalConfig` — YAML-based configuration system with Pydantic validation
- `EvaluationEngine` — Central orchestrator: load → infer → evaluate → report
- `ScenarioSuite` — Multi-scenario benchmark bundling with tag/task filtering
- Dataset loaders: `synthetic`, `coco`, `huggingface`, `custom`
- Metrics: BLEU-1/2/4, ROUGE-1/2/L, BERTScore F1, CLIPScore, Semantic Similarity, Exact Match, Accuracy
- Model backends: `HuggingFaceVLM` (generic), `MockVLM` (testing), `TritonVLM` (GPU serving)
- `ModelRegistry` — pluggable backend system with custom backend support
- `HTMLReporter` — standalone interactive HTML report with Plotly charts and prediction browser
- `vlm-eval` CLI: `run`, `validate`, `list`, `demo`, `report` commands
- FastAPI REST API: `/evaluate`, `/predict`, `/jobs`, `/health`, `/models`, `/metrics`
- Docker + Docker Compose support
- GitHub Actions CI: lint → test (3.10/3.11/3.12) → Docker build
- 5 pre-built YAML scenario configs: anomaly detection, scene understanding, captioning, VQA, safety-critical
- Examples: `run_evaluation.py`, `custom_vlm_example.py`
- 60+ unit and integration tests across config, dataset, metrics, models, engine, and API
