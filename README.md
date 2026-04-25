# 🔬 VLM Eval Framework

[![CI](https://github.com/bashiralam185/vlm-eval-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/bashiralam185/vlm-eval-framework/actions)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> A **modular, scenario-driven benchmarking toolkit** for evaluating Vision-Language Models (VLMs) across structured task categories. Plug in any HuggingFace VLM, define test suites in YAML, and get a full interactive HTML comparison report in minutes.

Built from experience evaluating large multimodal models at **AiLiveSim**, Turku, Finland — where robust VLM benchmarking was needed for autonomous systems deployment decisions.

---

## Why This Exists

Most VLM benchmarking is done ad-hoc — one model, one dataset, one script. In production settings (autonomous systems, robotics, retail analytics), you need:

- **Structured scenarios** — anomaly detection behaves differently from scene understanding
- **Multiple models compared simultaneously** — side-by-side on identical inputs
- **Reproducible configs** — the same YAML drives every run
- **Automated reports** — shareable HTML, not notebooks

This library provides all of that as a clean, extensible Python package.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      EvaluationEngine                            │
│                                                                  │
│  YAML Config ──→  ScenarioSuite  ──→  DatasetLoader             │
│                                             │                    │
│  ModelRegistry ──→  VLM Backends ──→  Inference                 │
│    HuggingFace          │                   │                    │
│    Triton               ▼                   ▼                    │
│    Custom           MetricRegistry ──→  ScenarioResult          │
│    Mock              BLEU/ROUGE                │                 │
│                      CLIPScore          HTMLReporter             │
│                      BERTScore          FastAPI                  │
│                      SemanticSim        CLI                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **YAML-driven configs** | Define full test suites without writing Python |
| **5 scenario types** | Anomaly detection, scene understanding, captioning, VQA, safety-critical |
| **4 dataset backends** | Synthetic, MS-COCO, HuggingFace Hub, custom image dirs |
| **6 metric types** | BLEU, ROUGE, CLIPScore, BERTScore, Semantic Similarity, Exact Match |
| **3 model backends** | HuggingFace VLMs, Triton Inference Server, Mock (for testing) |
| **HTML report** | Interactive Plotly charts, prediction browser, leaderboard |
| **REST API** | FastAPI endpoints for programmatic evaluation |
| **CLI** | `vlm-eval run`, `vlm-eval demo`, `vlm-eval validate` |
| **Extensible** | Register any custom backend in 10 lines |

---

## Quick Start

### Installation

```bash
git clone https://github.com/bashiralam185/vlm-eval-framework.git
cd vlm-eval-framework
poetry install
poetry shell
```

### Run the demo (no GPU, no downloads)

```bash
# Via CLI
vlm-eval demo --samples 10

# Via Python
python examples/run_evaluation.py

# Via YAML config
vlm-eval run configs/scenarios/demo.yaml
```

### Evaluate a real HuggingFace VLM

```bash
vlm-eval run configs/scenarios/full_suite.yaml \
  --model "LLaVA-1.5-7B" \
  --scenario "Anomaly Detection"
```

### Python API

```python
from vlm_eval import EvaluationEngine, EvalConfig

config = EvalConfig.from_yaml("configs/scenarios/full_suite.yaml")
engine = EvaluationEngine(config)
results = engine.run()

# Leaderboard
lb = engine.get_leaderboard(results, metric="clip_score")
print(lb)
# Open eval_results/eval_report.html in browser
```

---

## YAML Configuration

Define your entire benchmark in a single YAML file — no code needed for new test suites:

```yaml
suite_name: "My VLM Benchmark"
output_dir: "./eval_results/my_benchmark"

models:
  - name: "LLaVA-1.5-7B"
    hf_model_id: "llava-hf/llava-1.5-7b-hf"
    model_type: "hf_vlm"
    device: "auto"
    dtype: "float16"
    max_new_tokens: 256

  - name: "SmolVLM-256M"
    hf_model_id: "HuggingFaceTB/SmolVLM-256M-Instruct"
    model_type: "hf_vlm"

scenarios:
  - name: "Anomaly Detection"
    task_type: "anomaly_detection"
    prompt_template: >
      Examine this image carefully. Describe any unusual or anomalous
      elements. If the scene appears normal, say so explicitly.
    dataset:
      source: "synthetic"       # or: coco, huggingface, custom
      max_samples: 100
    metrics:
      bleu: true
      rouge: true
      clip_score: true
      semantic_similarity: true

  - name: "COCO Captioning"
    task_type: "image_captioning"
    prompt_template: "Describe this image in one sentence."
    dataset:
      source: "coco"
      image_dir: "/path/to/coco/val2017"
      annotation_file: "/path/to/captions_val2017.json"
      max_samples: 500
    metrics:
      bleu: true
      rouge: true
      clip_score: true
```

---

## Scenario Types

| `task_type` | Description | Example prompt |
|-------------|-------------|----------------|
| `anomaly_detection` | Detect unusual/unexpected elements | "Describe any anomalous elements you observe." |
| `scene_understanding` | Holistic scene comprehension | "What type of environment is shown and what are its key elements?" |
| `image_captioning` | General visual description | "Describe this image in one to two sentences." |
| `visual_question_answering` | Direct factual questions | "What color is the main object?" |
| `object_recognition` | Object identification | "List the main objects visible in this image." |
| `safety_critical` | Safety-relevant scenarios | "Identify any safety-critical elements requiring attention." |

---

## Dataset Sources

### Synthetic (no download)
```yaml
dataset:
  source: "synthetic"
  max_samples: 50
```
Programmatically generated images with reference captions — perfect for CI and quick demos.

### MS-COCO
```yaml
dataset:
  source: "coco"
  image_dir: "/path/to/coco/images/val2017"
  annotation_file: "/path/to/coco/annotations/captions_val2017.json"
  max_samples: 500
```

### HuggingFace Hub
```yaml
dataset:
  source: "huggingface"
  hf_dataset_id: "nlphuji/flickr30k"
  split: "test"
  max_samples: 200
```

### Custom image directory
```yaml
dataset:
  source: "custom"
  image_dir: "/my/images/"
  annotation_file: "/my/annotations.json"  # optional
```
Annotation JSON format:
```json
[{"id": "img001", "file_name": "img001.jpg", "references": ["Caption 1", "Caption 2"]}]
```

---

## Custom VLM Backend

Register any model backend in ~10 lines:

```python
from vlm_eval.models.registry import BaseVLM, ModelRegistry
from PIL import Image

class MyModel(BaseVLM):
    def __init__(self, config):
        super().__init__(config)
        # Load your model here
        self._loaded = True

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        # Your inference logic
        return "A description of the image."

    def unload(self):
        self._loaded = False

# Register globally
registry = ModelRegistry()
registry.register("my_backend", MyModel)

# Now use in config
# model_type: "my_backend"
```

---

## Metrics

| Metric | Library | Measures | Best for |
|--------|---------|----------|----------|
| **BLEU-1/2/4** | `nltk` | N-gram precision vs references | Standard captioning benchmark |
| **ROUGE-1/2/L** | `rouge-score` | N-gram recall vs references | Summarization-style tasks |
| **CLIPScore** | `transformers` (CLIP) | Image–text alignment | Reference-free quality check |
| **BERTScore F1** | `bert-score` | Semantic similarity via BERT | Open-ended generation |
| **Semantic Similarity** | `sentence-transformers` | Embedding cosine similarity | Paraphrase-aware evaluation |
| **Exact Match** | built-in | Binary string equality | VQA with fixed answers |
| **Accuracy** | built-in | Substring containment | Short-answer VQA |

---

## CLI Reference

```bash
# Run full suite from YAML
vlm-eval run configs/scenarios/full_suite.yaml

# Run specific model or scenario only
vlm-eval run configs/scenarios/full_suite.yaml --model "LLaVA-1.5-7B"
vlm-eval run configs/scenarios/full_suite.yaml --scenario "Anomaly Detection"

# Quick demo (mock models, synthetic data, no GPU)
vlm-eval demo --samples 20 --output ./my_results

# Validate a config without running
vlm-eval validate configs/scenarios/full_suite.yaml

# List available backends and metrics
vlm-eval list
```

---

## REST API

Start the server:
```bash
uvicorn api.main:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for interactive Swagger UI.

```bash
# Health check
curl http://localhost:8000/health

# Run evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "suite_name": "API Test",
    "models": [{"name": "MockVLM", "hf_model_id": "mock", "model_type": "mock"}],
    "scenarios": [{
      "name": "Scene Understanding",
      "task_type": "scene_understanding",
      "prompt_template": "Describe this scene.",
      "dataset": {"name": "ds1", "source": "synthetic", "max_samples": 5}
    }]
  }'

# Single-image inference
curl -X POST http://localhost:8000/predict \
  -d '{"model_name": "MockVLM", "hf_model_id": "mock", "model_type": "mock",
       "image_base64": "<base64>", "prompt": "Describe this image."}'
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | Available backends |
| `/metrics` | GET | Available metrics |
| `/evaluate` | POST | Run full evaluation |
| `/evaluate/file` | POST | Upload YAML config |
| `/predict` | POST | Single image inference |
| `/jobs` | GET | List all jobs |
| `/jobs/{id}` | GET | Job status/results |
| `/jobs/{id}/report` | GET | HTML report |

---

## Docker

```bash
# Build and start API
docker-compose up --build

# API available at http://localhost:8000/docs
```

---

## Development

```bash
# Install with dev deps
poetry install --with dev

# Run tests
poetry run pytest tests/ -v --cov=vlm_eval

# Lint
poetry run ruff check vlm_eval/

# Pre-commit hooks
pre-commit install

# Run demo
python examples/run_evaluation.py
```

---

## Project Structure

```
vlm-eval-framework/
├── vlm_eval/                    # Core library
│   ├── core/
│   │   ├── config.py            # Pydantic config (EvalConfig, ModelConfig, ...)
│   │   ├── scenario.py          # Scenario, ScenarioSuite, EvalSample, EvalResult
│   │   └── engine.py            # EvaluationEngine — main orchestrator
│   ├── models/
│   │   └── registry.py          # ModelRegistry, HuggingFaceVLM, MockVLM, TritonVLM
│   ├── metrics/
│   │   └── registry.py          # MetricRegistry, BLEU, ROUGE, CLIPScore, ...
│   ├── datasets/
│   │   └── loader.py            # DatasetLoader — synthetic, COCO, HF, custom
│   ├── reporters/
│   │   └── html_reporter.py     # HTMLReporter — standalone interactive report
│   └── cli.py                   # Typer CLI — run, demo, validate, list
├── api/
│   └── main.py                  # FastAPI REST endpoints
├── configs/
│   └── scenarios/
│       ├── full_suite.yaml      # Complete 5-scenario benchmark
│       ├── demo.yaml            # Quick demo (mock models)
│       └── coco_captioning.yaml # COCO-based evaluation
├── tests/
│   ├── test_config_and_dataset.py
│   ├── test_metrics.py
│   ├── test_models_and_engine.py
│   └── test_api.py
├── examples/
│   ├── run_evaluation.py        # Full end-to-end demo script
│   └── custom_vlm_example.py   # How to add a custom backend
├── .github/workflows/ci.yml    # GitHub Actions: lint → test → Docker → build
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Citation

If you use this framework in research, please cite:

```bibtex
@software{alam2025vlmeval,
  author    = {Alam, Bashir},
  title     = {VLM Eval Framework: Scenario-Driven Benchmarking for Vision-Language Models},
  year      = {2025},
  url       = {https://github.com/bashiralam185/vlm-eval-framework},
  note      = {Built from multimodal evaluation work at AiLiveSim, Turku, Finland}
}
```

Related work:
- **CALF** (arXiv:2504.04458) — Conditionally Adaptive Loss Function for class-imbalanced segmentation
- **AiLiveSim** evaluation work — benchmarking VLMs for autonomous systems deployment

---

## Author

**Bashir Alam** — Machine Learning Engineer / Researcher
Åbo Akademi University, Finland · AiLiveSim, Turku (Nov 2024 – May 2025)

📧 [bashir.alam@abo.fi](mailto:bashir.alam@abo.fi) · 🔗 [LinkedIn](https://linkedin.com/in/bashir-alam/) · 🐙 [GitHub](https://github.com/bashiralam185) · 📄 [ORCID](https://orcid.org/0009-0007-8672-5529)

---

## License

MIT — see [LICENSE](LICENSE).
