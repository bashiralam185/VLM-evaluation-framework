"""
examples/custom_vlm_example.py
===============================
Shows how to plug in any custom model backend into the framework —
without modifying the core library.

Three examples:
  1. Custom HuggingFace VLM (LLaVA-style)
  2. Custom API-based VLM (any REST endpoint)
  3. Registering the custom backend for use in YAML configs

Run with:
    python examples/custom_vlm_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from loguru import logger
from vlm_eval.models.registry import BaseVLM, ModelRegistry
from vlm_eval.core.config import (
    EvalConfig, ModelConfig, ScenarioConfig, DatasetConfig, MetricConfig,
)
from vlm_eval.core.engine import EvaluationEngine


# ─────────────────────────────────────────────────────────────────
# Example 1: Custom HuggingFace VLM
# ─────────────────────────────────────────────────────────────────

class MyCustomHFModel(BaseVLM):
    """
    Example: Wrapping a custom HuggingFace VLM.

    Swap the model_id and processor logic for your specific model.
    This example uses InstructBLIP-style API as a demonstration.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # In real usage, load your model here:
        # from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        # self._processor = InstructBlipProcessor.from_pretrained(config.hf_model_id)
        # self._model = InstructBlipForConditionalGeneration.from_pretrained(config.hf_model_id)
        self._loaded = True
        logger.info(f"Custom HF model {config.hf_model_id!r} initialized (placeholder).")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        # Real implementation:
        # inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        # output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        # return self._processor.decode(output[0], skip_special_tokens=True)
        return f"[CustomHF] Response to: {prompt[:50]}..."

    def unload(self):
        self._loaded = False


# ─────────────────────────────────────────────────────────────────
# Example 2: Custom REST-API VLM
# ─────────────────────────────────────────────────────────────────

class APIBackedVLM(BaseVLM):
    """
    Example: Calls any REST API (e.g. your own inference server,
    GPT-4V, or a custom FastAPI endpoint) to generate predictions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._api_url = config.extra_kwargs.get("api_url", "http://localhost:9000/generate")
        self._loaded = True
        logger.info(f"API-backed VLM pointing to: {self._api_url}")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        import base64
        import io

        # Encode image
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Real implementation:
        # import requests
        # response = requests.post(self._api_url, json={
        #     "image_base64": img_b64,
        #     "prompt": prompt,
        #     "max_new_tokens": max_new_tokens,
        # }, timeout=30)
        # return response.json()["prediction"]

        return f"[APIBacked] Mock response for prompt: {prompt[:50]}..."

    def unload(self):
        self._loaded = False


# ─────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Custom VLM Integration Example")
    print("=" * 60)

    # Step 1: Register custom backends
    registry = ModelRegistry()
    registry.register("custom_hf", MyCustomHFModel)
    registry.register("api_backed", APIBackedVLM)

    print(f"\nAvailable backends: {registry.list_backends()}")

    # Step 2: Test single inference
    print("\n--- Single inference test ---")
    cfg = ModelConfig(
        name="My Custom Model",
        hf_model_id="my-org/my-vlm",
        model_type="custom_hf",
    )
    model = registry.load(cfg)
    test_image = Image.new("RGB", (224, 224), (128, 200, 128))
    result = model.generate(test_image, "Describe this image.")
    print(f"Prediction: {result}")

    # Step 3: Use in a full evaluation pipeline
    print("\n--- Full pipeline with custom backend ---")
    eval_cfg = EvalConfig(
        suite_name="Custom VLM Evaluation",
        models=[
            ModelConfig(
                name="MyCustomHF",
                hf_model_id="my-org/my-vlm",
                model_type="custom_hf",
            ),
            ModelConfig(
                name="APIBackedVLM",
                hf_model_id="my-api-model",
                model_type="api_backed",
                extra_kwargs={"api_url": "http://localhost:9000/generate"},
            ),
        ],
        scenarios=[
            ScenarioConfig(
                name="Scene Understanding",
                task_type="scene_understanding",
                prompt_template="Describe this scene.",
                dataset=DatasetConfig(name="synthetic_scene", source="synthetic", max_samples=5),
                metrics=MetricConfig(bleu=True, rouge=True, semantic_similarity=True, clip_score=False),
            ),
        ],
        output_dir="./eval_results/custom_vlm_demo",
        export_html_report=True,
        save_predictions=True,
    )

    # IMPORTANT: Pass the custom registry to the engine
    engine = EvaluationEngine(eval_cfg)
    engine._model_registry = registry  # inject our custom registry

    results = engine.run()

    for model_name, scenario_results in results.items():
        for sr in scenario_results:
            agg = sr.aggregate_scores()
            print(f"\n{model_name} / {sr.scenario_name}:")
            for k, v in agg.items():
                print(f"  {k}: {v:.4f}")

    print("\nDone! Check ./eval_results/custom_vlm_demo/eval_report.html")


if __name__ == "__main__":
    main()
