"""
ModelRegistry: Manages loading and caching of VLM backends.

Supports:
  - HuggingFace VLMs (any model with AutoModelForVision2Seq / LLaVA-style)
  - Mock model (for testing without GPU)
  - Triton Inference Server client (for high-throughput batched eval)
  - OpenAI-compatible API (GPT-4V)
"""

from __future__ import annotations

import abc
import time
from typing import Any, Optional

from loguru import logger
from PIL import Image

from vlm_eval.core.config import ModelConfig


# ─────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────

class BaseVLM(abc.ABC):
    """Abstract base for all VLM backends."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._loaded = False

    @property
    def name(self) -> str:
        return self.config.name

    @abc.abstractmethod
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from an image + prompt."""

    def unload(self):
        """Release GPU memory."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ─────────────────────────────────────────────────────────────────
# HuggingFace VLM backend
# ─────────────────────────────────────────────────────────────────

class HuggingFaceVLM(BaseVLM):
    """
    Generic HuggingFace VLM backend.

    Works with any model that has a processor + model that accepts
    pixel_values + input_ids (LLaVA, InstructBLIP, Idefics, SmolVLM, etc.)

    The loader uses AutoProcessor + AutoModelForVision2Seq with
    automatic device mapping.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info(f"Loading {self.config.hf_model_id} on {device} ({self.config.dtype})...")
        t0 = time.perf_counter()

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.hf_model_id,
                trust_remote_code=True,
            )

            # Try LLaVA-specific loader first, then generic
            try:
                self._model = LlavaForConditionalGeneration.from_pretrained(
                    self.config.hf_model_id,
                    torch_dtype=dtype,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True,
                    **self.config.extra_kwargs,
                )
            except Exception:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.config.hf_model_id,
                    torch_dtype=dtype,
                    device_map=device if device != "cpu" else None,
                    trust_remote_code=True,
                    **self.config.extra_kwargs,
                )

            if device == "cpu" and self._model is not None:
                self._model = self._model.to(device)

            self._model.eval()
            self._loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"Model loaded in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Failed to load {self.config.hf_model_id}: {e}")
            raise

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        import torch

        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded.")

        try:
            # Build conversation-style input (works with most HF VLMs)
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    pad_token_id=self._processor.tokenizer.eos_token_id
                    if hasattr(self._processor, "tokenizer")
                    else None,
                )

            # Decode only the new tokens (strip prompt)
            input_len = inputs.get("input_ids", inputs.get("input_ids", None))
            if input_len is not None:
                generated = output_ids[:, input_len.shape[-1]:]
            else:
                generated = output_ids

            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            return text.strip()

        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

    def unload(self):
        import gc
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info(f"Model {self.name!r} unloaded.")


# ─────────────────────────────────────────────────────────────────
# Triton Inference Server backend (high-throughput)
# ─────────────────────────────────────────────────────────────────

class TritonVLM(BaseVLM):
    """
    Triton Inference Server client backend.

    Routes inference requests to a deployed Triton model for
    high-throughput batched evaluation.

    Requires: tritonclient[http] — `pip install tritonclient[http]`
    """

    def __init__(self, config: ModelConfig, triton_url: str = "localhost:8001"):
        super().__init__(config)
        self._triton_url = triton_url
        self._client = None
        self._load()

    def _load(self):
        try:
            import tritonclient.http as httpclient  # type: ignore

            self._client = httpclient.InferenceServerClient(url=self._triton_url)
            if self._client.is_server_live():
                logger.info(f"Connected to Triton at {self._triton_url}")
                self._loaded = True
            else:
                logger.warning(f"Triton server at {self._triton_url} is not live.")
        except ImportError:
            logger.warning(
                "tritonclient not installed. Install with: pip install tritonclient[http]"
            )
        except Exception as e:
            logger.warning(f"Could not connect to Triton: {e}")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Send image + prompt to Triton and return generated text."""
        if not self._loaded or self._client is None:
            raise RuntimeError("Triton client not connected. Check server at " + self._triton_url)

        try:
            import numpy as np
            import tritonclient.http as httpclient  # type: ignore

            # Encode image to bytes
            import io
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_array = np.frombuffer(img_bytes.getvalue(), dtype=np.uint8)
            img_array = img_array.reshape([1, -1])

            # Build Triton inputs
            inputs = [
                httpclient.InferInput("IMAGE", img_array.shape, "UINT8"),
                httpclient.InferInput("PROMPT", [1, 1], "BYTES"),
            ]
            inputs[0].set_data_from_numpy(img_array)
            inputs[1].set_data_from_numpy(
                np.array([[prompt.encode("utf-8")]], dtype=object)
            )

            outputs = [httpclient.InferRequestedOutput("GENERATED_TEXT")]
            response = self._client.infer(
                model_name=self.config.hf_model_id,
                inputs=inputs,
                outputs=outputs,
            )
            result = response.as_numpy("GENERATED_TEXT")[0][0].decode("utf-8")
            return result.strip()

        except Exception as e:
            logger.warning(f"Triton inference failed: {e}")
            return ""

    def unload(self):
        self._client = None
        self._loaded = False


# ─────────────────────────────────────────────────────────────────
# Mock VLM backend (testing / CI)
# ─────────────────────────────────────────────────────────────────

class MockVLM(BaseVLM):
    """
    Deterministic mock VLM for unit testing.

    Returns template responses based on the prompt and a seed,
    allowing full pipeline testing without any GPU or model download.
    """

    RESPONSES = {
        "anomaly": [
            "The image shows a road with an unusual object blocking the path.",
            "There is an anomalous placement of items in the scene.",
            "The scene appears normal with no unusual objects detected.",
            "An unexpected obstacle is visible in the upper left region.",
        ],
        "scene": [
            "The image depicts an outdoor urban environment with buildings and pedestrians.",
            "A natural landscape with trees and a clear sky is visible.",
            "The indoor scene shows a room with furniture and lighting.",
            "Multiple objects are arranged in a structured pattern.",
        ],
        "caption": [
            "A photograph showing various objects in a natural setting.",
            "An image containing multiple elements typical of everyday scenes.",
            "The visual content depicts a common real-world scenario.",
            "Several items are visible, arranged in a visually coherent manner.",
        ],
        "vqa": [
            "Yes, based on the visual content provided.",
            "The answer is approximately three objects of that type.",
            "No, that element is not present in this image.",
            "It appears to be in the center-left portion of the frame.",
        ],
    }

    def __init__(self, config: ModelConfig, seed: int = 42):
        super().__init__(config)
        self._seed = seed
        self._call_count = 0
        self._loaded = True

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Return a deterministic mock response based on prompt keywords."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["anomaly", "unusual", "strange", "obstacle"]):
            pool = self.RESPONSES["anomaly"]
        elif any(w in prompt_lower for w in ["scene", "environment", "describe", "what"]):
            pool = self.RESPONSES["scene"]
        elif any(w in prompt_lower for w in ["caption", "summarize"]):
            pool = self.RESPONSES["caption"]
        elif "?" in prompt:
            pool = self.RESPONSES["vqa"]
        else:
            pool = self.RESPONSES["caption"]

        idx = (self._call_count + self._seed) % len(pool)
        self._call_count += 1
        return pool[idx]

    def unload(self):
        self._loaded = False


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Loads and caches VLM instances based on ModelConfig."""

    _BACKENDS = {
        "hf_vlm": HuggingFaceVLM,
        "mock": MockVLM,
        "triton": TritonVLM,
    }

    def __init__(self):
        self._cache: dict = {}

    def load(self, config: ModelConfig) -> BaseVLM:
        """Load a model (or return cached instance)."""
        key = f"{config.model_type}::{config.hf_model_id}"
        if key not in self._cache:
            backend_cls = self._BACKENDS.get(config.model_type)
            if backend_cls is None:
                raise ValueError(
                    f"Unknown model_type: {config.model_type!r}. "
                    f"Valid: {list(self._BACKENDS.keys())}"
                )
            self._cache[key] = backend_cls(config)
        return self._cache[key]

    def register(self, model_type: str, backend_cls: type):
        """Register a custom model backend."""
        self._BACKENDS[model_type] = backend_cls

    def list_backends(self) -> list:
        return list(self._BACKENDS.keys())

    def clear_cache(self):
        for model in self._cache.values():
            model.unload()
        self._cache.clear()
