"""
DatasetLoader: Loads images + ground-truth references for each scenario.

Supported sources:
  - synthetic  : Generates programmatic test data (no download required)
  - coco       : MS-COCO via pycocotools (downloads on first use)
  - huggingface: Any HF datasets Hub dataset
  - custom     : User-provided image directory + annotation JSON/CSV
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter

from vlm_eval.core.config import DatasetConfig, ScenarioConfig
from vlm_eval.core.scenario import EvalSample


class DatasetLoader:
    """Routes dataset loading to the appropriate backend."""

    def load(
        self,
        dataset_cfg: DatasetConfig,
        scenario_cfg: ScenarioConfig,
        max_samples: Optional[int] = None,
    ) -> List[EvalSample]:
        source = dataset_cfg.source
        loaders = {
            "synthetic": self._load_synthetic,
            "coco": self._load_coco,
            "huggingface": self._load_huggingface,
            "custom": self._load_custom,
        }
        if source not in loaders:
            raise ValueError(f"Unknown dataset source: {source!r}. Valid: {list(loaders.keys())}")

        samples = loaders[source](dataset_cfg, scenario_cfg)

        if max_samples and len(samples) > max_samples:
            rng = random.Random(42)
            samples = rng.sample(samples, max_samples)

        logger.info(f"Loaded {len(samples)} samples from {source} ({scenario_cfg.name})")
        return samples

    # ─────────────────────────────────────────────────────────────
    # Synthetic dataset (no download, good for CI / demos)
    # ─────────────────────────────────────────────────────────────

    def _load_synthetic(
        self,
        cfg: DatasetConfig,
        scenario_cfg: ScenarioConfig,
    ) -> List[EvalSample]:
        """
        Generate synthetic evaluation data with PIL.

        Creates realistic-looking test images for each scenario type
        with pre-defined reference captions.
        """
        n = cfg.max_samples or 20
        samples = []

        generators = {
            "anomaly_detection": self._gen_anomaly_images,
            "scene_understanding": self._gen_scene_images,
            "image_captioning": self._gen_caption_images,
            "visual_question_answering": self._gen_vqa_images,
            "object_recognition": self._gen_object_images,
            "safety_critical": self._gen_safety_images,
        }

        gen_fn = generators.get(scenario_cfg.task_type, self._gen_caption_images)

        for i in range(n):
            image, refs = gen_fn(i)
            prompt = scenario_cfg.prompt_template

            samples.append(EvalSample(
                sample_id=f"synthetic_{scenario_cfg.task_type}_{i:04d}",
                image_path=None,
                image=image,
                prompt=prompt,
                references=refs,
                metadata={
                    "source": "synthetic",
                    "task_type": scenario_cfg.task_type,
                    "index": i,
                },
            ))

        return samples

    def _gen_anomaly_images(self, seed: int):
        """Generate road/traffic images with/without anomalies."""
        rng = random.Random(seed)
        img = Image.new("RGB", (224, 224), color=(rng.randint(80, 120), rng.randint(80, 120), rng.randint(80, 120)))
        draw = ImageDraw.Draw(img)

        # Draw road
        draw.rectangle([70, 0, 150, 224], fill=(60, 60, 60))
        draw.line([110, 0, 110, 224], fill=(255, 255, 0), width=3)

        # Draw sky
        draw.rectangle([0, 0, 224, 80], fill=(135, 206, 235))

        has_anomaly = rng.random() > 0.5
        if has_anomaly:
            # Draw anomalous object on road
            x, y = rng.randint(75, 140), rng.randint(100, 170)
            color = rng.choice([(255, 0, 0), (255, 165, 0), (128, 0, 128)])
            draw.rectangle([x, y, x + 30, y + 20], fill=color)
            refs = [
                "An unusual object is blocking the road ahead.",
                "There is an anomalous obstacle on the roadway.",
                "A foreign object is detected on the driving surface.",
            ]
        else:
            refs = [
                "The road ahead is clear with no obstacles.",
                "A normal road scene with no anomalies detected.",
                "The driving path appears safe and unobstructed.",
            ]

        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img, refs

    def _gen_scene_images(self, seed: int):
        """Generate multi-element scene images."""
        rng = random.Random(seed)
        scenes = [
            ("outdoor", (135, 206, 235), (34, 139, 34)),
            ("indoor", (200, 180, 160), (139, 90, 43)),
            ("urban", (100, 100, 110), (80, 80, 80)),
        ]
        scene_name, sky_color, ground_color = scenes[seed % len(scenes)]
        img = Image.new("RGB", (224, 224), sky_color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 140, 224, 224], fill=ground_color)

        # Add scene elements
        for _ in range(rng.randint(2, 5)):
            x = rng.randint(0, 200)
            y = rng.randint(50, 170)
            w, h = rng.randint(10, 40), rng.randint(10, 60)
            color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))
            draw.rectangle([x, y, x + w, y + h], fill=color)

        refs = {
            "outdoor": [
                "An outdoor natural scene with sky and green landscape.",
                "Exterior environment showing natural elements and open sky.",
            ],
            "indoor": [
                "An indoor scene with warm colors and enclosed space.",
                "Interior environment with furniture-like elements.",
            ],
            "urban": [
                "An urban scene with buildings and infrastructure.",
                "City environment showing architectural structures.",
            ],
        }
        return img, refs[scene_name]

    def _gen_caption_images(self, seed: int):
        """Generate general images for captioning."""
        rng = random.Random(seed)
        bg_colors = [(220, 220, 255), (255, 220, 220), (220, 255, 220), (255, 255, 200)]
        img = Image.new("RGB", (224, 224), bg_colors[seed % len(bg_colors)])
        draw = ImageDraw.Draw(img)

        shapes = rng.randint(2, 6)
        for _ in range(shapes):
            x1, y1 = rng.randint(10, 180), rng.randint(10, 180)
            x2, y2 = x1 + rng.randint(20, 60), y1 + rng.randint(20, 60)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            if rng.random() > 0.5:
                draw.ellipse([x1, y1, x2, y2], fill=color)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color)

        refs = [
            f"An image containing {shapes} geometric shapes on a colored background.",
            f"A visual composition with approximately {shapes} objects arranged in a scene.",
            "The image shows multiple colored shapes in a simple arrangement.",
        ]
        return img, refs

    def _gen_vqa_images(self, seed: int):
        """Generate images for visual question answering."""
        rng = random.Random(seed)
        img = Image.new("RGB", (224, 224), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Draw colored objects
        colors = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 200, 0)}
        chosen_color = list(colors.keys())[seed % len(colors)]
        rgb = colors[chosen_color]
        x, y = rng.randint(60, 150), rng.randint(60, 150)
        draw.ellipse([x, y, x + 50, y + 50], fill=rgb)

        refs = [
            f"Yes, there is a {chosen_color} circle in the image.",
            f"The main object is a {chosen_color} circle.",
        ]
        return img, refs

    def _gen_object_images(self, seed: int):
        """Generate images for object recognition."""
        rng = random.Random(seed)
        obj_types = ["circle", "rectangle", "triangle"]
        obj = obj_types[seed % len(obj_types)]
        img = Image.new("RGB", (224, 224), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        color = (rng.randint(50, 200), rng.randint(50, 200), rng.randint(50, 200))

        if obj == "circle":
            draw.ellipse([62, 62, 162, 162], fill=color)
            refs = ["A circle is present in the center of the image.", "Circular shape detected."]
        elif obj == "rectangle":
            draw.rectangle([50, 80, 174, 144], fill=color)
            refs = ["A rectangle is visible in the image.", "Rectangular object in the scene."]
        else:
            draw.polygon([(112, 40), (40, 184), (184, 184)], fill=color)
            refs = ["A triangle shape is visible.", "Triangular object detected in the image."]

        return img, refs

    def _gen_safety_images(self, seed: int):
        """Generate safety-critical scenario images (obstacles, hazards)."""
        rng = random.Random(seed)
        img = Image.new("RGB", (224, 224), (100, 100, 100))
        draw = ImageDraw.Draw(img)
        draw.rectangle([70, 0, 154, 224], fill=(50, 50, 50))  # road

        hazard_types = ["pedestrian", "vehicle", "debris"]
        hazard = hazard_types[seed % len(hazard_types)]

        if hazard == "pedestrian":
            draw.ellipse([95, 60, 125, 90], fill=(255, 200, 150))   # head
            draw.rectangle([100, 90, 120, 150], fill=(0, 100, 200))  # body
            refs = ["A pedestrian is detected on the road ahead.",
                    "There is a person crossing in the vehicle path."]
        elif hazard == "vehicle":
            draw.rectangle([80, 100, 140, 150], fill=(200, 50, 50))
            refs = ["A vehicle is stopped in the road.",
                    "Another car is blocking the driving lane."]
        else:
            for _ in range(rng.randint(3, 7)):
                x, y = rng.randint(75, 140), rng.randint(80, 170)
                draw.rectangle([x, y, x + 15, y + 10], fill=(139, 69, 19))
            refs = ["Debris is scattered across the road surface.",
                    "Road debris detected creating a hazard."]

        return img, refs

    # ─────────────────────────────────────────────────────────────
    # COCO dataset
    # ─────────────────────────────────────────────────────────────

    def _load_coco(
        self,
        cfg: DatasetConfig,
        scenario_cfg: ScenarioConfig,
    ) -> List[EvalSample]:
        """Load from MS-COCO using pycocotools."""
        try:
            from pycocotools.coco import COCO  # type: ignore

            ann_file = cfg.annotation_file
            img_dir = cfg.image_dir

            if not ann_file or not img_dir:
                logger.warning(
                    "COCO source requires 'annotation_file' and 'image_dir' in dataset config. "
                    "Falling back to synthetic."
                )
                return self._load_synthetic(cfg, scenario_cfg)

            ann_file = Path(ann_file)
            img_dir = Path(img_dir)

            if not ann_file.exists():
                logger.warning(f"COCO annotation file not found: {ann_file}. Falling back to synthetic.")
                return self._load_synthetic(cfg, scenario_cfg)

            coco = COCO(str(ann_file))
            img_ids = coco.getImgIds()
            if cfg.max_samples:
                img_ids = img_ids[: cfg.max_samples]

            samples = []
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                img_path = img_dir / img_info["file_name"]

                if not img_path.exists():
                    continue

                try:
                    image = Image.open(img_path).convert("RGB").resize((224, 224))
                except Exception:
                    continue

                # Get captions
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                refs = [a["caption"] for a in anns if "caption" in a]
                if not refs:
                    refs = [img_info.get("file_name", "an image")]

                samples.append(EvalSample(
                    sample_id=str(img_id),
                    image_path=img_path,
                    image=image,
                    prompt=scenario_cfg.prompt_template,
                    references=refs,
                    metadata={"coco_id": img_id, "source": "coco"},
                ))

            logger.info(f"Loaded {len(samples)} COCO samples.")
            return samples

        except ImportError:
            logger.warning("pycocotools not installed. Falling back to synthetic data.")
            return self._load_synthetic(cfg, scenario_cfg)

    # ─────────────────────────────────────────────────────────────
    # HuggingFace datasets
    # ─────────────────────────────────────────────────────────────

    def _load_huggingface(
        self,
        cfg: DatasetConfig,
        scenario_cfg: ScenarioConfig,
    ) -> List[EvalSample]:
        """Load from HuggingFace datasets Hub."""
        try:
            import datasets as hf_datasets  # type: ignore

            if not cfg.hf_dataset_id:
                logger.warning("hf_dataset_id not specified. Falling back to synthetic.")
                return self._load_synthetic(cfg, scenario_cfg)

            logger.info(f"Loading HuggingFace dataset: {cfg.hf_dataset_id} (split={cfg.split})")
            cache_dir = Path(cfg.cache_dir).expanduser()

            dataset = hf_datasets.load_dataset(
                cfg.hf_dataset_id,
                split=cfg.split,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
            )

            if cfg.max_samples:
                dataset = dataset.select(range(min(cfg.max_samples, len(dataset))))

            samples = []
            image_col = None
            caption_col = None

            # Auto-detect columns
            for col in dataset.column_names:
                if col in ("image", "img", "pixel_values"):
                    image_col = col
                if col in ("caption", "captions", "text", "label"):
                    caption_col = col

            for i, item in enumerate(dataset):
                try:
                    image = item.get(image_col) if image_col else None
                    if image is None or not isinstance(image, Image.Image):
                        image = Image.new("RGB", (224, 224), (200, 200, 200))
                    else:
                        image = image.convert("RGB").resize((224, 224))

                    caption = item.get(caption_col, "") if caption_col else ""
                    refs = [caption] if caption else ["An image."]

                    samples.append(EvalSample(
                        sample_id=f"hf_{i:06d}",
                        image_path=None,
                        image=image,
                        prompt=scenario_cfg.prompt_template,
                        references=refs,
                        metadata={"source": "huggingface", "hf_id": cfg.hf_dataset_id},
                    ))
                except Exception as e:
                    logger.warning(f"Skipping HF sample {i}: {e}")

            return samples

        except ImportError:
            logger.warning("datasets not installed. Falling back to synthetic.")
            return self._load_synthetic(cfg, scenario_cfg)
        except Exception as e:
            logger.warning(f"HuggingFace dataset loading failed: {e}. Falling back to synthetic.")
            return self._load_synthetic(cfg, scenario_cfg)

    # ─────────────────────────────────────────────────────────────
    # Custom dataset (image dir + JSON annotations)
    # ─────────────────────────────────────────────────────────────

    def _load_custom(
        self,
        cfg: DatasetConfig,
        scenario_cfg: ScenarioConfig,
    ) -> List[EvalSample]:
        """
        Load from a custom dataset directory.

        Expected annotation format (JSON):
        [
          {
            "id": "img_001",
            "file_name": "img_001.jpg",
            "references": ["A caption", "Another caption"],
            "metadata": {}
          },
          ...
        ]
        """
        img_dir = Path(cfg.image_dir) if cfg.image_dir else None
        ann_file = Path(cfg.annotation_file) if cfg.annotation_file else None

        if img_dir is None or not img_dir.exists():
            logger.warning(f"Custom image_dir not found: {img_dir}. Falling back to synthetic.")
            return self._load_synthetic(cfg, scenario_cfg)

        samples = []

        # Load annotations if provided
        annotations: List[Dict[str, Any]] = []
        if ann_file and ann_file.exists():
            with open(ann_file) as f:
                annotations = json.load(f)
        else:
            # Auto-discover images in directory
            extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            image_files = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in extensions]
            annotations = [
                {"id": f.stem, "file_name": f.name, "references": [f"An image of {f.stem}."]}
                for f in image_files
            ]

        if cfg.max_samples:
            annotations = annotations[: cfg.max_samples]

        for ann in annotations:
            img_path = img_dir / ann["file_name"]
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            try:
                image = Image.open(img_path).convert("RGB").resize((224, 224))
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")
                continue

            samples.append(EvalSample(
                sample_id=str(ann.get("id", img_path.stem)),
                image_path=img_path,
                image=image,
                prompt=scenario_cfg.prompt_template,
                references=ann.get("references", ["An image."]),
                metadata=ann.get("metadata", {}),
            ))

        logger.info(f"Loaded {len(samples)} custom samples from {img_dir}.")
        return samples
