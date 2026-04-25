"""
MetricRegistry: Pluggable metric computation system.

Supported metrics:
  - BLEU-1/2/3/4      (nltk)
  - ROUGE-1/2/L       (rouge-score or evaluate)
  - BERTScore F1      (bert-score)
  - CLIPScore         (openai/clip via transformers)
  - SemanticSimilarity (sentence-transformers cosine)
  - ExactMatch        (binary)
  - Accuracy          (for VQA / classification)

All metrics are computed lazily (model loaded on first call, then cached).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from PIL import Image

from vlm_eval.core.config import MetricConfig


# ─────────────────────────────────────────────────────────────────
# Individual metric functions
# ─────────────────────────────────────────────────────────────────

def compute_bleu(prediction: str, references: List[str]) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4.

    Uses nltk sentence_bleu with smoothing for short predictions.
    Returns dict: {"bleu_1": ..., "bleu_2": ..., "bleu_4": ...}
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoother = SmoothingFunction().method1
    refs = [r.lower().split() for r in references]
    hyp = prediction.lower().split()

    if len(hyp) == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    scores = {}
    for n, weights in [
        (1, (1.0, 0.0, 0.0, 0.0)),
        (2, (0.5, 0.5, 0.0, 0.0)),
        (4, (0.25, 0.25, 0.25, 0.25)),
    ]:
        try:
            scores[f"bleu_{n}"] = sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoother)
        except Exception:
            scores[f"bleu_{n}"] = 0.0

    return scores


def compute_rouge(prediction: str, references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L using rouge-score.
    Returns dict: {"rouge_1": ..., "rouge_2": ..., "rouge_l": ...}
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        # Average over all references
        r1_vals, r2_vals, rl_vals = [], [], []
        for ref in references:
            scores = scorer.score(ref, prediction)
            r1_vals.append(scores["rouge1"].fmeasure)
            r2_vals.append(scores["rouge2"].fmeasure)
            rl_vals.append(scores["rougeL"].fmeasure)
        return {
            "rouge_1": float(np.mean(r1_vals)),
            "rouge_2": float(np.mean(r2_vals)),
            "rouge_l": float(np.mean(rl_vals)),
        }
    except ImportError:
        logger.warning("rouge-score not installed. Install with: pip install rouge-score")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


# Lazy-loaded CLIP model for CLIPScore
_clip_model = None
_clip_processor = None


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
            import torch

            logger.info("Loading CLIP model for CLIPScore computation...")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model = _clip_model.to(device)
            _clip_model.eval()
            logger.info("CLIP loaded.")
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")
    return _clip_model, _clip_processor


def compute_clip_score(prediction: str, image: Image.Image) -> float:
    """
    Compute CLIPScore: cosine similarity between CLIP image embedding
    and CLIP text embedding of the prediction.
    Score range: [0, 100]
    """
    try:
        import torch

        model, processor = _get_clip()
        if model is None or processor is None:
            return 0.0

        device = next(model.parameters()).device
        inputs = processor(
            text=[prediction],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds  # (1, D)
            txt_emb = outputs.text_embeds   # (1, D)

        # Normalized cosine similarity
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        score = float((img_emb * txt_emb).sum().clamp(min=0.0)) * 100.0
        return score

    except Exception as e:
        logger.warning(f"CLIPScore computation failed: {e}")
        return 0.0


# Lazy-loaded sentence-transformer for semantic similarity
_st_model = None


def _get_sentence_transformer():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading sentence-transformers for semantic similarity...")
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence transformer loaded.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    return _st_model


def compute_semantic_similarity(prediction: str, references: List[str]) -> float:
    """
    Compute semantic similarity using sentence-transformers.
    Returns max cosine similarity between prediction and any reference.
    Range: [0, 1]
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        model = _get_sentence_transformer()
        if model is None:
            # Fallback: simple word overlap Jaccard
            pred_words = set(prediction.lower().split())
            max_sim = 0.0
            for ref in references:
                ref_words = set(ref.lower().split())
                if pred_words or ref_words:
                    jacc = len(pred_words & ref_words) / len(pred_words | ref_words)
                    max_sim = max(max_sim, jacc)
            return max_sim

        all_texts = [prediction] + references
        embeddings = model.encode(all_texts, convert_to_numpy=True)
        pred_emb = embeddings[0:1]
        ref_embs = embeddings[1:]
        sims = cosine_similarity(pred_emb, ref_embs)[0]
        return float(np.max(sims))

    except Exception as e:
        logger.warning(f"Semantic similarity computation failed: {e}")
        return 0.0


def compute_exact_match(prediction: str, references: List[str]) -> float:
    """Binary exact match (case-insensitive, stripped)."""
    pred = prediction.strip().lower()
    for ref in references:
        if pred == ref.strip().lower():
            return 1.0
    return 0.0


def compute_accuracy(prediction: str, references: List[str]) -> float:
    """Accuracy: 1.0 if prediction contains any reference answer."""
    pred_lower = prediction.lower()
    for ref in references:
        if ref.lower() in pred_lower or pred_lower in ref.lower():
            return 1.0
    return 0.0


def compute_bert_score(prediction: str, references: List[str]) -> float:
    """
    Compute BERTScore F1 using the bert-score library.
    Returns average F1 across all references.
    """
    try:
        from bert_score import score as bert_score_fn  # type: ignore

        cands = [prediction] * len(references)
        P, R, F = bert_score_fn(cands, references, lang="en", verbose=False)
        return float(F.mean().item())
    except ImportError:
        logger.warning("bert-score not installed. Install with: pip install bert-score")
        return 0.0
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────

class MetricRegistry:
    """
    Computes all configured metrics for a (prediction, references, image) triplet.

    Usage
    -----
    >>> registry = MetricRegistry()
    >>> scores = registry.compute(
    ...     prediction="A cat sitting on a mat",
    ...     references=["A cat on a mat", "Feline resting on rug"],
    ...     image=pil_image,
    ...     metric_cfg=MetricConfig(bleu=True, clip_score=True),
    ... )
    """

    def compute(
        self,
        prediction: str,
        references: List[str],
        image: Optional[Image.Image],
        metric_cfg: MetricConfig,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        if not references:
            references = [""]

        if metric_cfg.bleu:
            scores.update(compute_bleu(prediction, references))

        if metric_cfg.rouge:
            scores.update(compute_rouge(prediction, references))

        if metric_cfg.bert_score:
            scores["bert_score_f1"] = compute_bert_score(prediction, references)

        if metric_cfg.clip_score and image is not None:
            scores["clip_score"] = compute_clip_score(prediction, image)

        if metric_cfg.semantic_similarity:
            scores["semantic_similarity"] = compute_semantic_similarity(prediction, references)

        if metric_cfg.exact_match:
            scores["exact_match"] = compute_exact_match(prediction, references)

        if metric_cfg.accuracy:
            scores["accuracy"] = compute_accuracy(prediction, references)

        return scores

    def list_metrics(self) -> List[str]:
        return [
            "bleu_1", "bleu_2", "bleu_4",
            "rouge_1", "rouge_2", "rouge_l",
            "bert_score_f1",
            "clip_score",
            "semantic_similarity",
            "exact_match",
            "accuracy",
        ]
