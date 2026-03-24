"""
Thematic classification inference (DistilBERT multilingual, sequence classification).

Weights live under `artifacts/distilbert_thematic/` (tokenizer + model + config with id2label),
produced by notebook section 7.2 via Hugging Face `save_pretrained`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

# Max sequence length for tokenization (must match notebook / training)
DEFAULT_MAX_LENGTH = 128


def thematic_model_path(artifacts_dir: str) -> str:
    """Return the full path to the DistilBERT thematic model directory inside artifacts."""
    return os.path.join(artifacts_dir, "distilbert_thematic")


def is_thematic_model_ready(artifacts_dir: str) -> bool:
    """Check if the thematic model directory is present and complete (config and weights exist)."""
    d = thematic_model_path(artifacts_dir)
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(d, name)):
            return True
    return False


def _id2label_list(config: Any) -> List[str]:
    """
    Convert the id2label mapping in the config to a list form.
    The ith element in the list is the label for class index i.
    """
    raw = getattr(config, "id2label", None) or {}
    if not raw:
        return []
    n = len(raw)
    out = [""] * n
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def load_thematic_bundle(artifacts_dir: str) -> Dict[str, Any]:
    """
    Load the tokenizer and the fine-tuned DistilBERT model for thematic classification,
    and return a bundle of objects for inference.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = thematic_model_path(artifacts_dir)
    if not is_thematic_model_ready(artifacts_dir):
        raise FileNotFoundError(
            f"Model not found or incomplete in {path}. "
            "Run the DistilBERT (7.2) cell in the notebook to train and save the model."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = _id2label_list(model.config)
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "labels": labels,
        "max_length": DEFAULT_MAX_LENGTH,
    }


def predict_thematic_proba(
    bundle: Dict[str, Any],
    text: str,
    max_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Return the list of (label, probability) pairs predicted by the model, sorted in descending order of probability.
    """
    import torch

    t = (text or "").strip()
    if not t:
        return []

    tok = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    labels: List[str] = bundle["labels"]
    ml = int(max_length or bundle.get("max_length") or DEFAULT_MAX_LENGTH)

    enc = tok(
        t,
        truncation=True,
        padding="max_length",
        max_length=ml,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    # Single-sequence batch → one row of class logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
    if labels and len(labels) == len(probs):
        pairs = list(zip(labels, [float(p) for p in probs]))
    else:
        pairs = [(str(i), float(probs[i])) for i in range(len(probs))]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def bundle_mtime(artifacts_dir: str) -> float:
    """
    Find the most recent modification time of the model weights or config file
    (used to invalidate Streamlit cache).
    """
    d = thematic_model_path(artifacts_dir)
    best = 0.0
    for name in ("model.safetensors", "pytorch_model.bin", "config.json"):
        p = os.path.join(d, name)
        try:
            best = max(best, os.path.getmtime(p))
        except OSError:
            pass
    return best
