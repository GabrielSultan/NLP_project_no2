"""
Orchestrate a single user review through all Streamlit “full pipeline” models.

Order of operations:
  1) Preprocess text for theme (lemmatized string).
  2) DistilBERT theme probabilities (optional if model missing).
  3) French abstractive summary on raw text (BARThez).
  4) Multilingual 5-star sentiment → coarse polarity on raw text.
  5) Similar-review search on raw text (BM25 → MiniLM → optional Ollama rerank).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import review_preprocess
import thematic_distilbert as thematic

from similar_reviews_pipeline import (
    DEFAULT_OLLAMA_BASE_URL,
    SimilarReviewsIndex,
    SimilarReviewsResult,
    find_similar_reviews,
)

SUMMARY_MODEL = "moussaKam/barthez-orangesum-abstract"
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# BARThez: skip summarization when input is too short to be meaningful
MIN_WORDS_SUMMARY = 28
MAX_SUMMARY_INPUT_CHARS = 2500


@dataclass
class ReviewAnalysisResult:
    """Aggregated outputs for one pasted review (UI displays each block separately)."""

    raw_text: str
    thematic_processed: str  # lemmatized string fed to DistilBERT
    thematic_probs: List[Tuple[str, float]] = field(default_factory=list)
    thematic_best: Optional[Tuple[str, float]] = None
    summary_fr: str = ""
    sentiment_label: str = ""  # Positive / Negative / Neutral after star mapping
    sentiment_score: float = 0.0
    similar: Optional[SimilarReviewsResult] = None
    summary_skipped_reason: Optional[str] = None
    thematic_error: Optional[str] = None


def _map_stars_to_polarity(label: str, score: float) -> Tuple[str, float]:
    """Map nlptown 1–5 star label to three-way polarity for display."""
    m = re.search(r"([1-5])", str(label) or "")
    stars = int(m.group(1)) if m else 3
    if stars >= 4:
        return "Positive", float(score)
    if stars <= 2:
        return "Negative", float(score)
    return "Neutral", float(score)


def summarize_french(text: str, summarizer: Any) -> Tuple[str, Optional[str]]:
    """Returns (summary, skip reason if any)."""
    words = str(text).split()
    if len(words) < MIN_WORDS_SUMMARY:
        return str(text).strip(), "Text too short for automatic summary (raw text shown)."
    t = str(text).strip()[:MAX_SUMMARY_INPUT_CHARS]
    try:
        out = summarizer(
            t,
            max_length=min(120, max(32, len(words) // 3)),
            min_length=max(8, min(36, len(words) // 4)),
            do_sample=False,
            truncation=True,
        )
        if isinstance(out, list) and out:
            st = out[0].get("summary_text", "").strip()
            return (st if st else t, None)
    except Exception:
        pass
    return t, "Summarization model failed; truncated raw text shown."


def sentiment_multilingual(text: str, pipe: Any) -> Tuple[str, float]:
    t = (text or "").strip()
    if not t:
        return "Unknown", 0.0
    chunk = t[:2000]
    r = pipe(chunk)[0]
    return _map_stars_to_polarity(r.get("label", ""), r.get("score", 0.0))


def run_review_analysis(
    raw_text: str,
    artifacts_dir: str,
    thematic_bundle: Optional[Dict[str, Any]],
    similar_index: SimilarReviewsIndex,
    biencoder: Any,
    summarizer: Any,
    sentiment_pipe: Any,
    use_ollama_similarity: bool = True,
    ollama_base_url: Optional[str] = None,
) -> ReviewAnalysisResult:
    raw = (raw_text or "").strip()
    out = ReviewAnalysisResult(raw_text=raw, thematic_processed="")

    if not raw:
        return out

    # Theme model expects notebook-aligned lemmas; other heads use raw text
    processed = review_preprocess.preprocess_like_avis_traite(raw, artifacts_dir)
    out.thematic_processed = processed

    if thematic_bundle and processed:
        try:
            probs = thematic.predict_thematic_proba(thematic_bundle, processed)
            out.thematic_probs = probs
            out.thematic_best = probs[0] if probs else None
        except Exception as e:
            out.thematic_error = str(e)
    elif thematic_bundle and not processed:
        out.thematic_error = "Empty text after preprocessing (theme prediction skipped)."

    out.summary_fr, out.summary_skipped_reason = summarize_french(raw, summarizer)
    out.sentiment_label, out.sentiment_score = sentiment_multilingual(raw, sentiment_pipe)

    # Similarity index is built from raw `avis` rows; query with original paste
    out.similar = find_similar_reviews(
        similar_index,
        raw,
        embedder=biencoder,
        ollama_base_url=ollama_base_url or DEFAULT_OLLAMA_BASE_URL,
        use_ollama_rerank=use_ollama_similarity,
    )
    return out
