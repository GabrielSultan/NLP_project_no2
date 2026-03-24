"""
French review preprocessing aligned with notebook column `avis_traite`.

Pipeline (in order): lowercase, apostrophe spacing, strip punctuation, NLTK tokenize,
remove stopwords / digits / very short tokens, lemmatize with simplemma.

Not applied here: spell-check, translation, summarization (raw user text is unchanged for those paths).
Optional: load stopword set from `artifacts/preprocess.pkl` when the notebook has produced it.
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from string import punctuation
from typing import FrozenSet, Optional


def _default_artifacts_dir() -> str:
    """Project root `artifacts/` (parent of `streamlit_app/`)."""
    return str((Path(__file__).resolve().parent.parent / "artifacts").resolve())


@lru_cache(maxsize=8)
def _french_stopwords(artifacts_dir_abs: str) -> FrozenSet[str]:
    """
    Load French stopwords from preprocess.pkl if available; otherwise, use NLTK's French stopwords and add "très".
    """
    pkl = os.path.join(artifacts_dir_abs, "preprocess.pkl")
    if os.path.isfile(pkl):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            sw = data.get("stopwords")
            if sw:
                return frozenset(sw)
        except (OSError, pickle.PickleError, TypeError):
            pass
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    s = set(stopwords.words("french"))
    s.add("très")
    return frozenset(s)


def preprocess_like_avis_traite(text: object, artifacts_dir: Optional[str] = None) -> str:
    """
    Reproduce the `preprocess_text` function from the notebook on a raw string (like `avis` / `avis_source`).
    If `artifacts_dir` is omitted, uses `<project_root>/artifacts`.
    """
    import nltk
    import simplemma

    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""

    nltk.download("punkt", quiet=True)

    if artifacts_dir is None:
        artifacts_dir = _default_artifacts_dir()
    artifacts_abs = os.path.abspath(artifacts_dir)
    stop = _french_stopwords(artifacts_abs)

    s_low = s.lower()
    s_low = s_low.replace("'", " ")  # avoid glued tokens after apostrophe removal
    s_low = "".join(c for c in s_low if c not in punctuation)
    tokens = nltk.word_tokenize(s_low)
    tokens = [
        t
        for t in tokens
        if t not in stop and len(t) > 1 and not any(c.isdigit() for c in t)
    ]
    # Fallback to surface form if lemmatizer returns nothing for a token
    lemmas = [simplemma.lemmatize(t, lang="fr") or t for t in tokens]
    return " ".join(lemmas)
