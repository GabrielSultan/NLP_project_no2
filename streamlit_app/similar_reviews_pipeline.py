"""
Recherche d'avis similaires : BM25 (top 50) → bi-encodeur all-MiniLM-L6-v2 (top 25)
→ top 10 → rerank top 5 via scores Ollama (llama3.2), sur un sous-ensemble du CSV.

Inspiré du pipeline Projet 1 (BM25 + reranking + Ollama).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# Bi-encoder demandé (anglais ; OK pour lexique partagé assurance)
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BM25_TOP_K = 50
BIENCODER_TOP_K = 25
OLLAMA_POOL_K = 10
FINAL_TOP_K = 5

MAX_CHARS_OLLAMA = 1200

OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def tokenize_fr(text: str) -> List[str]:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    s = str(text).lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    tokens = word_tokenize(s)
    sw = set(stopwords.words("french"))
    return [t for t in tokens if t not in sw and len(t) > 2]


@dataclass
class SimilarReviewHit:
    corpus_index: int
    text: str
    meta: Dict[str, Any]
    bm25_score: float = 0.0
    biencoder_score: float = 0.0
    ollama_score: Optional[float] = None


@dataclass
class SimilarReviewsResult:
    query: str
    final: List[SimilarReviewHit] = field(default_factory=list)
    stage_bm25: List[SimilarReviewHit] = field(default_factory=list)
    stage_biencoder: List[SimilarReviewHit] = field(default_factory=list)
    stage_ollama_pool: List[SimilarReviewHit] = field(default_factory=list)
    ollama_used: bool = False
    ollama_error: Optional[str] = None


class SimilarReviewsIndex:
    """Corpus limité : textes, BM25, embeddings bi-encodeur normalisés."""

    def __init__(
        self,
        texts: List[str],
        metas: List[Dict[str, Any]],
        tokenized: List[List[str]],
        bm25: BM25Okapi,
        embeddings: np.ndarray,
        embed_model_name: str,
    ) -> None:
        self.texts = texts
        self.metas = metas
        self.tokenized = tokenized
        self.bm25 = bm25
        self.embeddings = embeddings.astype(np.float32)
        self.embed_model_name = embed_model_name

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        max_rows: int = 4000,
        text_column: str = "avis",
        embed_model_name: str = BI_ENCODER_MODEL,
        embedder: Any = None,
    ) -> SimilarReviewsIndex:
        df = pd.read_csv(csv_path, encoding="utf-8")
        df = df.head(int(max_rows))
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            raw = row.get(text_column, "")
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            s = str(raw).strip()
            if not s:
                continue
            texts.append(s)
            metas.append(
                {
                    "row_index": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                    "note": row.get("note", ""),
                    "assureur": row.get("assureur", ""),
                }
            )
        if not texts:
            raise ValueError("Aucun avis valide dans le sous-ensemble CSV.")
        tokenized = [tokenize_fr(t) for t in texts]
        tokenized = [t if t else ["vide"] for t in tokenized]
        bm25 = BM25Okapi(tokenized)
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(embed_model_name)
        emb = embedder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cls(texts, metas, tokenized, bm25, emb.astype(np.float32), embed_model_name)

    def encode_query(self, query: str, embedder: Any) -> np.ndarray:
        q = embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)
        return q


def _ollama_similarity_score(
    query: str,
    candidate: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_s: float = 90.0,
) -> float:
    """Score 0–10 : similarité entre deux avis (une requête /api/generate)."""
    q = (query or "")[:MAX_CHARS_OLLAMA]
    c = (candidate or "")[:MAX_CHARS_OLLAMA]
    prompt = (
        "Tu compares deux avis clients sur des assureurs (français). "
        "Évalue la similarité du fond (problèmes, satisfaction, thèmes) sur une échelle de 0 à 10 "
        "(10 = très similaires). Réponds par UN SEUL nombre décimal, rien d'autre.\n\n"
        f"Avis de référence :\n{q}\n\n"
        f"Avis à comparer :\n{c}\n\n"
        "Score (0-10) :"
    )
    url = base_url.rstrip("/") + "/api/generate"
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    response = (payload.get("response") or "").strip()
    m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.\d+)?)\b", response)
    if not m:
        return 0.0
    val = float(m.group(1))
    return max(0.0, min(10.0, val))


def find_similar_reviews(
    index: SimilarReviewsIndex,
    query: str,
    embedder: Any,
    ollama_base_url: Optional[str] = None,
    use_ollama_rerank: bool = True,
    exclude_corpus_indices: Optional[set] = None,
) -> SimilarReviewsResult:
    """
    Pipeline : BM25 top 50 → bi-encodeur top 25 → top 10 → Ollama scores → top 5.
    Si Ollama échoue ou use_ollama_rerank=False : top 5 = 5 premiers du top 10 bi-encodeur.
    """
    out = SimilarReviewsResult(query=(query or "").strip())
    q = out.query
    if not q:
        return out

    exclude_corpus_indices = exclude_corpus_indices or set()
    n = len(index.texts)
    if n == 0:
        return out

    q_tokens = tokenize_fr(q)
    if not q_tokens:
        q_tokens = ["vide"]

    bm25_scores = np.asarray(index.bm25.get_scores(q_tokens), dtype=np.float64)
    k_bm = min(BM25_TOP_K, n)
    idx_bm = np.argpartition(-bm25_scores, k_bm - 1)[:k_bm]
    idx_bm = idx_bm[np.argsort(-bm25_scores[idx_bm])]

    q_emb = index.encode_query(q, embedder)
    sub_emb = index.embeddings[idx_bm]
    bi_scores = (sub_emb @ q_emb).astype(np.float64)
    order_bi = np.argsort(-bi_scores)
    k_bi = min(BIENCODER_TOP_K, len(order_bi))
    order_bi = order_bi[:k_bi]
    idx_bi = idx_bm[order_bi]

    k_pool = min(OLLAMA_POOL_K, len(idx_bi))
    idx_pool = idx_bi[:k_pool]

    def hit_for(i: int, bm_s: float = 0.0, bi_s: float = 0.0) -> SimilarReviewHit:
        return SimilarReviewHit(
            corpus_index=int(i),
            text=index.texts[i],
            meta=dict(index.metas[i]),
            bm25_score=float(bm_s),
            biencoder_score=float(bi_s),
        )

    # Stage BM25 (top 50)
    for j, i in enumerate(idx_bm):
        ii = int(i)
        if ii in exclude_corpus_indices:
            continue
        out.stage_bm25.append(hit_for(ii, bm_s=float(bm25_scores[ii]), bi_s=0.0))

    # Stage bi-encoder (top 25 among BM25)
    for j in range(k_bi):
        pos = int(order_bi[j])
        ii = int(idx_bm[pos])
        if ii in exclude_corpus_indices:
            continue
        out.stage_biencoder.append(
            hit_for(ii, bm_s=float(bm25_scores[ii]), bi_s=float(bi_scores[pos]))
        )

    pool_hits: List[SimilarReviewHit] = []
    for j in range(k_pool):
        ii = int(idx_pool[j])
        if ii in exclude_corpus_indices:
            continue
        loc = np.where(idx_bm == ii)[0]
        if len(loc):
            pos = int(loc[0])
            bi_s = float(bi_scores[pos])
            bm_s = float(bm25_scores[ii])
        else:
            bi_s = 0.0
            bm_s = float(bm25_scores[ii])
        pool_hits.append(hit_for(ii, bm_s=bm_s, bi_s=bi_s))

    out.stage_ollama_pool = list(pool_hits)

    base = ollama_base_url or DEFAULT_OLLAMA_BASE_URL
    if use_ollama_rerank and pool_hits:
        try:
            scored: List[Tuple[SimilarReviewHit, float]] = []
            for h in pool_hits:
                s = _ollama_similarity_score(q, h.text, base_url=base)
                h2 = SimilarReviewHit(
                    corpus_index=h.corpus_index,
                    text=h.text,
                    meta=h.meta,
                    bm25_score=h.bm25_score,
                    biencoder_score=h.biencoder_score,
                    ollama_score=s,
                )
                scored.append((h2, s))
            scored.sort(key=lambda x: -x[1])
            out.final = [h for h, _ in scored[:FINAL_TOP_K]]
            out.ollama_used = True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            out.ollama_error = str(e)
            out.final = pool_hits[:FINAL_TOP_K]
    else:
        out.final = pool_hits[:FINAL_TOP_K]

    return out
