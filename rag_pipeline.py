"""
RAG utilities: chunk reviews, embed with sentence-transformers, retrieve by cosine
similarity, and answer with a seq2seq LLM (multilingual instruction-tuned mT0).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Multilingual model suited for French semantic search
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GEN_MODEL = "bigscience/mt0-base"

RAG_DIRNAME = "rag"


def _rag_dir(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, RAG_DIRNAME)


def config_fingerprint(csv_path: str, max_documents: int, chunk_max_words: int, overlap_words: int) -> str:
    """Stable id so we know when the on-disk index matches build settings."""
    try:
        st = os.stat(csv_path)
        raw = f"{os.path.abspath(csv_path)}|{st.st_size}|{int(st.st_mtime)}|{max_documents}|{chunk_max_words}|{overlap_words}"
    except OSError:
        raw = f"{csv_path}|missing|{max_documents}|{chunk_max_words}|{overlap_words}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def chunk_text(text: str, max_words: int = 90, overlap_words: int = 20) -> List[str]:
    """Split a long review into overlapping word windows."""
    words = str(text).split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, max_words - overlap_words)
    i = 0
    while i < len(words):
        piece = words[i : i + max_words]
        chunks.append(" ".join(piece))
        i += step
    return chunks


def build_chunks_from_dataframe(
    df: pd.DataFrame,
    max_documents: int,
    text_column: str = "avis",
    chunk_max_words: int = 90,
    overlap_words: int = 20,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build parallel lists of chunk strings and metadata (insurer, rating, source row).
    """
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    n = min(len(df), max_documents)
    subset = df.iloc[:n]
    for row_idx, row in subset.iterrows():
        raw = row.get(text_column, "")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        s = str(raw).strip()
        if not s:
            continue
        parts = chunk_text(s, max_words=chunk_max_words, overlap_words=overlap_words)
        note = row.get("note", "")
        insurer = row.get("assureur", "")
        for j, part in enumerate(parts):
            texts.append(part)
            meta.append(
                {
                    "row_index": int(row_idx) if isinstance(row_idx, (int, np.integer)) else row_idx,
                    "chunk_index": j,
                    "note": note,
                    "assureur": insurer,
                }
            )
    return texts, meta


def encode_chunks(
    chunks: List[str],
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode all chunks to L2-normalized rows for cosine similarity via dot product."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        chunks,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return emb.astype(np.float32)


def save_index(
    artifacts_dir: str,
    embeddings: np.ndarray,
    chunks: List[str],
    meta: List[Dict[str, Any]],
    embed_model: str,
    fingerprint: str,
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    d = _rag_dir(artifacts_dir)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "embeddings.npy"), embeddings)
    with open(os.path.join(d, "chunks.pkl"), "wb") as f:
        pickle.dump({"texts": chunks, "meta": meta}, f)
    cfg: Dict[str, Any] = {
        "embed_model": embed_model,
        "fingerprint": fingerprint,
        "num_vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
    }
    if extra_config:
        cfg.update(extra_config)
    with open(os.path.join(d, "build_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_index(artifacts_dir: str) -> Optional[Tuple[np.ndarray, List[str], List[Dict[str, Any]], Dict[str, Any]]]:
    d = _rag_dir(artifacts_dir)
    emb_path = os.path.join(d, "embeddings.npy")
    chk_path = os.path.join(d, "chunks.pkl")
    cfg_path = os.path.join(d, "build_config.json")
    if not (os.path.isfile(emb_path) and os.path.isfile(chk_path) and os.path.isfile(cfg_path)):
        return None
    embeddings = np.load(emb_path)
    with open(chk_path, "rb") as f:
        blob = pickle.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return embeddings, blob["texts"], blob["meta"], cfg


def index_matches_fingerprint(artifacts_dir: str, expected_fingerprint: str) -> bool:
    loaded = load_index(artifacts_dir)
    if loaded is None:
        return False
    _, _, _, cfg = loaded
    return cfg.get("fingerprint") == expected_fingerprint


def build_and_save_index(
    csv_path: str,
    artifacts_dir: str,
    max_documents: int = 3000,
    chunk_max_words: int = 90,
    overlap_words: int = 20,
    embed_model: str = DEFAULT_EMBED_MODEL,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Load CSV, chunk, embed, persist under artifacts/rag/."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    chunks, meta = build_chunks_from_dataframe(
        df,
        max_documents=max_documents,
        chunk_max_words=chunk_max_words,
        overlap_words=overlap_words,
    )
    if not chunks:
        raise ValueError("No text chunks produced; check CSV path and column 'avis'.")
    fp = config_fingerprint(csv_path, max_documents, chunk_max_words, overlap_words)
    embeddings = encode_chunks(chunks, model_name=embed_model, show_progress=show_progress)
    save_index(
        artifacts_dir,
        embeddings,
        chunks,
        meta,
        embed_model,
        fp,
        extra_config={
            "max_documents": max_documents,
            "chunk_max_words": chunk_max_words,
            "overlap_words": overlap_words,
            "source_csv": os.path.basename(csv_path),
        },
    )
    return embeddings, chunks, meta


def retrieve(
    query: str,
    embeddings: np.ndarray,
    chunks: List[str],
    meta: List[Dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 5,
    embedder: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Return top_k chunks with scores (cosine similarity, embeddings are pre-normalized)."""
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(embed_model)
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    scores = embeddings @ q
    k = min(top_k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    out: List[Dict[str, Any]] = []
    for i in top_idx:
        out.append(
            {
                "text": chunks[int(i)],
                "score": float(scores[int(i)]),
                "meta": meta[int(i)],
            }
        )
    return out


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]], max_context_chars: int = 2200) -> str:
    """
    English task wording for mT0. Put the question before the excerpts so HF truncation (tail /
    keep-prefix) does not drop the question on long contexts. Avoid quoting a fixed French
    refusal in the prompt or seq2seq models often echo it.
    """
    parts = []
    for i, item in enumerate(retrieved, start=1):
        m = item["meta"]
        parts.append(
            f"Excerpt {i} (rating {m.get('note', '')}, insurer {m.get('assureur', '')}): {item['text']}"
        )
    ctx = "\n".join(parts)
    ctx = _truncate_context(ctx, max_context_chars)
    return (
        "Answer in French only. Use the excerpts below; be short and factual. "
        "Do not repeat excerpt numbers or metadata. "
        "If the excerpts are only partly related, still summarize what they say about the topic.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{ctx}\n\n"
        "Réponse en français:"
    )


def _safe_max_input_tokens(tokenizer: Any) -> int:
    m = getattr(tokenizer, "model_max_length", None)
    if m is None or m > 4096:
        return 1024
    return int(m)


def generate_answer(
    prompt: str,
    model_name: str = DEFAULT_GEN_MODEL,
    max_new_tokens: int = 256,
    generator: Optional[Any] = None,
) -> str:
    """Run seq2seq generation (mT0 / mT5 family or compatible text2text model)."""
    if generator is None:
        from transformers import pipeline

        generator = pipeline("text2text-generation", model=model_name, max_length=1024)
    tok = getattr(generator, "tokenizer", None)
    max_in = 1024
    if tok is not None:
        max_in = _safe_max_input_tokens(tok)
    # Question is at the start of the prompt; default truncation keeps the prefix (question + early excerpts).
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
        max_length=max_in,
    )
    return out[0]["generated_text"].strip()


def answer_question(
    question: str,
    embeddings: np.ndarray,
    chunks: List[str],
    meta: List[Dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    top_k: int = 5,
    embedder: Optional[Any] = None,
    generator: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve then generate in one call."""
    retrieved = retrieve(
        question,
        embeddings,
        chunks,
        meta,
        embed_model=embed_model,
        top_k=top_k,
        embedder=embedder,
    )
    prompt = build_rag_prompt(question, retrieved)
    answer = generate_answer(prompt, model_name=gen_model, generator=generator)
    return answer, retrieved
