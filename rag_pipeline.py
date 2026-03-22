"""
RAG utilities: chunk reviews, embed with sentence-transformers, retrieve by cosine
similarity, then answer with a local Ollama chat model (llama3.2).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Multilingual model suited for French semantic search
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama (fixed model for this project)
OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

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


def _sanitize_generated_answer(text: str) -> str:
    """Light cleanup of leading numbering or junk prefixes in model output."""
    t = text.strip()
    t = re.sub(r"^(?:Excerpt\s*\d+\s*[.:]?\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\d{1,2}\s*[.)]\s*", "", t)
    t = re.sub(
        r"^[—\-–]\s*Avis\s*\([^)]+\)\s*[.:]?\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


def build_ollama_user_message(question: str, retrieved: List[Dict[str, Any]], max_context_chars: int = 12000) -> str:
    """User message for Ollama chat: French context + question."""
    parts = []
    for item in retrieved:
        m = item["meta"]
        ins = str(m.get("assureur", "") or "").strip() or "inconnu"
        note = m.get("note", "")
        parts.append(f"- {item['text']}\n  (assureur: {ins}, note: {note})")
    ctx = "\n".join(parts)
    ctx = _truncate_context(ctx, max_context_chars)
    return (
        "Tu es analyste sur des avis clients d'assurance (textes en français).\n"
        "À partir du contexte ci-dessous uniquement, réponds à la question en français.\n"
        "Écris un paragraphe de 4 à 8 phrases : synthétise les thèmes (satisfaction, problèmes, délais, "
        "téléphone, moto, etc.) sans inventer de faits absents du contexte.\n"
        "Ne commence pas par une liste numérotée.\n\n"
        f"Contexte :\n{ctx}\n\n"
        f"Question : {question}"
    )


def generate_answer_ollama(
    user_message: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    timeout_s: float = 180.0,
) -> str:
    """Call Ollama /api/chat (stdlib urllib)."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/api/chat"
    system = (
        "Tu réponds toujours en français. Tu t'appuies uniquement sur le contexte fourni par l'utilisateur. "
        "Si le contexte ne suffit pas, dis-le en une phrase au lieu d'inventer."
    )
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "options": {"temperature": 0.3, "num_predict": 512},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {url}. Is `ollama serve` running? ({e.reason})"
        ) from e
    msg = (payload.get("message") or {}).get("content") or ""
    return _sanitize_generated_answer(msg.strip())


def answer_question(
    question: str,
    embeddings: np.ndarray,
    chunks: List[str],
    meta: List[Dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = 5,
    embedder: Optional[Any] = None,
    ollama_base_url: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve excerpts, then generate an answer with Ollama (llama3.2)."""
    retrieved = retrieve(
        question,
        embeddings,
        chunks,
        meta,
        embed_model=embed_model,
        top_k=top_k,
        embedder=embedder,
    )
    user_msg = build_ollama_user_message(question, retrieved)
    base = ollama_base_url or DEFAULT_OLLAMA_BASE_URL
    answer = generate_answer_ollama(user_msg, base_url=base, model=OLLAMA_MODEL)
    return answer, retrieved
