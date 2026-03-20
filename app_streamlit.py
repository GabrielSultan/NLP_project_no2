"""
Streamlit app: RAG / QA on customer reviews.

How to launch:
    python app_streamlit.py
        Starts the Streamlit server via bootstrap (works if `streamlit` is not on PATH).

    python -m streamlit run app_streamlit.py
        Standard entry point; use this if the bare `streamlit` command is not found (common on Windows).
"""
import os

import streamlit as st
from transformers import pipeline

import rag_pipeline as rag

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"


@st.cache_resource
def load_rag_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(rag.DEFAULT_EMBED_MODEL)


@st.cache_resource
def load_rag_generator():
    return pipeline("text2text-generation", model=rag.DEFAULT_GEN_MODEL, max_length=1024)


def _rag_embeddings_mtime(artifacts_dir: str) -> float:
    path = os.path.join(artifacts_dir, rag.RAG_DIRNAME, "embeddings.npy")
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_resource
def load_rag_vector_bundle(artifacts_dir: str, _embed_mtime: float):
    """Reload when embeddings.npy is created or replaced (_embed_mtime changes)."""
    return rag.load_index(artifacts_dir)


def main():
    st.title("Insurance Reviews - NLP Analysis")
    st.subheader("RAG / QA on customer reviews")
    st.caption(
        "Dense retrieval with multilingual sentence embeddings, then mT0 (bigscience/mt0-base) generates "
        "a French answer from the top matching excerpts."
    )
    csv_path = os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv")
    mtime = _rag_embeddings_mtime(ARTIFACTS_DIR)
    bundle = load_rag_vector_bundle(ARTIFACTS_DIR, mtime)

    if bundle is None:
        st.info(
            "No vector index on disk. Recommended: run `python scripts/build_rag_index.py` from the "
            "project root (adjust `--max-documents` as needed). You can also build a smaller index here."
        )
        max_docs = st.number_input(
            "Max rows to index (from the start of the CSV)",
            min_value=100,
            max_value=50000,
            value=1200,
            step=100,
            key="rag_max_docs",
        )
        if st.button("Build embedding index now", key="rag_build"):
            if not os.path.isfile(csv_path):
                st.error(f"CSV not found: {csv_path}")
            else:
                with st.spinner("Chunking and encoding reviews (downloads model on first run)..."):
                    rag.build_and_save_index(
                        csv_path=csv_path,
                        artifacts_dir=ARTIFACTS_DIR,
                        max_documents=int(max_docs),
                        show_progress=False,
                    )
                st.success("Index saved under artifacts/rag/. Refreshing.")
                st.rerun()
    else:
        embeddings, chunks, meta, cfg = bundle
        st.success(
            f"Loaded index: **{cfg.get('num_vectors', embeddings.shape[0])}** vectors "
            f"({cfg.get('dim', embeddings.shape[1])} dims) — embedder: `{cfg.get('embed_model', '')}`"
        )
        if cfg.get("max_documents") is not None:
            st.caption(
                f"Build settings: max_documents={cfg.get('max_documents')}, "
                f"chunk_max_words={cfg.get('chunk_max_words')}, source={cfg.get('source_csv')}"
            )

        md_cfg = cfg.get("max_documents")
        if md_cfg is not None and cfg.get("fingerprint"):
            expected_fp = rag.config_fingerprint(
                csv_path,
                int(md_cfg),
                int(cfg.get("chunk_max_words", 90) or 90),
                int(cfg.get("overlap_words", 20) or 20),
            )
            if expected_fp != cfg.get("fingerprint"):
                st.warning(
                    "The CSV file or indexing parameters may have changed since this index was built. "
                    "Consider rebuilding for consistent results."
                )

        question = st.text_input("Ask a question about the reviews:", key="rag_question")
        top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=12, value=5, key="rag_topk")

        if st.button("Retrieve & generate answer", key="rag_answer_btn"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                embedder = load_rag_embedder()
                gen = load_rag_generator()
                with st.spinner("Retrieving relevant excerpts and generating..."):
                    answer, retrieved = rag.answer_question(
                        question.strip(),
                        embeddings,
                        chunks,
                        meta,
                        top_k=int(top_k),
                        embedder=embedder,
                        generator=gen,
                    )
                st.markdown("### Generated answer")
                st.write(answer)
                with st.expander("Retrieved excerpts (with similarity scores)"):
                    for item in retrieved:
                        m = item["meta"]
                        st.markdown(
                            f"**score** `{item['score']:.3f}` — **rating** {m.get('note', '')} — "
                            f"**insurer** {m.get('assureur', '')}"
                        )
                        st.text(item["text"][:1200] + ("..." if len(item["text"]) > 1200 else ""))


if __name__ == "__main__":
    # `python app_streamlit.py` runs in "raw mode" (no Runtime) and breaks widgets/session state.
    # Delegate to the Streamlit server; `streamlit run app_streamlit.py` already has a Runtime.
    from pathlib import Path

    import streamlit.runtime as st_runtime
    from streamlit.web import bootstrap

    if not st_runtime.exists():
        bootstrap.run(str(Path(__file__).resolve()), False, [], {})
    else:
        main()
