"""
Streamlit app: RAG / QA on customer reviews (Ollama + llama3.2 for answers).

Prerequisite for *generating* answers (not for building the vector index):
    1. Install Ollama and keep it running (Windows/macOS: open the Ollama app; Linux: run `ollama serve` if needed).
    2. Pull the model once: `ollama pull llama3.2`
    3. Then start this app and click "Retrieve & generate answer".

How to launch this app:
    python app_streamlit.py
        Starts the Streamlit server via bootstrap (works if `streamlit` is not on PATH).

    python -m streamlit run app_streamlit.py
        Standard entry point; use this if the bare `streamlit` command is not found (common on Windows).
"""
import os

import streamlit as st

import rag_pipeline as rag

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"

_ALL_INSURERS = "— Tous les assureurs —"


@st.cache_data
def insurer_choices_for_index(_mtime: float, artifacts_dir: str) -> list[str]:
    """Insurers present in the current vector index (metadata), sorted."""
    loaded = rag.load_index(artifacts_dir)
    if loaded is None:
        return []
    _, _, meta, _ = loaded
    return rag.unique_insurers_in_meta(meta)


@st.cache_resource
def load_rag_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(rag.DEFAULT_EMBED_MODEL)


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
    st.caption("Semantic search over reviews, then **Ollama** (`llama3.2`) generates the answer in French.")
    st.info(
        "**Avant de générer une réponse :** le serveur **Ollama doit tourner** (icône Ollama / appli lancée sur "
        "Windows ou macOS ; sur Linux : `ollama serve` si besoin). Une fois : `ollama pull llama3.2`. "
        "L’index vectoriel (RAG) peut être construit sans Ollama ; seule l’étape *génération* l’appelle."
    )

    csv_path = os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv")
    mtime = _rag_embeddings_mtime(ARTIFACTS_DIR)
    bundle = load_rag_vector_bundle(ARTIFACTS_DIR, mtime)

    if bundle is None:
        st.info("No vector index yet. Run `python scripts/build_rag_index.py` or build below.")
        max_docs = st.number_input(
            "Max rows to index (from the start of the CSV)",
            min_value=100,
            max_value=50000,
            value=1200,
            step=100,
            key="rag_max_docs",
        )
        if st.button("Build embedding index", key="rag_build"):
            if not os.path.isfile(csv_path):
                st.error(f"CSV not found: {csv_path}")
            else:
                with st.spinner("Indexing..."):
                    rag.build_and_save_index(
                        csv_path=csv_path,
                        artifacts_dir=ARTIFACTS_DIR,
                        max_documents=int(max_docs),
                        show_progress=False,
                    )
                st.success("Index saved. Refreshing.")
                st.rerun()
    else:
        embeddings, chunks, meta, cfg = bundle
        st.success(f"RAG index ready — **{cfg.get('num_vectors', embeddings.shape[0])}** chunks.")

        insurer_list = insurer_choices_for_index(mtime, ARTIFACTS_DIR)
        if not insurer_list:
            st.warning("No insurer names found in index metadata.")
        insurer_choice = st.selectbox(
            "Assureur (la recherche ne garde que les avis de cet assureur)",
            options=[_ALL_INSURERS] + insurer_list,
            index=0,
            key="rag_insurer",
        )
        insurer_filter = None if insurer_choice == _ALL_INSURERS else insurer_choice

        question = st.text_input("Ask a question about the reviews:", key="rag_question")
        top_k = st.slider("Chunks to retrieve", min_value=3, max_value=12, value=5, key="rag_topk")

        if st.button("Retrieve & generate answer", key="rag_answer_btn"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                embedder = load_rag_embedder()
                with st.spinner("Retrieving and calling Ollama..."):
                    try:
                        answer, retrieved = rag.answer_question(
                            question.strip(),
                            embeddings,
                            chunks,
                            meta,
                            top_k=int(top_k),
                            embedder=embedder,
                            insurer_filter=insurer_filter,
                        )
                    except RuntimeError as e:
                        st.error(str(e))
                        st.stop()
                st.markdown("### Generated answer")
                st.write(answer)
                with st.expander("Retrieved excerpts (scores)"):
                    for item in retrieved:
                        m = item["meta"]
                        st.markdown(
                            f"**score** `{item['score']:.3f}` — **rating** {m.get('note', '')} — "
                            f"**insurer** {m.get('assureur', '')}"
                        )
                        st.text(item["text"][:1200] + ("..." if len(item["text"]) > 1200 else ""))


if __name__ == "__main__":
    from pathlib import Path

    import streamlit.runtime as st_runtime
    from streamlit.web import bootstrap

    if not st_runtime.exists():
        bootstrap.run(str(Path(__file__).resolve()), False, [], {})
    else:
        main()
