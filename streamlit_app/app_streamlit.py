"""
Streamlit app: full pipeline (theme + summary + sentiment + similar reviews BM25 → MiniLM → Ollama)
and RAG / QA (Ollama).

Ollama: `ollama pull llama3.2` (RAG and similar-review reranking).

Run from repo root:
    streamlit run streamlit_app/app_streamlit.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import rag_pipeline as rag
import review_analysis_pipeline as review_pipe
import thematic_distilbert as thematic
from similar_reviews_pipeline import BI_ENCODER_MODEL, SimilarReviewsIndex

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(_PROJECT_ROOT / "data")
ARTIFACTS_DIR = str(_PROJECT_ROOT / "artifacts")

_ALL_INSURERS = "— All insurers —"


@st.cache_data
def insurer_choices_for_index(_mtime: float, artifacts_dir: str) -> list[str]:
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
    return rag.load_index(artifacts_dir)


@st.cache_resource
def load_thematic_bundle_cached(artifacts_dir: str, _model_mtime: float):
    return thematic.load_thematic_bundle(artifacts_dir)


@st.cache_resource
def load_similarity_bundle(_csv_mtime: float, csv_path: str, max_rows: int):
    """Sub-corpus + all-MiniLM-L6-v2 bi-encoder (shared for BM25 stage and query encoding)."""
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer(BI_ENCODER_MODEL)
    idx = SimilarReviewsIndex.from_csv(
        csv_path, max_rows=int(max_rows), text_column="avis", embedder=emb
    )
    return idx, emb


@st.cache_resource
def load_summarizer_hf():
    from transformers import pipeline

    return pipeline("summarization", model=review_pipe.SUMMARY_MODEL)


@st.cache_resource
def load_sentiment_hf():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=review_pipe.SENTIMENT_MODEL)


def render_rag_tab() -> None:
    st.subheader("RAG / QA over reviews")
    st.caption("Semantic search over reviews, then **Ollama** (`llama3.2`) generates the answer in French.")
    st.info(
        "**To generate answers:** run **Ollama**; once: `ollama pull llama3.2`. "
        "The vector index can be built without Ollama."
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
                with st.spinner("Indexing…"):
                    rag.build_and_save_index(
                        csv_path=csv_path,
                        artifacts_dir=ARTIFACTS_DIR,
                        max_documents=int(max_docs),
                        show_progress=False,
                    )
                st.success("Index saved. Reloading…")
                st.rerun()
        return

    embeddings, chunks, meta, cfg = bundle
    st.success(f"RAG index ready — **{cfg.get('num_vectors', embeddings.shape[0])}** segments.")

    insurer_list = insurer_choices_for_index(mtime, ARTIFACTS_DIR)
    if not insurer_list:
        st.warning("No insurer names in index metadata.")
    insurer_choice = st.selectbox(
        "Insurer (filter search)",
        options=[_ALL_INSURERS] + insurer_list,
        index=0,
        key="rag_insurer",
    )
    insurer_filter = None if insurer_choice == _ALL_INSURERS else insurer_choice

    question = st.text_input("Ask a question about the reviews:", key="rag_question")
    top_k = st.slider("Segments to retrieve", min_value=3, max_value=12, value=5, key="rag_topk")

    if st.button("Retrieve and generate answer", key="rag_answer_btn"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            embedder = load_rag_embedder()
            with st.spinner("Retrieval + Ollama…"):
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


def render_full_pipeline_tab() -> None:
    st.subheader("Pipeline: theme, summary, sentiment, similar reviews")
    st.caption(
        "**Similarity:** BM25 → top 50, **all-MiniLM-L6-v2** bi-encoder → top 25, "
        "top 10 → **Ollama llama3.2** (0–10 scores) → **top 5**. CSV subset. (from NLP project 1)"
    )
    csv_path = os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv")
    if not os.path.isfile(csv_path):
        st.error(f"File not found: `{csv_path}`")
        return

    m_csv = os.path.getmtime(csv_path)
    max_rows = st.slider(
        "CSV rows (from the start)",
        min_value=400,
        max_value=12000,
        value=4000,
        step=200,
        key="pipe_max_rows",
    )
    use_ollama_sim = st.checkbox(
        "Final reranking with Ollama (10 calls per analysis)",
        value=True,
        key="pipe_ollama_sim",
    )
    if use_ollama_sim:
        st.caption("If Ollama is off: top 5 = first 5 of the bi-encoder top 10.")

    try:
        with st.spinner("Loading sub-corpus + embeddings (first run can be slow)…"):
            sim_index, bi_encoder = load_similarity_bundle(m_csv, csv_path, max_rows)
    except Exception as e:
        st.error(f"Could not build similarity index: {e}")
        return

    st.success(f"**{len(sim_index.texts)}** reviews indexed (BM25 + MiniLM).")

    summarizer = load_summarizer_hf()
    sentiment_pipe = load_sentiment_hf()

    thematic_bundle = None
    if thematic.is_thematic_model_ready(ARTIFACTS_DIR):
        m_model = thematic.bundle_mtime(ARTIFACTS_DIR)
        try:
            thematic_bundle = load_thematic_bundle_cached(ARTIFACTS_DIR, m_model)
        except Exception as e:
            st.warning(f"Thematic model not loaded: {e}")
    else:
        st.info(
            "DistilBERT theme model missing (`artifacts/distilbert_thematic/`). "
            "The rest of the pipeline still runs."
        )

    text = st.text_area("Review to analyze", height=180, key="pipeline_input")
    if st.button("Run pipeline", key="pipeline_run"):
        raw = (text or "").strip()
        if not raw:
            st.warning("Enter review text.")
            return
        with st.spinner("Theme, summary, sentiment, similar reviews…"):
            result = review_pipe.run_review_analysis(
                raw,
                ARTIFACTS_DIR,
                thematic_bundle,
                sim_index,
                bi_encoder,
                summarizer,
                sentiment_pipe,
                use_ollama_similarity=use_ollama_sim,
            )

        st.markdown("### Theme (DistilBERT on preprocessed text)")
        if result.thematic_error:
            st.warning(result.thematic_error)
        elif result.thematic_best:
            lab, p = result.thematic_best
            st.markdown(f"**{lab}** — confidence **{p:.1%}**")
            with st.expander("Class probabilities"):
                for a, b in result.thematic_probs:
                    st.write(f"- `{a}` : {b:.2%}")
        else:
            st.caption("Not computed.")

        st.markdown("### Summary (FR)")
        st.write(result.summary_fr or "—")
        if result.summary_skipped_reason:
            st.caption(result.summary_skipped_reason)

        st.markdown("### Sentiment")
        st.write(
            f"**{result.sentiment_label}** (model score: {result.sentiment_score:.3f})"
        )

        st.markdown("### Top 5 similar reviews (BM25 → bi-encoder → Ollama)")
        sim = result.similar
        if sim is None or not sim.final:
            st.info("No neighbors found.")
            return
        if sim.ollama_error:
            st.warning(f"Ollama (rerank): {sim.ollama_error} — fallback: bi-encoder order.")
        elif use_ollama_sim and sim.ollama_used:
            st.caption("Final ranking: Ollama 0–10 scores on bi-encoder top 10.")

        for rank, h in enumerate(sim.final, start=1):
            meta = h.meta
            st.markdown(
                f"**#{rank}** — rating {meta.get('note', '')} · insurer {meta.get('assureur', '')} "
                f"· corpus idx {h.corpus_index}"
            )
            extra = f"BM25 `{h.bm25_score:.3f}` · cos MiniLM `{h.biencoder_score:.3f}`"
            if h.ollama_score is not None:
                extra += f" · Ollama `{h.ollama_score:.1f}`"
            st.caption(extra)
            st.text((h.text or "")[:900] + ("…" if len(h.text or "") > 900 else ""))


def main():
    st.title("Insurance Reviews — NLP")
    tab_pipe, tab_rag = st.tabs(["Full pipeline", "RAG / QA"])
    with tab_pipe:
        render_full_pipeline_tab()
    with tab_rag:
        render_rag_tab()


if __name__ == "__main__":
    import streamlit.runtime as st_runtime
    from streamlit.web import bootstrap

    if not st_runtime.exists():
        bootstrap.run(str(Path(__file__).resolve()), False, [], {})
    else:
        main()
