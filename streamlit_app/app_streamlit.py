"""
Streamlit : prédiction de thématique (DistilBERT fine-tuné), RAG / QA (Ollama),
et pipeline complet (thème + résumé + sentiment + avis similaires BM25 → MiniLM → Ollama).

Prérequis Ollama : `ollama pull llama3.2` (RAG et rerank des avis similaires).

Lancement (depuis la racine du dépôt) :
    streamlit run streamlit_app/app_streamlit.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import rag_pipeline as rag
import review_analysis_pipeline as review_pipe
import review_preprocess
import thematic_distilbert as thematic
from similar_reviews_pipeline import BI_ENCODER_MODEL, SimilarReviewsIndex

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(_PROJECT_ROOT / "data")
ARTIFACTS_DIR = str(_PROJECT_ROOT / "artifacts")

_ALL_INSURERS = "— Tous les assureurs —"


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
    """Sous-corpus + bi-encodeur all-MiniLM-L6-v2 (partagé BM25 / requêtes)."""
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


def render_thematic_tab() -> None:
    st.subheader("Prédiction de la thématique")
    st.caption(
        "Modèle **DistilBERT multilingue** fine-tuné (notebook §7.2). "
        "Ton texte est normalisé comme **`avis_traite`** (notebook : minuscules, sans ponctuation, "
        "stopwords FR, lemmes simplemma) — sans orthographe / traduction / résumé."
    )
    if not thematic.is_thematic_model_ready(ARTIFACTS_DIR):
        st.warning(
            "Aucun modèle dans `artifacts/distilbert_thematic/`. "
            "Exécute la cellule **7.2** du notebook jusqu’à la fin (sauvegarde automatique), puis relance l’app."
        )
        return

    m_model = thematic.bundle_mtime(ARTIFACTS_DIR)
    try:
        bundle = load_thematic_bundle_cached(ARTIFACTS_DIR, m_model)
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        return

    text = st.text_area("Avis client (français)", height=160, key="thematic_input")
    if st.button("Prédire la thématique", key="thematic_predict_btn"):
        raw = (text or "").strip()
        if not raw:
            st.warning("Saisis un texte d’avis.")
        else:
            processed = review_preprocess.preprocess_like_avis_traite(raw, ARTIFACTS_DIR)
            with st.expander("Texte après prétraitement (équivalent `avis_traite`)", expanded=False):
                st.text(processed if processed else "(vide)")
            if not processed:
                st.warning(
                    "Après prétraitement, le texte est vide (avis trop court, ou uniquement des "
                    "mots vides / chiffres). Élargis la formulation."
                )
            else:
                with st.spinner("Inférence DistilBERT…"):
                    probs = thematic.predict_thematic_proba(bundle, processed)
                if not probs:
                    st.error("Aucune prédiction.")
                else:
                    best_label, best_p = probs[0]
                    st.markdown(f"### Thématique prédite : **{best_label}**")
                    st.progress(min(1.0, max(0.0, best_p)))
                    st.caption(f"Confiance (probabilité max) : **{best_p:.1%}**")
                    st.markdown("**Probabilités par classe** (explication)")
                    for lab, p in probs:
                        st.write(f"- `{lab}` : {p:.2%}")


def render_rag_tab() -> None:
    st.subheader("RAG / QA sur les avis")
    st.caption("Recherche sémantique sur les avis, puis **Ollama** (`llama3.2`) génère la réponse en français.")
    st.info(
        "**Pour générer une réponse :** serveur **Ollama** actif ; une fois : `ollama pull llama3.2`. "
        "L’index vectoriel se construit sans Ollama."
    )

    csv_path = os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv")
    mtime = _rag_embeddings_mtime(ARTIFACTS_DIR)
    bundle = load_rag_vector_bundle(ARTIFACTS_DIR, mtime)

    if bundle is None:
        st.info("Pas d’index vectoriel. Lance `python scripts/build_rag_index.py` ou construis ci-dessous.")
        max_docs = st.number_input(
            "Nombre max de lignes à indexer (depuis le début du CSV)",
            min_value=100,
            max_value=50000,
            value=1200,
            step=100,
            key="rag_max_docs",
        )
        if st.button("Construire l’index d’embeddings", key="rag_build"):
            if not os.path.isfile(csv_path):
                st.error(f"CSV introuvable : {csv_path}")
            else:
                with st.spinner("Indexation…"):
                    rag.build_and_save_index(
                        csv_path=csv_path,
                        artifacts_dir=ARTIFACTS_DIR,
                        max_documents=int(max_docs),
                        show_progress=False,
                    )
                st.success("Index enregistré. Rechargement…")
                st.rerun()
        return

    embeddings, chunks, meta, cfg = bundle
    st.success(f"Index RAG prêt — **{cfg.get('num_vectors', embeddings.shape[0])}** segments.")

    insurer_list = insurer_choices_for_index(mtime, ARTIFACTS_DIR)
    if not insurer_list:
        st.warning("Aucun nom d’assureur dans les métadonnées de l’index.")
    insurer_choice = st.selectbox(
        "Assureur (filtrer la recherche)",
        options=[_ALL_INSURERS] + insurer_list,
        index=0,
        key="rag_insurer",
    )
    insurer_filter = None if insurer_choice == _ALL_INSURERS else insurer_choice

    question = st.text_input("Pose une question sur les avis :", key="rag_question")
    top_k = st.slider("Segments à récupérer", min_value=3, max_value=12, value=5, key="rag_topk")

    if st.button("Récupérer et générer la réponse", key="rag_answer_btn"):
        if not question.strip():
            st.warning("Entre une question.")
        else:
            embedder = load_rag_embedder()
            with st.spinner("Récupération + appel Ollama…"):
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
            st.markdown("### Réponse générée")
            st.write(answer)
            with st.expander("Extraits récupérés (scores)"):
                for item in retrieved:
                    m = item["meta"]
                    st.markdown(
                        f"**score** `{item['score']:.3f}` — **note** {m.get('note', '')} — "
                        f"**assureur** {m.get('assureur', '')}"
                    )
                    st.text(item["text"][:1200] + ("..." if len(item["text"]) > 1200 else ""))


def render_full_pipeline_tab() -> None:
    st.subheader("Pipeline : thème, résumé, sentiment, avis similaires")
    st.caption(
        "**Similarité** : BM25 → top 50, bi-encodeur **all-MiniLM-L6-v2** → top 25, "
        "top 10 → **Ollama llama3.2** (scores 0–10) → **top 5**. Sous-ensemble du CSV."
    )
    csv_path = os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv")
    if not os.path.isfile(csv_path):
        st.error(f"Fichier introuvable : `{csv_path}`")
        return

    m_csv = os.path.getmtime(csv_path)
    max_rows = st.slider(
        "Nombre de lignes du CSV (depuis le début)",
        min_value=400,
        max_value=12000,
        value=4000,
        step=200,
        key="pipe_max_rows",
    )
    use_ollama_sim = st.checkbox(
        "Reranking final avec Ollama (10 appels / analyse)",
        value=True,
        key="pipe_ollama_sim",
    )
    if use_ollama_sim:
        st.caption("Sans Ollama actif : les 5 avis sont les 5 premiers du top 10 bi-encodeur.")

    try:
        with st.spinner("Chargement du sous-corpus + embeddings (première fois : long)…"):
            sim_index, bi_encoder = load_similarity_bundle(m_csv, csv_path, max_rows)
    except Exception as e:
        st.error(f"Impossible de construire l’index de similarité : {e}")
        return

    st.success(f"**{len(sim_index.texts)}** avis indexés (BM25 + MiniLM).")

    summarizer = load_summarizer_hf()
    sentiment_pipe = load_sentiment_hf()

    thematic_bundle = None
    if thematic.is_thematic_model_ready(ARTIFACTS_DIR):
        m_model = thematic.bundle_mtime(ARTIFACTS_DIR)
        try:
            thematic_bundle = load_thematic_bundle_cached(ARTIFACTS_DIR, m_model)
        except Exception as e:
            st.warning(f"Modèle thématique non chargé : {e}")
    else:
        st.info(
            "Thématique DistilBERT indisponible (`artifacts/distilbert_thematic/`). "
            "Le reste du pipeline fonctionne."
        )

    text = st.text_area("Avis à analyser", height=180, key="pipeline_input")
    if st.button("Lancer le pipeline", key="pipeline_run"):
        raw = (text or "").strip()
        if not raw:
            st.warning("Saisis un texte d’avis.")
            return
        with st.spinner("Thématique, résumé, sentiment, recherche d’avis similaires…"):
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

        st.markdown("### Thématique (DistilBERT sur texte prétraité)")
        if result.thematic_error:
            st.warning(result.thematic_error)
        elif result.thematic_best:
            lab, p = result.thematic_best
            st.markdown(f"**{lab}** — confiance **{p:.1%}**")
            with st.expander("Probabilités par classe"):
                for a, b in result.thematic_probs:
                    st.write(f"- `{a}` : {b:.2%}")
        else:
            st.caption("Non calculé.")

        st.markdown("### Résumé (FR)")
        st.write(result.summary_fr or "—")
        if result.summary_skipped_reason:
            st.caption(result.summary_skipped_reason)

        st.markdown("### Sentiment")
        st.write(
            f"**{result.sentiment_label}** (score modèle : {result.sentiment_score:.3f})"
        )

        st.markdown("### 5 avis les plus similaires (pipeline BM25 → MiniLM → Ollama)")
        sim = result.similar
        if sim is None or not sim.final:
            st.info("Aucun voisin trouvé.")
            return
        if sim.ollama_error:
            st.warning(f"Ollama (rerank) : {sim.ollama_error} — fallback ordre bi-encodeur.")
        elif use_ollama_sim and sim.ollama_used:
            st.caption("Classement final : scores Ollama 0–10 sur le top 10 bi-encodeur.")

        for rank, h in enumerate(sim.final, start=1):
            meta = h.meta
            st.markdown(
                f"**#{rank}** — note {meta.get('note', '')} · assureur {meta.get('assureur', '')} "
                f"· idx corpus {h.corpus_index}"
            )
            extra = f"BM25 `{h.bm25_score:.3f}` · cos MiniLM `{h.biencoder_score:.3f}`"
            if h.ollama_score is not None:
                extra += f" · Ollama `{h.ollama_score:.1f}`"
            st.caption(extra)
            st.text((h.text or "")[:900] + ("…" if len(h.text or "") > 900 else ""))

        with st.expander("Détail des étapes (debug)"):
            st.write(f"**Top {len(sim.stage_bm25)} BM25** (aperçu indices)")
            st.json(
                [
                    {
                        "corpus_index": x.corpus_index,
                        "bm25": round(x.bm25_score, 4),
                        "assureur": str(x.meta.get("assureur", ""))[:40],
                    }
                    for x in sim.stage_bm25[:12]
                ]
            )
            st.write(f"**Top {len(sim.stage_biencoder)} bi-encodeur**")
            st.json(
                [
                    {
                        "corpus_index": x.corpus_index,
                        "cos": round(x.biencoder_score, 4),
                    }
                    for x in sim.stage_biencoder[:12]
                ]
            )


def main():
    st.title("Insurance Reviews — NLP")
    tab_pred, tab_pipe, tab_rag = st.tabs(
        ["Prédiction thématique (DistilBERT)", "Pipeline complet", "RAG / QA"]
    )
    with tab_pred:
        render_thematic_tab()
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
