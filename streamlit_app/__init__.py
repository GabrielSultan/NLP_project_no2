"""
Streamlit application package for insurance-review NLP.

Modules:
  - app_streamlit: UI entrypoint (tabs, caching, wiring).
  - review_preprocess: French text normalization aligned with the notebook.
  - thematic_distilbert: Fine-tuned theme classifier inference.
  - review_analysis_pipeline: Single-review orchestration (theme, summary, sentiment, similar).
  - similar_reviews_pipeline: BM25 + sentence embeddings + optional Ollama rerank.
  - rag_pipeline: Chunking, embedding index, retrieval, Ollama answer generation.
"""
