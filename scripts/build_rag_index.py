"""
Build and persist the vector index for the Streamlit RAG tab.

Example:
  python scripts/build_rag_index.py --max-documents 5000
"""
from __future__ import annotations

import argparse
import os
import sys

# Project root + streamlit_app on sys.path when run as `python scripts/build_rag_index.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMLIT_APP = os.path.join(ROOT, "streamlit_app")
for p in (STREAMLIT_APP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from rag_pipeline import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    build_and_save_index,
    config_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG embedding index from insurance reviews CSV.")
    parser.add_argument(
        "--csv",
        default=os.path.join(ROOT, "data", "insurance_reviews_cleaned.csv"),
        help="Path to cleaned reviews CSV.",
    )
    parser.add_argument(
        "--artifacts",
        default=os.path.join(ROOT, "artifacts"),
        help="Directory containing other artifacts; index is written to artifacts/rag/.",
    )
    parser.add_argument("--max-documents", type=int, default=4000, help="Max rows to index from the top of the CSV.")
    parser.add_argument("--chunk-max-words", type=int, default=90, help="Words per chunk window.")
    parser.add_argument("--overlap-words", type=int, default=20, help="Overlap between consecutive windows.")
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="sentence-transformers model id.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")

    fp = config_fingerprint(args.csv, args.max_documents, args.chunk_max_words, args.overlap_words)
    print(f"Build fingerprint: {fp}")
    print("Encoding (first run downloads embedding weights)...")
    emb, chunks, _meta = build_and_save_index(
        csv_path=args.csv,
        artifacts_dir=args.artifacts,
        max_documents=args.max_documents,
        chunk_max_words=args.chunk_max_words,
        overlap_words=args.overlap_words,
        embed_model=args.embed_model,
        show_progress=True,
    )
    print(f"Done. Vectors: {emb.shape[0]} x {emb.shape[1]}, chunks stored alongside metadata.")


if __name__ == "__main__":
    main()
