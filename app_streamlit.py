"""
Streamlit app: thematic prediction, automatic summary, RAG/QA.
"""
import os
import pickle
import streamlit as st
import pandas as pd
from transformers import pipeline
from gensim.models import Word2Vec

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"

def _ensure_nltk():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

@st.cache_resource
def load_artifacts():
    _ensure_nltk()
    with open(os.path.join(ARTIFACTS_DIR, "preprocess.pkl"), "rb") as f:
        prep = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "clf_thematic.pkl"), "rb") as f:
        clf = pickle.load(f)
    w2v = Word2Vec.load(os.path.join(ARTIFACTS_DIR, "w2v.model"))
    return prep, tfidf, clf, w2v

def preprocess_text(text, stopwords):
    import nltk
    import simplemma
    from string import punctuation
    if not text or not str(text).strip():
        return ""
    text = str(text).lower()
    text = text.replace("'", " ")
    text = "".join(c for c in text if c not in punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1 and not any(c.isdigit() for c in t)]
    lemmas = [simplemma.lemmatize(t, lang="fr") or t for t in tokens]
    return " ".join(lemmas)

def main():
    st.title("Insurance Reviews - NLP Analysis")
    prep, tfidf, clf, w2v = load_artifacts()
    stopwords = prep["stopwords"]

    tab1, tab2, tab3 = st.tabs(["Thematic Prediction", "Summary", "RAG / QA"])

    with tab1:
        st.subheader("Prediction of thematic")
        text_input = st.text_area("Enter a review (French):")
        if st.button("Predict thematic"):
            if text_input:
                processed = preprocess_text(text_input, stopwords)
                X = tfidf.transform([processed])
                pred = clf.predict(X)[0]
                st.write(f"**Predicted thematic:** {pred}")
            else:
                st.warning("Please enter a review.")

    with tab2:
        st.subheader("Automatic summary")
        text_input = st.text_area("Enter text to summarize:", key="summ")
        if st.button("Summarize", key="btn_summ"):
            if text_input:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summary = summarizer(str(text_input)[:1024], max_length=100, min_length=30)[0]["summary_text"]
                st.write(summary)
            else:
                st.warning("Please enter text.")

    with tab3:
        st.subheader("RAG / QA - Search in reviews")
        df = pd.read_csv(os.path.join(DATA_DIR, "insurance_reviews_cleaned.csv"), encoding="utf-8")
        query = st.text_input("Search query (keyword):")
        if query:
            query_lower = query.lower()
            matches = df[df["avis"].str.contains(query_lower, na=False, case=False)]
            if len(matches) > 0:
                st.write(f"Found {len(matches)} reviews containing '{query}'")
                for i, row in matches.head(5).iterrows():
                    st.write(f"- **Note {row['note']}**: {str(row['avis'])[:200]}...")
            else:
                st.write("No matching reviews.")

if __name__ == "__main__":
    main()
