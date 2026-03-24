Using RAG in the Streamlit Application

  1. Install Ollama: https://ollama.com
  2. **Keep Ollama running** (when launched, the Ollama app typically starts the server in the background).
     - Windows / macOS: open the Ollama application (look for its icon in the taskbar).
     - Linux: if needed, run in a terminal:  ollama serve
  3. Download the model once:
        ollama pull llama3.2
  4. Start Streamlit (from the project root):
        python -m streamlit run app_streamlit.py
  5. In the RAG tab/page: enter a question and click “Retrieve & generate answer”.

If Ollama isn’t running, answer generation will fail (connection error to the local API,
usually http://127.0.0.1:11434).

Building the vector index (embeddings)
--------------------------------------
You can build the index **without** Ollama, for example:

        python scripts/build_rag_index.py --max-documents 4000

or by using the “Build embedding index” button in the app. Only the LLM call to generate
an answer requires Ollama.
