Using RAG in the Streamlit Application

  Code for the app (RAG, preprocessing, DistilBERT inference, similar-reviews pipeline) is under **`streamlit_app/`**.

  1. Install Ollama: https://ollama.com
  2. **Keep Ollama running** (when launched, the Ollama app typically starts the server in the background).
     - Windows / macOS: open the Ollama application (look for its icon in the taskbar).
     - Linux: if needed, run in a terminal:  ollama serve
  3. Download the model once:
        ollama pull llama3.2
  4. Start Streamlit (from the project root):
        python -m streamlit run streamlit_app/app_streamlit.py
  5. In the RAG tab/page: enter a question and click “Retrieve & generate answer”.

If Ollama isn’t running, answer generation will fail (connection error to the local API,
usually http://127.0.0.1:11434).

Building the vector index (embeddings)
--------------------------------------
You can build the index **without** Ollama, for example:

        python scripts/build_rag_index.py --max-documents 4000

or by using the “Build embedding index” button in the app. Only the LLM call to generate
an answer requires Ollama.

Example reviews (French, invented)
----------------------------------
The dataset and preprocessing expect **French** insurance-style reviews. You can paste these
into the app to try the analysis / similar-review pipeline:

**Negative (claim delays):**  
J’ai déclaré un accident auto il y a trois mois. Chaque fois que j’appelle, on me dit que le dossier est « en cours ». Les pièces ont été envoyées deux fois et personne ne me rappelle. Je paie encore ma franchise alors que je ne suis pas en tort. Très déçu, je ne recommande pas cette compagnie.

**Positive (subscription & service):**  
Souscription en ligne très claire, tarif correct pour mon habitation. J’ai eu un petit dégât des eaux : l’expert est passé sous une semaine et le virement a été fait rapidement. Équipe téléphonique aimable. Je reste chez eux.

**Mixed (price vs exclusions):**  
Prix attractif au départ, mais au moment du sinistre on m’a expliqué que la garantie ne couvrait pas ce que je pensais lire dans le contrat. Relis bien les exclusions. Sinon le site est simple à utiliser.
