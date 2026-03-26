Using RAG in the Streamlit Application

  Code for the app (RAG, preprocessing, DistilBERT inference, similar-reviews pipeline) is under **`streamlit_app/`**.

  1. `pip install -r requirements.txt`
  2. Install Ollama: https://ollama.com
  3. **Keep Ollama running** in backgroud.
     - Windows / macOS: open the Ollama application.
     - Linux: if needed, run in a terminal:  ollama serve
  4. Download the model once:
        `ollama pull llama3.2`
  5. Start Streamlit (from the project root):
        `python -m streamlit run streamlit_app/app_streamlit.py`
  6. In the different tabs: enter a question or a review (see below to copy/paste) and click to generate.

If Ollama isn’t running, answer generation will fail (connection error to the local API,
usually http://127.0.0.1:11434).



Example questions (RAG / QA tab)
--------------------------------
The corpus is French insurance-style reviews. After the embedding index is ready, you can
test retrieval + generation with questions such as:

**Delays & claims:**  
- Que disent les clients sur les délais de remboursement ou de traitement des dossiers ?  
- Quels avis mentionnent des difficultés après un sinistre auto ou habitation ?

**Service & satisfaction:**  
- Comment les assurés décrivent le service client ou le contact téléphonique ?  
- Qu’est-ce qui ressort sur la souscription en ligne ou la clarté des contrats ?

**Price & guarantees:**  
- Les avis parlent-ils du rapport qualité-prix ou des tarifs ?  
- Y a-t-il des retours sur les exclusions de garantie ou la lecture du contrat ?

You can also narrow retrieval with the **Insurer** filter if several companies appear in the index.



Building the vector index (embeddings)
--------------------------------------
You can build the index **without** Ollama, for example:

`python scripts/build_rag_index.py --max-documents 4000`

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
