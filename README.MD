================================================================================
Utilisation du RAG dans l’application Streamlit
================================================================================

Faut-il lancer Ollama avant Streamlit ?
-----------------------------------------
Oui, pour **poser une question et obtenir une réponse générée** :

  1. Installe Ollama : https://ollama.com
  2. **Laisse Ollama actif** (au démarrage, l’application Ollama lance en général le serveur en arrière-plan).
     - Windows / macOS : ouvre l’application Ollama (icône dans la barre des tâches).
     - Linux : si besoin, dans un terminal :  ollama serve
  3. Télécharge le modèle une fois :
        ollama pull llama3.2
  4. Lance Streamlit (depuis la racine du projet) :
        python -m streamlit run app_streamlit.py
  5. Dans l’onglet / page RAG : saisis une question puis « Retrieve & generate answer ».

Sans Ollama qui tourne, la génération échouera (erreur de connexion à l’API locale,
souvent http://127.0.0.1:11434).

Construction de l’index vectoriel (embeddings)
-----------------------------------------------
Tu peux construire l’index **sans** Ollama, par exemple :

        python scripts/build_rag_index.py --max-documents 4000

ou le bouton « Build embedding index » dans l’app. Seul l’appel au LLM pour rédiger
la réponse nécessite Ollama.
