# VentureLens Tech Startups Knowledge Graph & GraphRAG

End-to-end Knowledge Graph & GraphRAG pipeline built with **Neo4j**, **Sentence-Transformers (`intfloat/multilingual-e5-small`)**, **Google Gemini**, and **FastAPI**.

## Setup

1. Place `tech_startups_dataset_STUDENT.xlsx` inside `data/raw/`.
2. Copy `.env.example` to `.env` and fill in your `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `GEMINI_API_KEY`.
3. Create and activate the virtual environment using `uv`:
   ```powershell
   uv venv --python 3.14
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```
4. (Optional Local Neo4j) Run `docker compose up -d` if not using Neo4j Aura Cloud.

## Run Pipeline

```powershell
python -m src.clean_data
python -m src.db
python -m src.loader
python -m src.queries
python -m src.embeddings
python -m src.graphrag
uvicorn src.api:app --reload
```
