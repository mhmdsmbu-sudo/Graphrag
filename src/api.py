from fastapi import FastAPI
from pydantic import BaseModel
from src.graphrag import ask
from src.embeddings import semantic_search_bruteforce, semantic_search_vector_index
from src.db import run_query

app = FastAPI(
    title="VentureLens GraphRAG API",
    description="Knowledge Graph & GraphRAG API for Tech Startups",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    use_vector_index: bool = True


@app.get("/health")
def health_check():
    status = run_query("RETURN 'ok' AS status")[0]["status"]
    return {"status": status, "database": "Neo4j"}


@app.get("/stats")
def graph_stats():
    nodes = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label")
    rels = run_query("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type")
    return {"nodes": nodes, "relationships": rels}


@app.get("/search")
def search_nodes(q: str, top_k: int = 5, use_vector_index: bool = True):
    search_fn = semantic_search_vector_index if use_vector_index else semantic_search_bruteforce
    return {"query": q, "results": search_fn(q, top_k=top_k)}


@app.post("/ask")
def ask_graphrag(req: QuestionRequest):
    return ask(req.question, top_k=req.top_k, use_vector_index=req.use_vector_index)
