import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM
from src.db import run_query, run_write, close_driver

NODE_LABELS = ["Company", "Person", "Investor", "Product", "Award"]
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the local multilingual-e5-small model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def node_to_text(label: str, props: dict) -> str:
    """Convert a node's label and properties into a rich descriptive sentence."""
    name = props.get("name", "")
    if label == "Company":
        return (
            f"Company {name} operates in the {props.get('sector')} sector, "
            f"headquartered in {props.get('headquarters')}, founded in {props.get('founded')}, "
            f"valued at {props.get('valuation_billion')} billion. Description: {props.get('description')}"
        )
    if label == "Person":
        return f"Person {name} is a {props.get('role')} born in {props.get('born')} from {props.get('hometown')}."
    if label == "Investor":
        return f"Investor {name} is a {props.get('type')} firm founded in {props.get('founded')} based in {props.get('hq')}."
    if label == "Product":
        return f"Product {name} is a {props.get('category')} launched in {props.get('launch_year')}."
    if label == "Award":
        return f"Award {name} is a {props.get('category')} recognition from {props.get('year')}."
    return f"{label} {name}"


def generate_and_store_embeddings():
    """Embed every node in Neo4j and store the vector on n.embedding."""
    model = get_embedding_model()
    nodes = run_query("MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS props")

    for node in nodes:
        text = node_to_text(node["label"], node["props"])
        # E5 models require 'passage: ' prefix for indexed documents
        vec = model.encode(f"passage: {text}", normalize_embeddings=True).tolist()
        run_write(
            "MATCH (n) WHERE elementId(n) = $id SET n.embedding = $vec, n.embedding_text = $text",
            {"id": node["id"], "vec": vec, "text": text},
        )
    print(f"Stored embeddings for {len(nodes)} nodes using {EMBEDDING_MODEL_NAME}.")


def create_vector_indexes():
    """Bonus: Create native Neo4j HNSW vector indexes for all node labels."""
    for label in NODE_LABELS:
        run_write(f"""
            CREATE VECTOR INDEX {label.lower()}_embedding_idx IF NOT EXISTS
            FOR (n:{label}) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
    print("Neo4j native vector indexes verified.")


def semantic_search_bruteforce(query: str, top_k: int = 5) -> list[dict]:
    """Task 5: Brute-force cosine similarity search in Python."""
    model = get_embedding_model()
    # E5 models require 'query: ' prefix for search queries
    q_vec = model.encode(f"query: {query}", normalize_embeddings=True)

    nodes = run_query(
        "MATCH (n) WHERE n.embedding IS NOT NULL "
        "RETURN labels(n)[0] AS label, n.name AS name, n.embedding AS vec"
    )
    scored = [
        {
            "label": n["label"],
            "name": n["name"],
            "score": round(float(np.dot(q_vec, np.array(n["vec"], dtype=np.float32))), 4),
        }
        for n in nodes
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def semantic_search_vector_index(query: str, top_k: int = 5) -> list[dict]:
    """Bonus: Native Neo4j vector index search across all node labels."""
    model = get_embedding_model()
    q_vec = model.encode(f"query: {query}", normalize_embeddings=True).tolist()

    results = []
    for label in NODE_LABELS:
        rows = run_query(
            f"CALL db.index.vector.queryNodes('{label.lower()}_embedding_idx', $k, $vec) "
            "YIELD node, score RETURN labels(node)[0] AS label, node.name AS name, score",
            {"k": top_k, "vec": q_vec},
        )
        for r in rows:
            results.append({"label": r["label"], "name": r["name"], "score": round(float(r["score"]), 4)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    generate_and_store_embeddings()
    create_vector_indexes()

    test_query = "paying people across borders"
    print(f"\n--- Brute-force Concept Search: '{test_query}' ---")
    for res in semantic_search_bruteforce(test_query, top_k=5):
        print(" ", res)

    print(f"\n--- Neo4j Vector Index Search: '{test_query}' ---")
    for res in semantic_search_vector_index(test_query, top_k=5):
        print(" ", res)
    close_driver()
