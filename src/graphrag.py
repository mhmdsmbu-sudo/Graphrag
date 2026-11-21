from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, LLM_MODEL
from src.db import run_query, close_driver
from src.embeddings import semantic_search_bruteforce, semantic_search_vector_index


def traverse_subgraph(node_names: list[str], limit: int = 120) -> list[str]:
    """Stage 2: Traverse 1-2 hops from top-k nodes and collect unique fact triples."""
    rows = run_query("""
        MATCH (n) WHERE n.name IN $names
        OPTIONAL MATCH (n)-[r1]-(m)
        OPTIONAL MATCH (m)-[r2]-(k)
        RETURN n.name AS n_name, type(r1) AS r1_type, properties(r1) AS r1_props,
               m.name AS m_name, type(r2) AS r2_type, properties(r2) AS r2_props, k.name AS k_name
        LIMIT $limit
    """, {"names": node_names, "limit": limit})

    triples = set()
    for r in rows:
        if r["r1_type"] and r["m_name"]:
            p1 = f" {r['r1_props']}" if r["r1_props"] else ""
            triples.add(f"({r['n_name']}) -[{r['r1_type']}{p1}]-> ({r['m_name']})")
        if r["r2_type"] and r["k_name"]:
            p2 = f" {r['r2_props']}" if r["r2_props"] else ""
            triples.add(f"({r['m_name']}) -[{r['r2_type']}{p2}]-> ({r['k_name']})")
    return sorted(triples)


def build_context(triples: list[str]) -> str:
    """Stage 3: Serialize subgraph triples into clean, readable context text."""
    return "\n".join(f"- {t}" for t in triples) if triples else "No facts found."


def generate_llm_answer(question: str, context: str) -> str:
    """Stage 4: Send question + context to Gemini with strict grounding instructions."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    system_prompt = (
        "You are a precise startup ecosystem analyst at VentureLens. "
        "Answer the user's question using ONLY the provided knowledge graph facts. "
        "If the facts do not contain the answer, state that clearly. Do not invent information."
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=f"Context Facts:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )
    return response.text


def ask(question: str, top_k: int = 5, use_vector_index: bool = True) -> dict:
    """Full 4-stage GraphRAG pipeline."""
    # Stage 1: Semantic retrieval
    search_fn = semantic_search_vector_index if use_vector_index else semantic_search_bruteforce
    top_nodes = search_fn(question, top_k=top_k)

    # Stage 2: 1-2 hop graph traversal
    triples = traverse_subgraph([n["name"] for n in top_nodes])

    # Stage 3: Context assembly
    context = build_context(triples)

    # Stage 4: Grounded LLM generation
    answer = generate_llm_answer(question, context)

    return {
        "question": question,
        "search_mode": "neo4j_vector_index" if use_vector_index else "bruteforce_cosine",
        "retrieved_nodes": top_nodes,
        "facts_count": len(triples),
        "context": context,
        "answer": answer,
    }


if __name__ == "__main__":
    sample_questions = [
        "Which investors backed NovaPay and what awards did NovaPay win?",
        "Which companies were founded by Marcus Chen and what happened to VoiceCraft?",
        "Who founded companies that were later acquired by CloudNest or DataForge?",
    ]
    for q in sample_questions:
        res = ask(q, top_k=5, use_vector_index=True)
        print(f"\nQ: {res['question']}")
        print(f"Retrieved Nodes: {[n['name'] for n in res['retrieved_nodes']]}")
        print(f"A: {res['answer']}\n{'-'*60}")
    close_driver()
