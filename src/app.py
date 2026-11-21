import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.graphrag import ask
from src.db import run_query

st.set_page_config(page_title="VentureLens GraphRAG", page_icon="🔍", layout="wide")
st.title("🔍 VentureLens — Startup Knowledge Graph & GraphRAG")

# الشريط الجانبي للإعدادات وإحصائيات الجراف
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K Retrieved Nodes", min_value=1, max_value=10, value=5)
    use_vector_index = st.toggle("Use Neo4j Vector Index (HNSW)", value=True)

    st.divider()
    st.header("📊 Graph Stats")
    if st.button("Refresh Stats"):
        nodes = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label")
        rels = run_query("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type")
        st.write("**Nodes:**", {n["label"]: n["count"] for n in nodes})
        st.write("**Relationships:**", {r["type"]: r["count"] for r in rels})

# واجهة طرح الأسئلة
question = st.text_input(
    "Ask a question about startups, founders, investors, or acquisitions:",
    placeholder="e.g., Which investors backed NovaPay and what awards did NovaPay win?",
)

if st.button("Ask GraphRAG", type="primary") and question.strip():
    with st.spinner("Searching Knowledge Graph & generating answer..."):
        res = ask(question, top_k=top_k, use_vector_index=use_vector_index)

    st.subheader("💡 Answer")
    st.write(res["answer"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Retrieved Nodes")
        st.dataframe(res["retrieved_nodes"], use_container_width=True)
    with col2:
        st.subheader(f"🕸️ Subgraph Facts ({res['facts_count']} triples)")
        st.text_area("Context sent to Gemini", res["context"], height=250)