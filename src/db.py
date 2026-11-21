from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_driver = None


def get_driver():
    """Return a shared Neo4j driver instance (lazy initialization)."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    """Safely close the Neo4j driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def run_query(query: str, params: dict | None = None) -> list[dict]:
    """Execute a read Cypher query and return results as a list of dicts."""
    with get_driver().session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def run_write(query: str, params: dict | None = None):
    """Execute a write Cypher query inside an explicit write transaction."""
    with get_driver().session() as session:
        return session.execute_write(lambda tx: tx.run(query, params or {}).consume())


if __name__ == "__main__":
    status = run_query("RETURN 'Neo4j Connected Successfully!' AS status")[0]["status"]
    print(status)
    close_driver()
