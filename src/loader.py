import pandas as pd
from src.config import CLEAN_DATA_PATH
from src.db import run_query, run_write, close_driver


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts, replacing pandas NaN with Python None."""
    return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def create_constraints():
    """Create uniqueness constraints on 'name' for all 5 node labels."""
    for label in ["Person", "Company", "Investor", "Product", "Award"]:
        run_write(
            f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )
    print("Uniqueness constraints verified.")


def load_nodes(xls: pd.ExcelFile):
    """Load all 5 node sheets idempotently using UNWIND + MERGE."""
    node_queries = [
        ("Founders", """
            UNWIND $rows AS row
            MERGE (p:Person {name: row.name})
            SET p.born = row.born, p.role = row.role, p.hometown = row.hometown
        """),
        ("Companies", """
            UNWIND $rows AS row
            MERGE (c:Company {name: row.name})
            SET c.founded = row.founded, c.sector = row.sector,
                c.headquarters = row.headquarters, c.valuation_billion = row.valuation_billion,
                c.description = row.description
        """),
        ("Investors", """
            UNWIND $rows AS row
            MERGE (i:Investor {name: row.name})
            SET i.founded = row.founded, i.type = row.type, i.hq = row.hq
        """),
        ("Products", """
            UNWIND $rows AS row
            MERGE (pr:Product {name: row.name})
            SET pr.category = row.category, pr.launch_year = row.launch_year
        """),
        ("Awards", """
            UNWIND $rows AS row
            MERGE (a:Award {name: row.name})
            SET a.category = row.category, a.year = row.year
        """),
    ]
    for sheet, query in node_queries:
        run_write(query, {"rows": df_to_records(pd.read_excel(xls, sheet))})


def load_relationships(xls: pd.ExcelFile):
    """Load all 6 relationship sheets idempotently using MATCH + MERGE."""
    rel_queries = [
        ("Founded", """
            UNWIND $rows AS row
            MATCH (p:Person {name: row.founder_name})
            MATCH (c:Company {name: row.company_name})
            MERGE (p)-[r:FOUNDED]->(c)
            SET r.year = row.year
        """),
        ("WorksAt", """
            UNWIND $rows AS row
            MATCH (p:Person {name: row.person_name})
            MATCH (c:Company {name: row.company_name})
            MERGE (p)-[r:WORKS_AT {role: row.role}]->(c)
            SET r.current = row.current
        """),
        ("InvestedIn", """
            UNWIND $rows AS row
            MATCH (i:Investor {name: row.investor_name})
            MATCH (c:Company {name: row.company_name})
            MERGE (i)-[r:INVESTED_IN {round: row.round, year: row.year}]->(c)
            SET r.amount_million = row.amount_million
        """),
        ("Acquired", """
            UNWIND $rows AS row
            MATCH (acq:Company {name: row.acquirer_company})
            MATCH (tgt:Company {name: row.target_company})
            MERGE (acq)-[r:ACQUIRED]->(tgt)
            SET r.year = row.year, r.amount_million = row.amount_million
        """),
        ("Develops", """
            UNWIND $rows AS row
            MATCH (c:Company {name: row.company_name})
            MATCH (pr:Product {name: row.product_name})
            MERGE (c)-[:DEVELOPS]->(pr)
        """),
    ]
    for sheet, query in rel_queries:
        run_write(query, {"rows": df_to_records(pd.read_excel(xls, sheet))})

    # Handle polymorphic WON relationship (split by entity_type)
    won_rows = df_to_records(pd.read_excel(xls, "Won"))
    won_companies = [r for r in won_rows if r["entity_type"] == "Company"]
    won_people = [r for r in won_rows if r["entity_type"] == "Person"]

    run_write("""
        UNWIND $rows AS row
        MATCH (c:Company {name: row.entity_name})
        MATCH (a:Award {name: row.award_name})
        MERGE (c)-[:WON]->(a)
    """, {"rows": won_companies})

    run_write("""
        UNWIND $rows AS row
        MATCH (p:Person {name: row.entity_name})
        MATCH (a:Award {name: row.award_name})
        MERGE (p)-[:WON]->(a)
    """, {"rows": won_people})


def print_graph_counts():
    """Print final node and relationship counts for verification."""
    nodes = run_query("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label")
    rels = run_query("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY type")
    print("\n--- Final Node Counts ---")
    for n in nodes:
        print(f"  {n['label']:12s}: {n['count']}")
    print(f"  TOTAL NODES : {sum(n['count'] for n in nodes)}")

    print("\n--- Final Relationship Counts ---")
    for r in rels:
        print(f"  {r['type']:12s}: {r['count']}")
    print(f"  TOTAL RELS  : {sum(r['count'] for r in rels)}")


def load_all_data():
    create_constraints()
    xls = pd.ExcelFile(CLEAN_DATA_PATH)
    load_nodes(xls)
    load_relationships(xls)
    print_graph_counts()


if __name__ == "__main__":
    load_all_data()
    close_driver()
