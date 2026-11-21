from src.db import run_query, close_driver

QUERIES = {
    "Q1_Fintech_Above_5B": """
        MATCH (c:Company)
        WHERE c.sector = 'Fintech' AND c.valuation_billion > 5
        RETURN c.name AS company, c.valuation_billion AS valuation_billion
        ORDER BY c.valuation_billion DESC
    """,
    "Q2_Serial_Founders": """
        MATCH (p:Person)-[:FOUNDED]->(c:Company)
        WITH p, collect(c.name) AS companies, count(c) AS company_count
        WHERE company_count > 1
        RETURN p.name AS founder, company_count, companies
        ORDER BY company_count DESC
    """,
    "Q3_Investor_Portfolio": """
        MATCH (i:Investor {name: $investor_name})-[r:INVESTED_IN]->(c:Company)
        RETURN i.name AS investor,
               collect(DISTINCT c.name) AS funded_companies,
               round(sum(r.amount_million), 2) AS total_capital_million
    """,
    "Q4_Awarded_And_Acquired": """
        MATCH (acquirer:Company)-[:ACQUIRED]->(target:Company)-[:WON]->(a:Award)
        RETURN target.name AS company, a.name AS award, acquirer.name AS acquirer
    """,
    "Q5_Investor_Backed_Acquirer_And_Target": """
        MATCH (i:Investor)-[:INVESTED_IN]->(acquirer:Company)-[:ACQUIRED]->(target:Company)<-[:INVESTED_IN]-(i)
        RETURN DISTINCT i.name AS investor, acquirer.name AS acquirer, target.name AS target
    """,
    "Q6_Shortest_Path_Founders": """
        MATCH path = shortestPath((p1:Person {name: $p1})-[*]-(p2:Person {name: $p2}))
        RETURN [n IN nodes(path) | coalesce(n.name, '')] AS path_nodes,
               [r IN relationships(path) | type(r)] AS path_rels,
               length(path) AS hops
    """,
}


def run_all_queries():
    params_map = {
        "Q3_Investor_Portfolio": {"investor_name": "Summit Ventures"},
        "Q6_Shortest_Path_Founders": {"p1": "Elena Rossi", "p2": "Marcus Chen"},
    }
    for name, query in QUERIES.items():
        print(f"\n=== {name} ===")
        rows = run_query(query, params_map.get(name))
        for row in rows:
            print(" ", row)


if __name__ == "__main__":
    run_all_queries()
    close_driver()
