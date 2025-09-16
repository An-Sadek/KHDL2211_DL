from neo4j import GraphDatabase
import itertools

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "12345678")

# Initialize the Neo4j driver
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

    # Read the file and collect edges
    edges = []
    with open("data/facebook/facebook_combined.txt", "r") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                node1, node2 = map(int, line.strip().split())
                edges.append({"node1": node1, "node2": node2})
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
                continue

    # Process edges in batches
    batch_size = 1000
    with driver.session(database="fbegonets") as session:
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            query = """
            UNWIND $batch AS edge
            MERGE (a:Person {id: edge.node1})
            MERGE (b:Person {id: edge.node2})
            MERGE (a)-[:KNOWS]->(b)
            """
            result = session.run(query, batch=batch)
            summary = result.consume()  # Use .consume() to get the summary
            print(f"Batch {i//batch_size + 1}: Created {summary.counters.nodes_created} nodes and "
                  f"{summary.counters.relationships_created} relationships in "
                  f"{summary.result_available_after} ms.")