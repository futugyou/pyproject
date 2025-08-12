import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import json
import re

from collections import defaultdict, Counter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple, Optional, Any, Literal
from py2neo import Graph, Node, Relationship
from py2neo.matching import NodeMatcher

from models.llm_response import LLMResponse

from dotenv import load_dotenv

load_dotenv()

# The URI and authentication details for the Neo4j database are loaded from environment variables
URI = os.getenv("NEO_URL")
AUTH = (os.getenv("NEO_NAME"), os.getenv("NEO_PASSWORD"))

# Pydantic models remain the same as they are independent of the database driver.
# They are used for data validation and structure.


def create_event_triple_py2neo(graph, subject, obj, predicate, description):
    """
    Safely import an event triple into the database using py2neo.
    This function is a synchronous equivalent of the original.
    """
    query = """
    MERGE (food:Food {name: $subject})
    MERGE (hazard:Hazard {name: $obj})
    MERGE (food)-[rel:DETECTED]->(hazard)
    ON CREATE SET rel.description = $description, rel.predicate = $predicate
    """

    # Use graph.run() to execute the Cypher query.
    # It returns a py2neo.Cursor object, which provides access to results and execution statistics.
    result = graph.run(
        query, subject=subject, obj=obj, predicate=predicate, description=description
    )

    # Return the summary of execution, similar to the original function.
    # The stats() method on the cursor provides information like nodes and relationships created.
    return result.stats()


def generate_knowledge_graph_from_jsonl_py2neo(
    input_path: str,
):
    """
    Read JSONL files to generate knowledge graphs using py2neo.
    This is a synchronous implementation.
    """
    # Establish a synchronous connection to the database using py2neo.Graph.
    graph = Graph(URI, auth=AUTH)

    # Open the file and process it line by line.
    with open(input_path, "r", encoding="utf-8") as fr:
        for line in fr:
            # Data validation using Pydantic remains the same.
            llm_response = LLMResponse.model_validate_json(line)
            if (
                not llm_response
                or not llm_response.analysis_result
                or not llm_response.analysis_result.event_triples
            ):
                continue

            print(llm_response.analysis_result.event_triples)
            for event in llm_response.analysis_result.event_triples:
                # Call the new synchronous creation function.
                stats = create_event_triple_py2neo(
                    graph,
                    subject=event.subject,
                    predicate=event.predicate,
                    obj=event.object,
                    description=event.description,
                )
                nodes_created = stats.get("nodes_created", 0)
                relationships_created = stats.get("relationships_created", 0)

                print(f"Nodes created: {nodes_created}")
                print(f"Relationships created: {relationships_created}")


# Query mode and helper maps/regex remain the same.
QueryMode = Literal[
    "list",
    "detection_count",
    "number_of_hazards",
]

_NODE_ALIAS_MAP = {
    "food": "f",
    "f": "f",
    "hazard": "h",
    "h": "h",
    "relation": "r",
    "rel": "r",
    "r": "r",
}

_PROP_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def search_knowledge_graph_py2neo(
    mode: QueryMode = "list",
    node: Optional[str] = None,
    prop: Optional[str] = None,
    value: Optional[str] = None,
    fuzzy: bool = False,
):
    """
    A synchronous implementation of the search function using py2neo.
    """
    cypher_map = {
        "list": "MATCH (f:Food)-[r:DETECTED]->(h:Hazard)\n",
        "detection_count": (
            "MATCH (f:Food)-[r:DETECTED]->(h:Hazard)\n"
            "RETURN h.name AS Hazard, count(f) AS DetectionCount\n"
            "ORDER BY DetectionCount DESC\n"
        ),
        "number_of_hazards": (
            "MATCH (f:Food)-[r:DETECTED]->(h:Hazard)\n"
            "RETURN f.name AS Food, collect(h.name) AS DetectedHazards, count(r) AS NumberOfHazards\n"
        ),
    }

    if mode not in cypher_map:
        raise ValueError(f"unsupported mode: {mode}")

    query = cypher_map[mode]
    params: Dict[str, str] = {}

    if mode == "list":
        if (node is None) ^ (prop is None):
            raise ValueError(
                "If you want to filter, please pass in both node and prop, for example, node='hazard', prop='name'"
            )

        if node and prop:
            alias_key = node.lower()
            if alias_key not in _NODE_ALIAS_MAP:
                raise ValueError(
                    f"unsupported node: {node}. allowed: {list(_NODE_ALIAS_MAP.keys())}"
                )

            alias = _NODE_ALIAS_MAP[alias_key]

            if not _PROP_RE.match(prop):
                raise ValueError(f"invalid property name: {prop}")

            if fuzzy or ("*" in (value or "") or "%" in (value or "")):
                v = (value or "").replace("*", "").replace("%", "")
                query += f" WHERE toLower({alias}.{prop}) CONTAINS toLower($value)"
                params["value"] = v
            else:
                query += f" WHERE {alias}.{prop} = $value"
                params["value"] = value

        query += (
            " RETURN f.name AS Food, h.name AS Hazard, r.description AS Description"
        )

    print(query)

    graph = Graph(URI, auth=AUTH)

    # Execute the query using py2neo.Graph.run().
    result = graph.run(query, **params)

    # Process the result to match the original function's output format.
    records = list(result)
    keys = result.keys()

    return {
        "records": records,
        "keys": keys,
        "query": query,
        "records_count": len(records),
        # py2neo does not provide a direct equivalent for 'result_available_after'.
        # We can return None or a placeholder.
        "time": None,
    }


def search_demo_py2neo():
    """A synchronous demo function for searching."""
    result = search_knowledge_graph_py2neo(mode="list")
    print(result)
    result = search_knowledge_graph_py2neo(
        mode="list", node="hazard", prop="name", value="硼砂成分"
    )
    print(result)
    result = search_knowledge_graph_py2neo(
        mode="list", node="hazard", prop="name", value="*硼砂*"
    )
    print(result)
    result = search_knowledge_graph_py2neo(
        mode="list", node="food", prop="name", value="燕皮扁食", fuzzy=True
    )
    print(result)
    result = search_knowledge_graph_py2neo(mode="detection_count")
    print(result)
    result = search_knowledge_graph_py2neo(mode="number_of_hazards")
    print(result)


if __name__ == "__main__":
    input_jsonl_file = "3.1.weibo_data_analyzed_structured.jsonl"

    # Use the synchronous py2neo implementation.
    generate_knowledge_graph_from_jsonl_py2neo(input_jsonl_file)
    search_demo_py2neo()

    print(f"Processing completed")
