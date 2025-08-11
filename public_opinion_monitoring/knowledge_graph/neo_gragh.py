import json
import asyncio
import aiofiles
import os
import re

from collections import defaultdict, Counter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple, Optional, Any, Literal
from neo4j import AsyncGraphDatabase

from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO_URL")
AUTH = (os.getenv("NEO_NAME"), os.getenv("NEO_PASSWORD"))


class EventTriple(BaseModel):
    """事件三元组模型"""

    subject: str = Field(..., description="事件的施事或主体。")
    predicate: str = Field(..., description="事件的谓词或动作。")
    object: str = Field(..., description="事件的受事或客体。")
    description: str = Field(..., description="结合主谓客体，用一句话概括事件。")


class AnalysisResult(BaseModel):
    """最终分析结果模型"""

    event_triples: List[EventTriple]


class LLMResponse(BaseModel):
    """LLM输出的顶层模型"""

    analysis_result: Optional[AnalysisResult] = None


def create_event_triple(session, subject, predicate, obj, description):
    """
    Safely import an event triple into the database (without duplication).
    """
    # Use the MERGE statement to ensure idempotence (no duplication if executed repeatedly)
    # MERGE (food:Food {name: $subject}) finds or creates a food node
    # MERGE (hazard:Hazard {name: $obj}) finds or creates a hazard node
    # MERGE (food)-[rel:DETECTED]->(hazard) finds or creates a relationship
    # ON CREATE SET sets properties only when a relationship is first created
    query = """
    MERGE (food:Food {name: $subject})
    MERGE (hazard:Hazard {name: $obj})
    MERGE (food)-[rel:DETECTED]->(hazard)
    ON CREATE SET rel.description = $description, rel.predicate = $predicate
    RETURN food, hazard, rel
    """

    # Add an additional attribute to handle the "Michelin one-star restaurant" information
    # In this example, we use it as the source attribute of the Food node
    # MATCH (food:Food {name: $subject}) SET food.source = 'Michelin one-star restaurant'

    result = session.run(
        query, subject=subject, obj=obj, predicate=predicate, description=description
    )

    # Return a summary of the execution results to see how many nodes and relationships were created
    return result.consume()


async def create_event_triple_with_execute_query(
    driver, subject, obj, predicate, description
):
    query = """
    MERGE (food:Food {name: $subject})
    MERGE (hazard:Hazard {name: $obj})
    MERGE (food)-[rel:DETECTED]->(hazard)
    ON CREATE SET rel.description = $description, rel.predicate = $predicate
    """
    records, summary, keys = await driver.execute_query(
        query,
        subject=subject,
        obj=obj,
        predicate=predicate,
        description=description,
        database_="neo4j",
    )

    return summary


async def generate_knowledge_graph_from_jsonl(
    input_path: str,
    count_mode: bool = False,
):
    """
    Read JSONL files to generate knowledge graphs.
    """
    async with AsyncGraphDatabase.driver(URI, auth=AUTH) as driver:
        async with aiofiles.open(input_path, "r", encoding="utf-8") as fr:
            await driver.verify_connectivity()

            async for line in fr:
                data = json.loads(line)
                llm_response = LLMResponse.model_validate_json(line)
                if (
                    not llm_response
                    or not llm_response.analysis_result
                    or not llm_response.analysis_result.event_triples
                ):
                    continue
                print(llm_response.analysis_result.event_triples)
                for event in llm_response.analysis_result.event_triples:
                    summary = await create_event_triple_with_execute_query(
                        driver,
                        subject=event.subject,
                        predicate=event.predicate,
                        obj=event.object,
                        description=event.description,
                    )
                    print(f"Nodes created: {summary.counters.nodes_created}")
                    print(
                        f"Relationships created: {summary.counters.relationships_created}"
                    )


# Query mode
QueryMode = Literal[
    "list",  # List mode (originally by_hazard / all)
    "detection_count",  # Number of hazard detections (originally search_detection_count)
    "number_of_hazards",  # Number of hazards associated with each food item (originally search_number_of_hazards)
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


async def search_knowledge_graph(
    mode: QueryMode = "list",
    node: Optional[str] = None,  # For example, "hazard" or "h" or "food"
    prop: Optional[str] = None,  # For example, "name", "id", "description"
    value: Optional[str] = None,  # Filter value
    fuzzy: bool = False,  # Whether to perform fuzzy matching (only valid in list mode)
):
    """
    - mode="list": Returns (Food, Hazard, Relation.description), supports optional filtering
    - mode="detection_count": Returns (Hazard, DetectionCount)
    - mode="number_of_hazards": Returns (Food, DetectedHazards, NumberOfHazards)
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
    async with AsyncGraphDatabase.driver(URI, auth=AUTH) as driver:
        await driver.verify_connectivity()
        records, summary, keys = await driver.execute_query(
            query,
            **params,
            routing_="r",
            database_="neo4j",
        )

    return {
        "records": records,
        "keys": keys,
        "query": summary.query,
        "records_count": len(records),
        "time": summary.result_available_after,
    }


async def search_demo():
    result = await search_knowledge_graph(mode="list")
    print(result)
    result = await search_knowledge_graph(
        mode="list", node="hazard", prop="name", value="硼砂成分"
    )
    print(result)
    result = await search_knowledge_graph(
        mode="list", node="hazard", prop="name", value="*硼砂*"
    )
    print(result)
    result = await search_knowledge_graph(
        mode="list", node="food", prop="name", value="燕皮扁食", fuzzy=True
    )
    print(result)
    result = await search_knowledge_graph(mode="detection_count")
    print(result)
    result = await search_knowledge_graph(mode="number_of_hazards")
    print(result)


if __name__ == "__main__":
    input_jsonl_file = "3.1.weibo_data_analyzed_structured.jsonl"
    asyncio.run(generate_knowledge_graph_from_jsonl(input_jsonl_file))
    # asyncio.run(search_demo())
    print(f"Processing completed")
