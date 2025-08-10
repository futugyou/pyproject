import json
import asyncio
import os

from collections import defaultdict, Counter
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Tuple, Optional, Any
from neo4j import GraphDatabase

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


async def process_jsonl_file(
    input_path: str,
    count_mode: bool = False,
):
    """
    Process JSONL files according to the new logic.
    """

    with (
        open(input_path, "r", encoding="utf-8") as fr,
        GraphDatabase.driver(URI, auth=AUTH) as driver,
        driver.session() as session,
    ):
        driver.verify_connectivity()

        for line in fr:
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
                summary = create_event_triple(
                    session,
                    subject=event.subject,
                    predicate=event.predicate,
                    obj=event.object,
                    description=event.description,
                )
                print(f"Nodes created: {summary.counters.nodes_created}")
                print(
                    f"Relationships created: {summary.counters.relationships_created}"
                )


if __name__ == "__main__":
    input_jsonl_file = "3.1.weibo_data_analyzed_structured.jsonl"
    asyncio.run(process_jsonl_file(input_jsonl_file))
    print(f"Processing completed")
