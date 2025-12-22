from fastapi import APIRouter, Depends

from pydantic import BaseModel
from langchain_adapter import (
    chat,
    option,
    tool,
    multimodal,
    agent,
    embedding,
    compression,
)

router = APIRouter(prefix="/langchain", tags=["lang_chain"])


class ChatRequest(BaseModel):
    input_text: str = "Tell me a joke about cats"


@router.post("/joke_generator")
async def joke_generator(request: ChatRequest):
    input_text = request.input_text
    config = option.LangChainOption()
    result = chat.generate_joke(input_text, config)
    return result


class CalculateRequest(BaseModel):
    query: str = "What is 3 * 12? Also, what is 11 + 49?"


@router.post("/calculate")
async def calculate(request: CalculateRequest):
    query = request.query
    config = option.LangChainOption()
    result = tool.calculate(query, config)
    return result


@router.post("/calculate2")
async def calculate2(request: CalculateRequest):
    """use  langchain agent or langgraph agent"""

    query = request.query
    config = option.LangChainOption()
    result = agent.calculate2(query, config)
    return result


@router.post("/describe_image")
async def describe_image(request: multimodal.MultimodalData):
    config = option.LangChainOption()
    result = multimodal.multimodal(request, config)
    return result


@router.post("/retriever")
async def retriever(question: str = "what is `Structure`?"):
    """Get information from local files"""

    config = option.LangChainOption()
    result = embedding.retriever(question, "./README.md", config)
    return result


@router.post("/multi_query_retriever")
async def multi_query_retriever(
    question: str = "What is ai-concepts?",
    path: str = "https://learn.microsoft.com/zh-cn/azure/architecture/ai-ml/",
):
    """Get information from network files"""

    config = option.LangChainOption()
    vectordb = embedding.vectordb_with_Chroma(path, config)
    doc = embedding.multi_query_retriever_with_output(question, vectordb, config)
    return doc


@router.post("/contextual_compression")
async def contextual_compression(
    question: str = "What did the president say about Ketanji Jackson Brown",
):
    """do retrieval with contextual compression"""

    config = option.LangChainOption()
    retriever = compression.get_retriever(config)
    doc = compression.contextual_compression(question, retriever, config)
    return doc
