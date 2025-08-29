from fastapi import APIRouter, Depends

from pydantic import BaseModel
from langchain_adapter import chat, option, tool, multimodal, agent

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
