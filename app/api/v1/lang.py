from fastapi import APIRouter, Depends

from pydantic import BaseModel
from langchain_adapter import chat, option, tool

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
