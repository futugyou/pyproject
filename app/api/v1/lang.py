from fastapi import APIRouter, Depends

from pydantic import BaseModel
from langchain_adapter import chat, option

router = APIRouter(prefix="/langchain", tags=["lang_chain"])


class ChatRequest(BaseModel):
    input_text: str = "Tell me a joke about cats"


@router.post("/joke_generator")
async def sk_local_plugin(request: ChatRequest):
    input_text = request.input_text
    config = option.LangChainOption()
    result = chat.generate_joke(input_text, config)
    return result
