from fastapi import APIRouter, Depends
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction, KernelArguments
from pydantic import BaseModel
from app.dependencies import get_kernel_full
from semantic_kernel_adapter import basic, service

router = APIRouter(prefix="/sk_basic", tags=["semantic_kernel_basic"])


class ChatRequest(BaseModel):
    input_text: str = "travel to dinosaur age"


@router.post("/joke_generator")
async def sk_local_plugin(request: ChatRequest, kernel=Depends(get_kernel_full)):
    """Add KernelPlugin from directory"""

    input_text = request.input_text
    result = await basic.local_plugin.generate_joke(kernel, input_text, "silly")
    return result


class BookRecommendationRequest(BaseModel):
    input_text: str = "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"


@router.post("/book_recommendation")
async def book_recommendation(
    request: BookRecommendationRequest, kernel=Depends(get_kernel_full)
) -> str:
    """use `ChatHistorySummarizationReducer` to reduce the chat history"""

    input_text = request.input_text
    chat_history = await basic.history_summarization_reducer.chat_history_with_summary(
        kernel, [input_text]
    )
    summary = basic.history_summarization_reducer.get_chat_history_summary(chat_history)
    return summary


@router.post("/call_kernel_function")
async def call_kernel_function(kernel=Depends(get_kernel_full)) -> list[str]:
    """use three ways to call kernel function"""

    results = await basic.function.three_ways_to_call_kernel_function(kernel)
    return results
