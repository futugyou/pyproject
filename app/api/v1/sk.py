from fastapi import APIRouter, Depends
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction, KernelArguments
from pydantic import BaseModel
from app.dependencies import get_kernel
from semantic_kernel_service import v00_chat, v01_prompt, v02_argument

router = APIRouter(prefix="/sk")


class ChatRequest(BaseModel):
    input_text: str = "travel to dinosaur age"


@router.post("/base")
async def sk_base(request: ChatRequest, kernel=Depends(get_kernel)):
    input_text = request.input_text
    result = await v00_chat.generate_joke(kernel, input_text, "silly")
    return result


class PromptRequest(BaseModel):
    input_text: str


@router.post("/prompt")
async def sk_prompt(request: PromptRequest, kernel=Depends(get_kernel)):
    input_text = request.input_text
    summary = await v01_prompt.generate_summarize(kernel, input_text)
    return summary


@router.post("/argument")
async def sk_argument(kernel=Depends(get_kernel)):
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful chatbot who is good about giving book recommendations."
    )
    input_texts = [
        "Hi, I'm looking for book suggestions",
        "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?",
    ]
    await v02_argument.generate_arguments(kernel, input_texts, chat_history)
    return chat_history
