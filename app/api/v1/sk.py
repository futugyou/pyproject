from fastapi import APIRouter, Depends
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction, KernelArguments
from pydantic import BaseModel
from app.dependencies import get_kernel_full
from semantic_kernel_adapter import basic, service

router = APIRouter(prefix="/sk", tags=["semantic_kernel"])


class ChatRequest(BaseModel):
    input_text: str = "travel to dinosaur age"


@router.post("/base")
async def sk_base(request: ChatRequest, kernel=Depends(get_kernel_full)):
    input_text = request.input_text
    result = await basic.chat.generate_joke(kernel, input_text, "silly")
    return result


class PromptRequest(BaseModel):
    input_text: str
