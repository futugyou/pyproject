from fastapi import APIRouter, Depends
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunction, KernelArguments
from pydantic import BaseModel
from app.dependencies import get_kernel

router = APIRouter(prefix="/sk")


class ChatRequest(BaseModel):
    input_text: str = "travel to dinosaur age"


@router.get("/base")
async def sk_base(request: ChatRequest, kernel=Depends(get_kernel)):
    input_text = request.input_text
    plugins_directory = "prompt_template_samples"
    funFunctions = kernel.add_plugin(
        parent_directory=str(plugins_directory), plugin_name="FunPlugin"
    )

    jokeFunction = funFunctions["Joke"]
    result = await kernel.invoke(jokeFunction, input=input_text, style="silly")
    return result


class PromptRequest(BaseModel):
    input_text: str


@router.post("/prompt")
async def sk_prompt(request: PromptRequest, kernel=Depends(get_kernel)):
    input_text = request.input_text

    prompt = """{{$input}}
    Summarize the content above.
    """
    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="summarize",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(
                name="input", description="The user input", is_required=True),
        ],
    )

    summarize = kernel.add_function(
        function_name="summarizeFunc",
        plugin_name="summarizePlugin",
        prompt_template_config=prompt_template_config,
    )
    summary = await kernel.invoke(summarize, input=input_text)
    return summary


@router.post("/argument")
async def sk_argument(kernel=Depends(get_kernel)):
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if it does not have an answer.

    {{$history}}
    User: {{$user_input}}
    ChatBot: """

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        name="chat",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(
                name="user_input", description="The user input", is_required=True
            ),
            InputVariable(
                name="history", description="The conversation history", is_required=True
            ),
        ],
    )

    chat_function = kernel.add_function(
        function_name="chat",
        plugin_name="chatPlugin",
        prompt_template_config=prompt_template_config,
    )
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful chatbot who is good about giving book recommendations."
    )
    await chat("Hi, I'm looking for book suggestions", kernel, chat_function, chat_history)
    await chat(
        "I love history and philosophy, I'd like to learn something new about Greece, any suggestion?", kernel, chat_function, chat_history
    )
    return chat_history


async def chat(input_text: str, kernel: Kernel, chat_function: KernelFunction, chat_history: ChatHistory) -> None:
    answer = await kernel.invoke(
        chat_function, KernelArguments(
            user_input=input_text, history=chat_history)
    )
    chat_history.add_user_message(input_text)
    chat_history.add_assistant_message(str(answer))
