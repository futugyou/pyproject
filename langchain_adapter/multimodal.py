from langchain.chat_models import init_chat_model

from typing import Optional
from pydantic import BaseModel, Field

from .option import LangChainOption


class MultimodalData(BaseModel):
    image_url: str


def multimodal(data: MultimodalData, config: LangChainOption):
    llm = init_chat_model(
        config.lang_google_chat_model,
        model_provider="google_genai",
        api_key=config.lang_google_api_key,
    )

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this image in short text:",
            },
            {
                "type": "image_url",
                "source_type": "url",
                "image_url": data.image_url,
            },
        ],
    }
    response = llm.invoke([message])
    return response.text()


if __name__ == "__main__":
    data = MultimodalData(
        image_url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
    )
    result = multimodal(data, LangChainOption())
    print(result)
