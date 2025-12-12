import asyncio
import os
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

load_dotenv()


async def JokeAgent(query: str) -> str:
    client = OpenAIChatClient(
        model_id=os.getenv("GOOGLE_CHAT_MODEL_ID"),
        base_url=os.getenv("GOOGLE_URL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    agent = client.create_agent(
        instructions="You are good at telling jokes.", name="Joker"
    )

    result = await agent.run(query)
    text = result.text
    print(f"message: {text}")
    return text


if __name__ == "__main__":
    asyncio.run(JokeAgent("Tell me a joke about a pirate."))
