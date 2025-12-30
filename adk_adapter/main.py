from dotenv import load_dotenv

load_dotenv()

import os
from fastapi import FastAPI
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini


model_with_key = Gemini(
    model=os.getenv("GOOGLE_CHAT_MODEL_ID"), api_key=os.getenv("GOOGLE_API_KEY")
)

agent = LlmAgent(
    name="assistant", model=model_with_key, instruction="Be helpful and fun!"
)

adk_agent = ADKAgent(
    adk_agent=agent,
    app_name="demo_app",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True,
)

app = FastAPI()
add_adk_fastapi_endpoint(app, adk_agent, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
