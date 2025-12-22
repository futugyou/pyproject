import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL_ID = os.getenv("OPENAI_CHAT_MODEL_ID")
GOOGLE_TEXT_EMBEDDING_MODEL_ID = os.getenv("GOOGLE_TEXT_EMBEDDING_MODEL_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
