from motor.motor_asyncio import AsyncIOMotorClient
from web.core.config import MONGODB_URI

client = AsyncIOMotorClient(MONGODB_URI)
db = client.get_default_database()
