from fastapi import Depends
from app.core.database import db


async def get_db():
    return db
