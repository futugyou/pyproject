from fastapi import Depends
from app.core.database import db
from app.core.kernel import kernel


async def get_db():
    return db


async def get_kernel():
    return kernel
