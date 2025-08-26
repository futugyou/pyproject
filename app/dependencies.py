from fastapi import Depends
from app.core.database import db

from semantic_kernel_adapter import service


async def get_db():
    return db


async def get_kernel():
    return service.get_kernel()


async def get_kernel_full():
    return service.build_kernel_pipeline()
