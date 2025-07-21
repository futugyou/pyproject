from fastapi import APIRouter
from . import hello, earnings

router = APIRouter()
router.include_router(hello.router)
router.include_router(earnings.router)
