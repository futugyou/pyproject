from fastapi import APIRouter
from . import hello, earnings, sk_basic

router = APIRouter()
router.include_router(hello.router)
router.include_router(earnings.router)
router.include_router(sk_basic.router)
