from fastapi import APIRouter, Depends
from app.dependencies import get_db

router = APIRouter()


@router.get("/earnings")
async def list_earnings(db=Depends(get_db)):
    earnings = await db["earningss"].find().to_list(length=10)
    return earnings
