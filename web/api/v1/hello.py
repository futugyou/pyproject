from fastapi import APIRouter

router = APIRouter()


@router.get("/hello")
def read_root():
    return {"message": "Hello from FastAPI (local or Vercel)"}
