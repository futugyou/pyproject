from fastapi import FastAPI
from app.api import hello

app = FastAPI()
app.include_router(hello.router)
