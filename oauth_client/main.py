from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from client import AuthClient
from config import AuthOptions

app = FastAPI()

auth_service = AuthClient(
    AuthOptions(
        auth_server_url="https://auth.example.com",
        client_id="client-id",
        client_secret="secret",
        scopes=["openid", "profile"],
        redirect_url="http://localhost:5000/oauth/callback",
        auth_url="/authorize",
        token_url="/token",
        db_url="mongodb://localhost:27017",
        db_name="oauth_db",
    )
)


@app.get("/oauth/start")
def start_oauth(request: Request):
    auth_url = auth_service.create_auth_code_url(request)
    return RedirectResponse(auth_url)


@app.get("/oauth/callback")
def oauth_callback(request: Request, code: str = "", state: str = ""):
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")
    token = auth_service.exchange_token(request, code, state)
    return JSONResponse(token)


@app.get("/oauth/verify")
def verify_token(token: str):
    info = auth_service.verify_token(token)
    return JSONResponse(info)
