import uuid, hashlib, base64, requests
from datetime import datetime
from pymongo import MongoClient
from fastapi import Request, HTTPException
from requests_oauthlib import OAuth2Session
from jose import jwt, jwk
from jose.utils import base64url_decode
from models import AuthModel, TokenModel
from config import AuthOptions


class AuthClient:
    def __init__(self, options: AuthOptions):
        self.options = options
        self.client = MongoClient(options.db_url)
        self.db = self.client[options.db_name]
        self.oauth = OAuth2Session(
            client_id=options.client_id,
            redirect_uri=options.redirect_url,
            scope=options.scopes,
        )

    def gen_code_challenge_s256(self, verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def create_auth_code_url(self, req: Request) -> str:
        code_verifier = str(uuid.uuid4())
        code_challenge = self.gen_code_challenge_s256(code_verifier)
        state = uuid.uuid4().hex

        model = AuthModel(
            id=state,
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            code_challenge_method="S256",
            state=state,
            request_uri=str(req.url),
            create_at=datetime.utcnow(),
        )
        self.db.oauth_request.insert_one(model.to_mongo())

        auth_url, _ = self.oauth.authorization_url(
            url=self.options.auth_server_url + self.options.auth_url,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method="S256",
            access_type="offline",
        )
        return auth_url

    def get_auth_request_info(self, state: str) -> AuthModel:
        doc = self.db.oauth_request.find_one({"_id": state})
        if not doc:
            raise HTTPException(status_code=400, detail="Invalid state")
        return AuthModel.model_validate(doc)

    def exchange_token(self, req: Request, code: str, state: str):
        auth_model = self.get_auth_request_info(state)
        token = self.oauth.fetch_token(
            token_url=self.options.auth_server_url + self.options.token_url,
            code=code,
            client_secret=self.options.client_secret,
            include_client_id=True,
            code_verifier=auth_model.code_verifier,
        )
        model = TokenModel(
            id=token["access_token"],
            access_token=token["access_token"],
            token_type=token.get("token_type", ""),
            refresh_token=token.get("refresh_token"),
            expiry=datetime.utcfromtimestamp(token["expires_at"]),
        )
        self.db.oauth_request.insert_one(model.to_mongo())
        return token

    def verify_token(self, token_str: str) -> dict:
        jwks_uri = self.options.auth_server_url + "/.well-known/jwks.json"
        jwks = requests.get(jwks_uri).json()
        headers = jwt.get_unverified_header(token_str)
        kid = headers["kid"]
        key_data = next(k for k in jwks["keys"] if k["kid"] == kid)
        public_key = jwk.construct(key_data)
        message, encoded_sig = token_str.rsplit(".", 1)
        if not public_key.verify(
            message.encode(), base64url_decode(encoded_sig.encode())
        ):
            raise HTTPException(status_code=401, detail="Invalid token signature")
        return jwt.decode(token_str, public_key, algorithms=[headers["alg"]])
