import uuid
import hashlib
import base64
import json
from datetime import datetime
from pymongo import MongoClient
from models import TokenModel, AuthModel
from config import AuthOptions
from requests_oauthlib import OAuth2Session
import requests
from jose import jwt, jwk
from jose.utils import base64url_decode


class AuthService:
    def __init__(self, options: AuthOptions):
        self.options = options
        self.client = MongoClient(options.db_url)
        self.db = self.client[options.db_name]

        self.oauth = OAuth2Session(
            client_id=options.client_id,
            redirect_uri=options.redirect_url,
            scope=options.scopes
        )

    def gen_code_challenge_s256(self, verifier: str) -> str:
        sha256_digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(sha256_digest).rstrip(b'=').decode()

    def create_auth_code_url(self) -> str:
        code_verifier = str(uuid.uuid4())
        code_challenge = self.gen_code_challenge_s256(code_verifier)
        code_challenge_method = "S256"
        state = uuid.uuid4().hex

        auth_model = AuthModel(
            _id=state,
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
            request_uri=request.full_path,
            create_at=datetime.utcnow()
        )
        self.db.oauth_request.insert_one(auth_model.__dict__)

        return self.oauth.authorization_url(
            url=self.options.auth_server_url + self.options.auth_url,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            access_type='offline'
        )[0]

    def get_auth_request_info(self, state: str) -> AuthModel:
        doc = self.db.oauth_request.find_one({'_id': state})
        if not doc:
            raise ValueError("State not found")
        return AuthModel(**doc)

    def save_token(self, token: dict):
        model = TokenModel(
            _id=token["access_token"],
            access_token=token["access_token"],
            token_type=token.get("token_type", ""),
            refresh_token=token.get("refresh_token", ""),
            expiry=datetime.fromisoformat(token["expires_at"])
        )
        self.db.oauth_request.insert_one(model.__dict__)

    def exchange_token(self, code: str, state: str):
        auth_model = self.get_auth_request_info(state)
        token = self.oauth.fetch_token(
            token_url=self.options.auth_server_url + self.options.token_url,
            code=code,
            client_secret=self.options.client_secret,
            include_client_id=True,
            code_verifier=auth_model.code_verifier
        )
        self.save_token(token)
        return token

    def verify_token(self, token_str: str) -> dict:
        jwks_uri = self.options.auth_server_url + "/.well-known/jwks.json"
        jwks = requests.get(jwks_uri).json()
        unverified_header = jwt.get_unverified_header(token_str)

        kid = unverified_header["kid"]
        key_data = next(k for k in jwks["keys"] if k["kid"] == kid)
        public_key = jwk.construct(key_data)

        message, encoded_sig = token_str.rsplit('.', 1)
        decoded_sig = base64url_decode(encoded_sig.encode())

        if not public_key.verify(message.encode(), decoded_sig):
            raise ValueError("Signature verification failed")

        return jwt.decode(token_str, public_key, algorithms=[unverified_header["alg"]])
