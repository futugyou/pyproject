from dataclasses import dataclass
from datetime import datetime


@dataclass
class TokenModel:
    _id: str
    access_token: str
    token_type: str
    refresh_token: str
    expiry: datetime


@dataclass
class AuthModel:
    _id: str
    code_verifier: str
    code_challenge: str
    code_challenge_method: str
    state: str
    request_uri: str
    create_at: datetime
