from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class TokenModel(BaseModel):
    id: str = Field(alias="_id")
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expiry: datetime

    def to_mongo(self):
        return self.model_dump(by_alias=True)


class AuthModel(BaseModel):
    id: str = Field(alias="_id")
    code_verifier: str
    code_challenge: str
    code_challenge_method: str
    state: str
    request_uri: str
    create_at: datetime

    def to_mongo(self):
        return self.model_dump(by_alias=True)
