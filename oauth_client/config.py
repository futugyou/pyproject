from pydantic import BaseModel
from typing import List


class AuthOptions(BaseModel):
    auth_server_url: str
    client_id: str
    client_secret: str
    scopes: List[str]
    redirect_url: str
    auth_url: str
    token_url: str
    db_url: str
    db_name: str
