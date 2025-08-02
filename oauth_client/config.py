from dataclasses import dataclass


@dataclass
class AuthOptions:
    auth_server_url: str
    client_id: str
    client_secret: str
    scopes: list[str]
    redirect_url: str
    auth_url: str
    token_url: str
    db_url: str
    db_name: str
