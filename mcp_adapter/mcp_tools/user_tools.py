from ..models import UserInfo, UserRequest
from typing import Annotated
from pydantic import Field


async def get_user_info(
    userFilter: Annotated[UserRequest, Field(description="User search filter")],
) -> list[UserInfo]:
    """
    get user info by name and age
    """
    print(f"name: {userFilter.name}, age: {userFilter.age}")
    users: list[UserInfo] = [
        UserInfo(name="Alice", age=30),
        UserInfo(name="Bob", age=25),
    ]
    return users
