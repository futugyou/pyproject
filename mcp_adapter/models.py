from typing import Annotated
from pydantic import BaseModel, Field


class Shrimp(BaseModel):
    name: Annotated[str, Field(max_length=10)]


class ShrimpTank(BaseModel):
    shrimp: list[Shrimp]


class UserRequest(BaseModel):
    name: Annotated[str, Field(max_length=10)]
    age: Annotated[int, Field(ge=0, le=120)]


class UserInfo(BaseModel):
    name: Annotated[str, Field(max_length=10)]
    age: Annotated[int, Field(ge=0, le=120)]


class BookingPreferences(BaseModel):
    """Schema for collecting user preferences."""

    checkAlternative: bool = Field(description="Would you like to check another date?")
    alternativeDate: str = Field(
        default="2024-12-26",
        description="Alternative date (YYYY-MM-DD)",
    )
