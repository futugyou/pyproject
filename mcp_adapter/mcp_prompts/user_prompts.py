from typing import Any, Generic, Literal, Annotated
from pydantic import BaseModel, Field, AnyUrl


def greet_user(
    name: Annotated[str, Field(description="The user's name")],
    style: Annotated[
        str, Field(default="friendly", description="The style of greeting")
    ],
) -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."
