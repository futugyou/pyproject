from pydantic import BaseModel, Field
from typing import Annotated
from agent_framework import ai_function


class LightInfo(BaseModel):
    """Information about a light."""

    id: str
    name: str
    is_on: bool


lights = [
    LightInfo(id="1", name="Table Lamp", is_on=False),
    LightInfo(id="2", name="Porch light", is_on=False),
    LightInfo(id="3", name="Chandelier", is_on=True),
]


@ai_function(
    name="change_state",
    description="Changes the state of the light",
    approval_mode="always_require",
)
def change_state(
    id: Annotated[str, Field(description="the light id")],
    is_on: Annotated[
        bool,
        Field(
            description="light status, `true` means open the light, `false` means close the light"
        ),
    ],
) -> LightInfo | None:
    light = next((light for light in lights if light.id == id), None)

    if not light:
        return None

    light.is_on = is_on
    return light


@ai_function(
    name="get_lights", description="Gets a list of lights and their current state"
)
def get_lights() -> list[LightInfo]:
    return lights
