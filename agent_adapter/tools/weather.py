from typing import Annotated
from pydantic import Field
from agent_framework import ai_function


@ai_function(
    name="weather_tool", description="Retrieves weather information for any location"
)
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    return f"The weather in {location} is cloudy with a high of 15°C."
