from ..models import ShrimpTank, UserRequest
from typing import Annotated
from pydantic import Field


def name_shrimp(
    tank: ShrimpTank, extra_names: Annotated[list[str], Field(max_length=10)]
) -> list[str]:
    """List all shrimp names in the tank"""

    return [shrimp.name for shrimp in tank.shrimp] + extra_names
