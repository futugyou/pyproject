import datetime
from typing import Any


async def get_time() -> dict[str, Any]:
    """
    Get the current server time.

    This tool demonstrates that system information can be protected
    by OAuth authentication. User must be authenticated to access it.
    """
    now = datetime.datetime.now()
    return {
        "current_time": now.isoformat(),
        "timezone": "UTC",
        "timestamp": now.timestamp(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
