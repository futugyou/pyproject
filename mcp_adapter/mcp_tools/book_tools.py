from mcp.server.fastmcp import Context
from mcp.types import (
    ClientCapabilities,
    ElicitationCapability,
)


async def book_table(
    date: str,
    time: str,
    ctx: Context,
) -> str:
    """Book a table with date availability check."""
    # Check if date is available
    if date == "2024-12-25":
        if ctx.session.check_client_capability(
            ClientCapabilities(elicitation=ElicitationCapability())
        ):
            result = await ctx.elicit(
                message=(
                    f"No tables available on {date}. Would you like to try another date?"
                ),
                schema=BookingPreferences,
            )
            if result.action == "accept" and result.data:
                if result.data.checkAlternative:
                    return f"[SUCCESS] Booked for {result.data.alternativeDate}"
                return "[CANCELLED] No booking made"
            return "[CANCELLED] Booking cancelled"
    # Date available
    return f"[SUCCESS] Booked for {date} at {time}"
