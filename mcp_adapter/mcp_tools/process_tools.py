from mcp.server.fastmcp import Context


async def process_data(data: str, ctx: Context, steps: int = 5) -> str:
    """Process data with logging."""
    # Different log levels
    await ctx.debug(f"Debug: Processing '{data}'")
    for i in range(steps):
        progress = (i + 1) / steps
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"Step {i + 1}/{steps}",
        )
    await ctx.debug(f"Completed step {i + 1}")
    # Notify about resource changes
    await ctx.session.send_resource_list_changed()
    return f"Processed: {data}"
