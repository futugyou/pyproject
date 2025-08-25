from mcp.server.fastmcp.utilities.types import Image


def take_screenshot() -> Image:
    """
    Load a screenshot from a local file and return it as a compressed JPEG image.
    Replace the file path with the actual screenshot you want to send.
    """
    from PIL import Image as PILImage
    import io

    file_path = "./17871902.png"
    buffer = io.BytesIO()
    image = PILImage.open(file_path).convert("RGB")
    image.save(buffer, format="JPEG", quality=60, optimize=True)
    return Image(data=buffer.getvalue(), format="jpeg")
