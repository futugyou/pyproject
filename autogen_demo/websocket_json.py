import orjson
from starlette.websockets import WebSocket, WebSocketDisconnect


async def send_orjson(websocket: WebSocket, obj: dict) -> None:
    try:
        await websocket.send_bytes(orjson.dumps(obj))
    except WebSocketDisconnect:
        pass


async def receive_orjson(websocket: WebSocket) -> dict:
    msg = await websocket.receive()
    raw = msg.get("text") or msg.get("bytes")

    if (
        raw is None
        or (isinstance(raw, str) and not raw.strip())
        or (isinstance(raw, bytes) and not raw.strip())
    ):
        raise ValueError("Received empty WebSocket message")

    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    return orjson.loads(raw)
