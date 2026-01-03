from pydantic import BaseModel


class StreamHubPacketMetadata(BaseModel):
    stream_id: str
    frame_id: int
    timestamp: float
    source: str
    events: dict
    frame_size: int

    model_config = {"extra": "allow"}
