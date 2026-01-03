from pydantic import BaseModel
import numpy as np

from schemas.stream_metadata import StreamHubPacketMetadata


class StreamHubPacket(BaseModel):
    metadata: StreamHubPacketMetadata
    frame: np.ndarray

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}
