from pydantic import BaseModel
import numpy as np

from schemas.metadata import StreamHubPacketMetadata

class StreamHubPacket(BaseModel):
    metadata: StreamHubPacketMetadata
    frame: np.ndarray

    model_config = {
        "arbitrary_types_allowed": True
    }