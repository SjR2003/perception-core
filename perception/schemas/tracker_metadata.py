from pydantic import BaseModel
from typing import List
import numpy as np

from schemas.stream_metadata import StreamHubPacketMetadata


class TrackerData(BaseModel):
    metadata: StreamHubPacketMetadata
    tracks: np.ndarray

    model_config = {"arbitrary_types_allowed": True}

class TrackerPacket(BaseModel):
    result: List[TrackerData]

    model_config = {"arbitrary_types_allowed": True}