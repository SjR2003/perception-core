from pydantic import BaseModel
from typing import List
import numpy as np

from schemas.stream_metadata import StreamHubPacketMetadata


class BatchItem(BaseModel):
    metadata: StreamHubPacketMetadata
    frame: np.ndarray

    model_config = {"arbitrary_types_allowed": True}


class Batch(BaseModel):
    metadata: List[StreamHubPacketMetadata]
    frames: List[np.ndarray]

    model_config = {"arbitrary_types_allowed": True}
