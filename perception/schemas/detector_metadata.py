from pydantic import BaseModel
from typing import List
import numpy as np

from schemas.stream_metadata import StreamHubPacketMetadata

class Detections(BaseModel):
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    model_config = {"arbitrary_types_allowed": True}

class DetectorData(BaseModel):
    metadata: StreamHubPacketMetadata
    detections: List

    model_config = {"arbitrary_types_allowed": True}

class DetectorPacket(BaseModel):
    result: List[DetectorData]

    model_config = {"arbitrary_types_allowed": True}