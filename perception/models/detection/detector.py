from typing import List
import torch

from models.detection.detection_baseclass import DetectionBackend
from schemas.detector_metadata import Detections, DetectorData, DetectorPacket


class YoloTensorRTDetector:
    def __init__(self, backend: DetectionBackend) -> None:
        self.__backend = backend

    def infer(self, batch_tensor: torch.Tensor, metas: List) -> DetectorPacket:
        outputs = self.__backend.infer(batch_tensor)

        results = []
        for i, r in enumerate(outputs):
            results.append(
                DetectorData(
                    metadata=metas[i],
                    detections=Detections(
                        boxes=r.boxes.xyxy.cpu().numpy(),
                        scores=r.boxes.conf.cpu().numpy(),
                        classes=r.boxes.cls.cpu().numpy(),
                    )
                )
            )
        return DetectorPacket(result=results)
