from typing import List
import torch

from models.detection.detection_baseclass import DetectionBackend


class YoloTensorRTDetector:
    def __init__(self, backend: DetectionBackend) -> None:
        self.__backend = backend

    def infer(self, batch_tensor: torch.Tensor, metas: List) -> List:
        outputs = self.__backend.infer(batch_tensor)

        results = []
        for i, r in enumerate(outputs):
            results.append(
                {
                    "meta": metas[i],
                    "boxes": r.boxes.xyxy.cpu(),
                    "scores": r.boxes.conf.cpu(),
                    "classes": r.boxes.cls.cpu(),
                }
            )

        return results
