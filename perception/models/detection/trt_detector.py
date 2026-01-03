from typing import List
import torch

from models.detection.detection_baseclass import DetectionBackend
from models.detection.loader import TensorRTModel
from utils.latency_logger import measure_latency


class YoloTensorRTDetectionBackend(DetectionBackend):
    def __init__(self, trt_model: TensorRTModel) -> None:
        self.__model = trt_model

    @torch.no_grad()
    @measure_latency
    def infer(self, batch_tensor: torch.Tensor) -> List:
        return self.__model.model.predict(source=batch_tensor, device=0, verbose=False)
