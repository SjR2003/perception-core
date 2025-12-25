import torch

from models.yolo.loader import TensorRTModel


class YoloTensorRTDetectionBackend:
    def __init__(self, trt_model: TensorRTModel):
        self.__model = trt_model.__model

    @torch.no_grad()
    def infer(self, batch_tensor):
        return self.__model.predict(batch_tensor, device=0, verbose=False)
