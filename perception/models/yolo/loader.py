from ultralytics import YOLO
import torch


class TensorRTModel:
    def __init__(self, engine_path: str, task="detect"):
        self.__model = YOLO(engine_path, task=task)

    def warmup(self, batch=1, imgsz=640):
        dummy = torch.zeros(
            (batch, 3, imgsz, imgsz), device="cuda", dtype=torch.float16
        )
        self.__model.predict(dummy, device=0, verbose=False)
