from ultralytics import YOLO
import torch


class TensorRTModel:
    def __init__(self, engine_path: str, task: str = "detect") -> None:
        self.model = YOLO(engine_path, task=task)

    def warmup(self, batch: int = 1, imgsz: tuple = (640, 640)) -> None:
        dummy = torch.zeros(
            (batch, 3, imgsz[0], imgsz[1]), device="cuda", dtype=torch.float16
        )
        self.model.predict(dummy, device=0, verbose=False)
