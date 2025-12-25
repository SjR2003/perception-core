from models.yolo.trt_detector import YoloTensorRTDetectionBackend
from utils.latency_logger import measure_latency


class YoloTensorRTDetector:
    def __init__(self, backend: YoloTensorRTDetectionBackend):
        self.__backend = backend

    @measure_latency
    def infer(self, batch_tensor, metas):
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
