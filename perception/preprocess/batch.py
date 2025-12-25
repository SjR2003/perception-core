from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2

from utils.latency_logger import measure_latency


def decode_and_preprocess_worker(item, input_size):
    if "jpeg_bytes" not in item:
        return None
    arr = np.frombuffer(item["jpeg_bytes"], np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, input_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.uint8)

    return {
        "image": frame,
        "meta": {**item.get("metadata", {}), "height": h, "width": w},
    }


class BatchDecodePipeline:
    def __init__(self, workers=4, input_size=(640, 640)):
        self.__pool = ThreadPoolExecutor(max_workers=workers)
        self.__input_size = input_size

    @measure_latency
    def process_batch(self, batch_items):
        futures = [
            self.__pool.submit(decode_and_preprocess_worker, item, self.__input_size)
            for item in batch_items
        ]

        images = []
        metas = []

        for f in as_completed(futures):
            r = f.result()
            if r is None:
                continue
            images.append(r["image"])
            metas.append(r["meta"])

        if not images:
            return None

        return {"images": images, "meta": metas}
