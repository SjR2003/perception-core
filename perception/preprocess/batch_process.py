from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import cv2

from schemas.stream_packet import StreamHubPacket
from utils.latency_logger import measure_latency
from schemas.batch import BatchItem, Batch


def decode_and_preprocess_worker(
    item, input_size, decode_enabled, decode_type
) -> BatchItem:
    frame = item.frame
    if frame is None:
        return None

    if decode_enabled and decode_type == "ffmpeg":
        arr = np.frombuffer(frame, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, tuple(input_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.uint8)

    item.metadata.height = h
    item.metadata.width = w
    if "is_dummy" not in item.metadata.model_fields:
        item.metadata.is_dummy = False

    return BatchItem(frame=frame, metadata=item.metadata)


class BatchDecodePipeline:
    def __init__(
        self,
        decode_enabled: bool,
        decode_type: str,
        workers: int = 4,
        input_size: tuple = (640, 640),
    ) -> None:
        self.__pool = ThreadPoolExecutor(max_workers=workers)
        self.__input_size = input_size
        self.__decode_enabled = decode_enabled
        self.__decode_type = decode_type

    @measure_latency
    def process_batch(self, batch_items: List[StreamHubPacket]) -> Batch:
        futures = [
            self.__pool.submit(
                decode_and_preprocess_worker,
                item,
                self.__input_size,
                self.__decode_enabled,
                self.__decode_type,
            )
            for item in batch_items
        ]

        frames = []
        metas = []

        for f in as_completed(futures):
            r = f.result()
            if r is None:
                continue
            frames.append(r.frame)
            metas.append(r.metadata)

        if not frames:
            return None

        return Batch(frames=frames, metadata=metas)
