from typing import Dict, List, Optional
from collections import deque
import time
import copy


class FrameBatchSampler:
    def __init__(
        self,
        buffers: Dict[str, deque],
        batch_size: int,
        max_wait_ms: int = 20,
    ):
        self.__buffers = buffers
        self.__batch_size = batch_size
        self.__max_wait_ms = max_wait_ms
        self.__stream_ids = list(buffers.keys())
        self.__rr_index = 0

        self.__last_valid_item: Optional[dict] = None

    def _make_dummy(self) -> dict:
        if self.__last_valid_item is None:
            time.sleep(0.001)
            return None

        dummy = copy.deepcopy(self.__last_valid_item)
        dummy["is_dummy"] = True
        dummy["frame_id"] = -1
        dummy["timestamp"] = time.time()
        return dummy

    def get_batch(self) -> List[dict]:
        batch = []
        start = time.monotonic()

        while len(batch) < self.__batch_size:
            elapsed_ms = (time.monotonic() - start) * 1000
            if elapsed_ms >= self.__max_wait_ms:
                break

            if not self.__stream_ids:
                time.sleep(0.001)
                continue

            stream_id = self.__stream_ids[self.__rr_index]
            self.__rr_index = (self.__rr_index + 1) % len(self.__stream_ids)

            buf = self.__buffers.get(stream_id)
            if not buf or len(buf) == 0:
                continue

            item = buf.pop()
            item["is_dummy"] = False

            self.__last_valid_item = item

            batch.append(item)

        while len(batch) < self.__batch_size:
            dummy = self._make_dummy()
            if dummy is not None:
                batch.append(dummy)

        return batch
