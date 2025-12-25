from bytetracker import BYTETracker
from typing import Dict, List
import numpy as np
import torch

from utils.latency_logger import measure_latency


class YoloTensorRTTracker:
    def __init__(self, backend, tracker_config: Dict):
        self.__backend = backend
        self.__trackers: Dict[str, BYTETracker] = {}

        self.__track_thresh = tracker_config.get("track_thresh", 0.5)
        self.__match_thresh = tracker_config.get("match_thresh", 0.8)
        self.__track_buffer = tracker_config.get("track_buffer", 30)
        self.__fps = tracker_config.get("fps", 30)

    def _get_tracker(self, stream_id: str) -> BYTETracker:
        if stream_id not in self.__trackers:
            self.__trackers[stream_id] = BYTETracker(
                track_thresh=self.__track_thresh,
                track_buffer=self.__track_buffer,
                match_thresh=self.__match_thresh,
                frame_rate=self.__fps,
            )
        return self.__trackers[stream_id]

    @torch.no_grad()
    @measure_latency
    def infer(self, batch_tensor: torch.Tensor, metas: List[Dict]):

        outputs = self.__backend.infer(batch_tensor)
        results = []

        for out, meta in zip(outputs, metas):
            if meta.get("is_dummy", False):
                results.append({"meta": meta, "tracks": []})
                continue

            stream_id = meta["stream_id"]
            tracker = self._get_tracker(stream_id)

            if out.boxes is None or len(out.boxes) == 0:
                tracks = np.empty((0, 7), dtype=np.float32)

            else:
                boxes = out.boxes.xyxy.detach().cpu()
                scores = out.boxes.conf.detach().cpu()
                classes = out.boxes.cls.detach().cpu()

                dets = torch.cat([boxes, scores[:, None], classes[:, None]], dim=1)

                h = int(meta["height"])
                w = int(meta["width"])

                tracks = tracker.update(dets, [h, w])

            results.append({"meta": meta, "tracks": tracks})

        return results
