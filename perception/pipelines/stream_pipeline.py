import zmq

from models.detection.detector import YoloTensorRTDetector as YoloDetector
from models.detection.tracker import YoloTensorRTTracker as YoloTracker
from models.detection.trt_detector import YoloTensorRTDetectionBackend
from preprocess.batch_scheduler import FrameBatchSampler
from preprocess.batch_process import BatchDecodePipeline
from models.detection.exporter import YoloExporter
from models.detection.loader import TensorRTModel
from network.zmq_subscriber import ZMQSubscriber
from network.zmq_publisher import ZMQPublisher


class StreamProcessor:
    def __init__(self, zmq_config: dict, stream_config: dict, main_stream: bool = True):
        self.__running = True
        self.__main_stream = main_stream
        self.__zmq_config = zmq_config
        self.__receive_config = stream_config.get("receive_config", {})
        self.__batch_config = stream_config.get("batch", {})
        self.__model_config = stream_config.get("model", {})
        self.__preprocess_config = stream_config.get("preprocess", {})

        self.__decode_pipeline = BatchDecodePipeline(
            decode_enabled=self.__preprocess_config.get("decode", {}).get(
                "enabled", False
            ),
            decode_type=self.__preprocess_config.get("decode", {}).get(
                "type", "ffmpeg"
            ),
            workers=self.__preprocess_config.get("workers", 4),
            input_size=self.__preprocess_config.get("input_size", (640, 640)),
        )

        self.__load_model()
        self.__setup()

    def __setup(self) -> None:
        zmq_ctx = zmq.Context.instance()
        self.__hub_zmq = ZMQSubscriber(
            context=zmq_ctx,
            endpoint=self.__zmq_config["hub_endpoint"],
            receive_config=self.__receive_config,
        )
        self.__hub_zmq.start()

        self.__pub_node = None
        if self.__main_stream:
            self.__pub_node = ZMQPublisher(
                context=zmq_ctx, endpoint=self.__zmq_config["perception_endpoint"]
            )
            self.__pub_node.start()

        self.__batch_sampler = FrameBatchSampler(
            buffers=self.__hub_zmq.buffers,
            batch_size=self.__batch_config.get("max_batch_size", 8),
            max_wait_ms=self.__batch_config.get("max_wait_ms", 5),
        )

    def __load_model(self) -> None:
        prefix = "main" if self.__main_stream else "sub"
        exporter = YoloExporter(prefix)
        ret = exporter.export_to_trt(
            model_path=self.__model_config["model_path"],
            model_name=self.__model_config["model_name"],
            img_size=self.__model_config["img_size"],
            batch=self.__model_config["batch_size"],
            fp16=self.__model_config["fp16"],
        )
        if ret is None:
            raise RuntimeError("Failed to export model to TensorRT")

        loader = TensorRTModel(engine_path=ret)
        backend = YoloTensorRTDetectionBackend(loader)
        if self.__main_stream:
            self.__model = YoloTracker(
                backend=backend, tracker_config=self.__model_config.get("tracker", {})
            )
        else:
            self.__model = YoloDetector(backend=backend)

    def run(self) -> None:
        while self.__running:
            batch_items = self.__batch_sampler.get_batch()
            if not batch_items:
                continue

            batch = self.__decode_pipeline.process_batch(batch_items)

            if batch is None:
                continue

            outputs = self.__model.infer(batch.frames, batch.metadata)
            if outputs is None:
                continue

            if self.__pub_node:
                self.__pub_node.publish(topic="perception", metadata=outputs)
