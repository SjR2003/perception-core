from network.zmq_handler import ZmqHandler
from preprocess.batch_scheduler import FrameBatchSampler
from preprocess.batch import BatchDecodePipeline
from models.yolo.detector import YoloTensorRTDetector as YoloDetector
from models.yolo.tracker import YoloTensorRTTracker as YoloTracker
from models.yolo.trt_detector import YoloTensorRTDetectionBackend
from models.yolo.exporter import YoloExporter
from models.yolo.loader import TensorRTModel


class StreamProcessor:
    def __init__(self, zmq_config: dict, stream_config: dict, main_stream: bool = True):
        self.__running = True
        self.__main_stream = main_stream
        self.__zmq_config = zmq_config
        self.__streams = stream_config.get("receive_config", {})
        self.__batch_config = stream_config.get("batch", {})
        self.__model_config = stream_config.get("model", {})
        self.__preprocess_config = stream_config.get("preprocess", {})

        self.__decode_pipeline = BatchDecodePipeline(
            workers=self.__preprocess_config.get("workers", 4),
            input_size=self.__preprocess_config.get("input_size", (640, 640)),
        )

        self.__load_model()
        self.__setup()

    def __setup(self):
        self.__hub_zmq = ZmqHandler(
            module_endpoint=self.__zmq_config["hub_endpoint"],
            publisher=False,
            receiver=True,
            receive_endpoint=self.__zmq_config["hub_endpoint"],
            receive_config=self.__streams,
        )
        self.__hub_zmq.initialize_runtime()

        self.__pub_node = None
        if self.__main_stream:
            self.__pub_node = ZmqHandler(
                module_endpoint=self.__zmq_config["perception_endpoint"],
                publisher=True,
                receiver=False,
                receive_endpoint=self.__zmq_config["hub_endpoint"],
                receive_config=self.__streams,
            )
            self.__pub_node.initialize_runtime()

        self.__batch_sampler = FrameBatchSampler(
            buffers=self.__hub_zmq.buffers,
            batch_size=self.__batch_config.get("max_batch_size", 8),
            max_wait_ms=self.__batch_config.get("max_wait_ms", 5),
        )

    def __load_model(self):
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

    def run(self):
        while self.__running:
            batch_items = self.__batch_sampler.get_batch()
            if not batch_items:
                continue

            batch = self.__decode_pipeline.process_batch(batch_items)

            if batch is None:
                continue

            outputs = self.__model.infer(batch["images"], batch["meta"])
            if outputs is None:
                continue
            
            if self.__pub_node:
                self.__pub_node.publish(outputs)
