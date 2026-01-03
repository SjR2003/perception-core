import time

from pipelines.stream_pipeline import StreamProcessor


def start_main_stream_process(zmq_config: dict, stream_config: dict) -> None:
    mainstream = StreamProcessor(
        zmq_config=zmq_config, stream_config=stream_config, main_stream=True
    )
    print("Starting main stream process...")
    time.sleep(2)
    mainstream.run()


def start_sub_stream_process(zmq_config: dict, stream_config: dict) -> None:
    substream = StreamProcessor(
        zmq_config=zmq_config, stream_config=stream_config, main_stream=False
    )
    print("Starting sub stream process...")
    time.sleep(2)
    substream.run()
