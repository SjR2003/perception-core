from argparse import ArgumentParser
import multiprocessing
import threading
import logging
import signal
import time

from network.proxy import ZmqProxy
from pipelines.pipeline_process import (
    start_main_stream_process,
    start_sub_stream_process,
)
from utils.logger import setup_logger
from utils.utils import load_yaml


def main():
    arg_parser = ArgumentParser(description="Perception node")
    arg_parser.add_argument(
        "--network_config",
        type=str,
        default=None,
        help="Path to Network configuration YAML file",
    )

    arg_parser.add_argument(
        "--main_stream_config",
        type=str,
        default=None,
        help="Path to Main stream configuration YAML file",
    )

    arg_parser.add_argument(
        "--sub_stream_config",
        type=str,
        default=None,
        help="Path to Sub stream configuration YAML file",
    )

    args = arg_parser.parse_args()

    logger = setup_logger("perception-node", level=logging.INFO)

    mainstream_cfg = load_yaml(args.main_stream_config)
    substream_cfg = load_yaml(args.main_stream_config)

    network_cfg = load_yaml(args.network_config)
    zmq_cfg = network_cfg.get("zmq", {})
    rest_cfg = network_cfg.get("rest", {})

    proxy = ZmqProxy(
        pub_port=zmq_cfg["perception_endpoint"],
        sub_port=zmq_cfg["proxy_endpoint"],
    )

    proxy.start()

    main_stream = multiprocessing.Process(
        target=start_main_stream_process,
        name="MainStreamProcess",
        args=(zmq_cfg, mainstream_cfg),
    )
    sub_stream = multiprocessing.Process(
        target=start_sub_stream_process,
        name="SubStreamProcess",
        args=(zmq_cfg, substream_cfg),
    )
    main_stream.start()
    sub_stream.start()

    logger.info("Perception node started")
    stop_event = threading.Event()

    def handle_sig(signum, frame):
        logger.info("Received signal %s → shutting down ...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    logger.info("Waiting for workers to close ...")
    logger.info("Stream-hub stopped cleanly")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

# python E:\projects\Ai\TraVis\perception\perception\main.py --network_config E:\projects\Ai\TraVis\perception\configs\network.yaml --main_stream_config E:\projects\Ai\TraVis\perception\configs\main_stream.yaml --sub_stream_config perception\configs\sub_stream.yaml
