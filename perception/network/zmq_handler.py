from typing import Any, Dict, Optional
from collections import deque
from threading import Thread
import logging
import time
import zmq

from utils.latency_logger import measure_latency


class ZmqHandler:
    def __init__(
        self,
        module_endpoint: str,
        receive_endpoint: str,
        publisher: bool,
        receiver: bool,
        receive_config: Dict[str, Any],
    ):
        self.__logger = logging.getLogger(__name__)
        self.__module_endpoint = module_endpoint
        self.__publisher = publisher
        self.__receiver = receiver
        self.__receive_endpoint = receive_endpoint
        self.__receive_config = receive_config

        self.__ctx: Optional[zmq.Context] = None
        self.__pub_socket: Optional[zmq.Socket] = None
        self.__sub_socket: Optional[zmq.Socket] = None

        self.__receive_thread: Optional[Thread] = None

        if self.__receiver:
            self.__running = True
            self.__recv_queues: Dict[str, deque] = {
                stream_id: deque(maxlen=1)
                for stream_id in self.__receive_config.get("stream_ids", [])
            }

    def initialize_runtime(self):
        self.__ctx = zmq.Context.instance()

        if self.__publisher:
            self.__pub_socket = self.__ctx.socket(zmq.PUB)
            self.__pub_socket.setsockopt(zmq.SNDHWM, 20)
            self.__pub_socket.setsockopt(zmq.LINGER, 0)
            self.__pub_socket.connect(self.__module_endpoint)

        if self.__receiver:
            self.__sub_socket = self.__ctx.socket(zmq.SUB)
            self.__sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.__sub_socket.setsockopt(zmq.RCVHWM, 20)
            self.__sub_socket.connect(self.__receive_endpoint)

            self.__receive_thread = Thread(target=self.__receive_loop, daemon=True)
            self.__receive_thread.start()

    @measure_latency
    def publish(self, metadata: Dict[str, Any]) -> None:
        if not self.__publisher:
            raise RuntimeError("ZmqHandler.publish() called in receive-only mode")
        else:
            if self.__pub_socket is None:
                raise RuntimeError(
                    "ZmqHandler.publish() called before initialize_runtime()"
                )

            try:
                self.__pub_socket.send_pyobj(metadata)
            except Exception as e:
                print(f"[ZMQ] Publish error on {self.__endpoint}: {e}")

    def __receive_loop(self):
        print(f"[ZMQ] Starting receive loop...")
        while self.__running:
            try:
                msg = self.__sub_socket.recv_pyobj()
            except Exception as e:
                print(f"[ZMQ] Receive error: {e}")
                time.sleep(0.1)
                continue
            if not isinstance(msg, dict):
                continue

            metadata = msg.get("metadata")
            jpeg_bytes = msg.get("jpeg_bytes")
            if metadata is None or jpeg_bytes is None:
                continue

            if metadata["stream_id"] in self.__recv_queues:
                self.__recv_queues[metadata["stream_id"]].append(msg)

    @property
    def buffers(self) -> Dict[str, deque]:
        """
        Read-only access for batch sampler
        """
        if not self.__receiver:
            raise RuntimeError("ZmqHandler.buffers called in publish-only mode")
        else:
            return self.__recv_queues

    def get_message(self, stream_id: str) -> Dict[str, Dict[str, Any]]:
        if not self.__receiver:
            raise RuntimeError("ZmqHandler.get_message called in publish-only mode")
        else:
            if stream_id not in self.__recv_queues:
                raise ValueError(f"Stream ID {stream_id} not configured for receiving")

            if len(self.__recv_queues[stream_id]) == 0:
                return {}

            return self.__recv_queues[stream_id].pop()

    def close(self):
        self.__running = False
        time.sleep(0.2)
        if self.__receive_thread and self.__receive_thread.is_alive():
            self.__receive_thread.join(timeout=1.0)
            self.__receive_thread = None

        if self.__pub_socket:
            try:
                self.__pub_socket.close()
            except Exception:
                pass
        if self.__sub_socket:
            try:
                self.__sub_socket.close()
            except Exception:
                pass

        self.__pub_socket = None
        self.__sub_socket = None

    def __del__(self):
        self.close()
