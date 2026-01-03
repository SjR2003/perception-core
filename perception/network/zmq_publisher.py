from collections import deque
from threading import Thread
import logging
import time
import zmq

from utils.latency_logger import measure_latency


class ZMQPublisher:
    def __init__(
        self, context: zmq.Context.instance, endpoint: str, buffer_size: int = 30
    ):
        self.__logger = logging.getLogger(__name__)
        self.__context = context
        self.__endpoint = endpoint
        self.__socket = None

        self.__message_queue = deque(maxlen=buffer_size)

        self.__running = False
        self.__send_thread = Thread(target=self.__send_loop, daemon=True)

    def start(self) -> bool:
        if not self.__connect():
            return False

        time.sleep(0.2)
        self.__running = True
        self.__send_thread.start()
        self.__logger.info("Started ZMQ sending thread")
        return True

    @measure_latency
    def publish(self, topic: str, metadata: bytes) -> None:
        topic_b = topic.encode("utf-8")
        self.__message_queue.append([topic_b, metadata])

    def close(self) -> None:
        self.__running = False
        time.sleep(0.2)
        if self.__send_thread and self.__send_thread.is_alive():
            self.__send_thread.join(timeout=1.0)
            self.__send_thread = None

        if self.__socket:
            try:
                self.__socket.close()
            except Exception as e:
                self.__logger.error(f"There is an issue with closing PUB socket! {e}")
                pass

        self.__socket = None

    def __connect(self) -> bool:
        try:
            self.__socket = self.__context.socket(zmq.PUB)
            self.__socket.setsockopt(zmq.SNDHWM, 20)
            self.__socket.setsockopt(zmq.LINGER, 0)
            self.__socket.connect(self.__endpoint)
            self.__logger.info(f"Connected PUB to {self.__endpoint}")
            return True
        except Exception as e:
            self.__logger.error(f"Connect Failed: {e}")
            return False

    def __send_loop(self) -> None:
        while self.__running:
            if len(self.__message_queue) > 0:
                message = self.__message_queue.popleft()
                try:
                    self.__socket.send_pyobj(message)
                except Exception as e:
                    self.__logger.error(f"Publish error on {self.__endpoint}: {e}")
            else:
                time.sleep(0.005)
