from collections import deque
from threading import Thread
from typing import Dict
import logging
import zmq

from schemas.stream_packet import StreamHubPacket


class ZMQSubscriber:
    def __init__(
        self,
        context: zmq.Context.instance,
        endpoint: str,
        receive_config: Dict,
        buffer_size: int = 1,
    ) -> None:
        self.__logger = logging.getLogger(__name__)
        self.__context = context
        self.__endpoint = endpoint
        self.__socket = None
        self.__topic = receive_config.get("topic", []).encode("utf-8")

        self.__packet_queues: Dict[str, deque] = {
            stream_id: deque(maxlen=buffer_size)
            for stream_id in receive_config.get("stream_ids", [])
        }

        self.__running = False
        self.__recv_thread = Thread(target=self.__receive_loop, daemon=True)

    def start(self) -> bool:
        if not self.__connect():
            return False
        self.__running = True
        self.__recv_thread.start()
        self.__logger.info("Started ZMQ receive thread")
        print("Started ZMQ receive thread")
        return True

    @property
    def buffers(self) -> Dict[str, deque]:
        return self.__packet_queues

    def get_message(self, stream_id: str) -> StreamHubPacket:
        if len(self.__packet_queues[stream_id]) == 0:
            return {}

        return self.__packet_queues[stream_id][-1]

    def __connect(self) -> bool:
        try:
            self.__socket = self.__context.socket(zmq.SUB)
            self.__socket.setsockopt(zmq.RCVHWM, 10)
            self.__socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.__socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.__socket.connect(self.__endpoint)
            self.__logger.info(f"Connected SUB to {self.__endpoint}")
            return True
        except Exception as e:
            self.__logger.error(f"Connect Failed: {e}")
            return False

    def __receive_loop(self) -> None:
        while self.__running:
            try:
                message = self.__socket.recv_pyobj()
                self.__logger.debug(f"Received message with {len(message)} parts")

                # Validate received data
                packet = StreamHubPacket.model_validate(message)
                metadata = packet.metadata
                if packet.frame is None or metadata is None:
                    self.__logger.debug(f"Packet.frame or packet.metadata was none")
                    continue

                if metadata.stream_id in self.__packet_queues:
                    self.__packet_queues[metadata.stream_id].append(packet)
            except zmq.Again:
                print("Receive timed out, retrying...")
                self.__logger.debug("Receive timed out, retrying...")
                continue
            except Exception as e:
                self.__logger.error(f"Receive Failed: {e}")
                print(f"Receive Failed: {e}")
                self.__running = False
                break
