import threading
import logging
import zmq


class ZmqProxy:
    def __init__(self, pub_port="tcp://*:7201", sub_port="tcp://*:7502"):
        self.__logger = logging.getLogger(__name__)
        self.__pub_port = pub_port
        self.__sub_port = sub_port
        self.__running = False

    def start(self):
        self.__running = True
        threading.Thread(target=self._run_proxy, daemon=True).start()

    @property
    def running(self):
        return self.__running

    def _run_proxy(self):
        print(f"[Proxy] Starting ZeroMQ Proxy...")
        context = zmq.Context.instance()

        xsub = context.socket(zmq.XSUB)
        xpub = context.socket(zmq.XPUB)

        xsub.bind(self.__sub_port)
        print(f"[Proxy] XSUB bound at {self.__sub_port}")

        xpub.bind(self.__pub_port)
        print(f"[Proxy] XPUB bound at {self.__pub_port}")

        try:
            zmq.proxy(xsub, xpub)
        except Exception as e:
            self.__running = False
            print(f"[Proxy] Error: {e}")

        print("[Proxy] Shutting down...")
        xsub.close()
        xpub.close()
