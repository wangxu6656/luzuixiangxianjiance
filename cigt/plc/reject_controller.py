import time
from typing import Optional


class RejectController:
    def __init__(
        self,
        enabled: bool = False,
        mode: str = "simulate",
        address: str = "127.0.0.1",
        port: int = 5000,
        serial_port: str = "COM3",
        baudrate: int = 9600,
        command_prefix: str = "REJECT",
    ):
        self.enabled = enabled
        self.mode = mode
        self.address = address
        self.port = int(port)
        self.serial_port = serial_port
        self.baudrate = int(baudrate)
        self.command_prefix = command_prefix
        self._conn = None
        self._last_send = 0.0
        self._min_interval = 0.05
        self._open()

    def _open(self) -> None:
        if not self.enabled:
            return
        try:
            if self.mode == "serial":
                import serial  # type: ignore

                self._conn = serial.Serial(self.serial_port, self.baudrate, timeout=0.1)
            elif self.mode == "udp":
                import socket

                self._conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif self.mode == "tcp":
                import socket

                self._conn = socket.create_connection((self.address, self.port), timeout=1.0)
        except Exception:
            self._conn = None

    def _build_payload(self, reject_id: Optional[str]) -> bytes:
        rid = reject_id or str(int(time.time() * 1000))
        return f"{self.command_prefix} {rid}\n".encode("ascii")

    def send_reject(self, result, reject_id: Optional[str] = None) -> None:
        if not self.enabled or self._conn is None:
            return
        if not result.reject:
            return
        now = time.time()
        if now - self._last_send < self._min_interval:
            return
        self._last_send = now
        payload = self._build_payload(reject_id)
        try:
            if self.mode == "serial":
                self._conn.write(payload)
            elif self.mode == "udp":
                import socket

                self._conn.sendto(payload, (self.address, self.port))
            elif self.mode == "tcp":
                self._conn.sendall(payload)
        except Exception:
            pass

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass