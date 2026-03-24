"""UDP Receiver module for receiving and processing UDP packets."""

import json
import pickle
import base64
import socket
import threading
from typing import Any, Callable, Optional

from .config import get_config
from .datastore import DataStore


class UDPReceiver:
    """UDP Receiver that stores incoming data in a DataStore."""

    def __init__(
        self,
        data_store: DataStore,
        on_message: Callable[[dict], None] = None,
        ip: str = None,
        port: int = None,
    ):
        """Initialize the UDP receiver.

        Args:
            data_store: DataStore object where received data will be stored.
            on_message: Optional callback called for each received message.
            ip: IP address to bind to. If None, uses config value.
            port: Port to bind to. If None, uses config value.
        """
        self._config = get_config()
        self._data_store = data_store
        self._on_message = on_message

        # Use provided IP/port or fall back to config
        self._ip = ip if ip is not None else self._config.receiver_ip
        self._port = port if port is not None else self._config.receiver_port

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._ip, self._port))
        self._socket.settimeout(1.0)

        self._running = False
        self._receive_thread: Optional[threading.Thread] = None

    @property
    def ip(self) -> str:
        """Get the bound IP address."""
        return self._ip

    @property
    def port(self) -> int:
        """Get the bound port."""
        return self._port

    def start(self) -> None:
        """Start the receiver in a background thread."""
        if self._running:
            return

        self._running = True

        # Ensure data store cleanup is running
        self._data_store.start_cleanup()

        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

    def stop(self) -> None:
        """Stop the receiver."""
        self._running = False

        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None

    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self._config.buffer_size)
                self._data_store.add_bytes(len(data))
                self._process_message(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"UDP Receiver error: {e}")

    def _process_message(self, data: bytes) -> None:
        """Process a received UDP message.

        Args:
            data: Raw bytes received.
        """
        try:
            message = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        msg_type = message.get("type")

        if msg_type == "spike":
            spike_id = message.get("id")
            if isinstance(spike_id, int):
                self._data_store.add_spike(spike_id)

        elif msg_type == "value":
            value_id = message.get("id")
            value = message.get("value")
            if value_id is not None:
                self._data_store.set_value(value_id, value)

        elif msg_type == "coordinate":
            name = message.get("name")
            x = message.get("x")
            y = message.get("y")
            z = message.get("z")
            if name is not None and all(v is not None for v in (x, y, z)):
                self._data_store.set_coordinate(name, x, y, z)

        elif msg_type == "model":
            model_b64 = message.get("data")
            if model_b64:
                try:
                    model_bytes = base64.b64decode(model_b64)
                    model = pickle.loads(model_bytes)
                    self._data_store.set_model(model)
                except Exception:
                    pass

        if self._on_message:
            self._on_message(message)

    def close(self) -> None:
        """Stop the receiver and close the socket."""
        self.stop()
        self._socket.close()

    @property
    def data_store(self) -> DataStore:
        """Get the data store."""
        return self._data_store
