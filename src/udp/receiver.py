"""UDP Receiver module with data storage for inference."""

import json
import pickle
import base64
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import get_config, UDPConfig


class DataStore:
    """Thread-safe data storage for received UDP data.

    This class stores:
    - Spike queues: timestamps of spikes per ID, automatically cleaned
    - Last values: most recent value for each feature ID
    - Coordinates: most recent coordinate for each name
    - Model: most recently received sklearn model

    The user creates this object and passes it to the UDPReceiver.
    This allows the user to access the data from their code.
    """

    def __init__(self, num_spike_queues: int = 5, spike_window_ms: int = 1000):
        """Initialize the data store.

        Args:
            num_spike_queues: Number of spike queues (N).
            spike_window_ms: Time window in milliseconds (W) for keeping spikes.
        """
        self._num_spike_queues = num_spike_queues
        self._spike_window_ms = spike_window_ms

        self._spike_queues: List[deque] = [deque() for _ in range(num_spike_queues)]
        self._spike_lock = threading.Lock()

        self._values: Dict[str, Any] = {}
        self._values_lock = threading.Lock()

        self._coordinates: Dict[str, Dict[str, float]] = {}
        self._coordinates_lock = threading.Lock()

        self._model: Any = None
        self._model_lock = threading.Lock()

    @property
    def num_spike_queues(self) -> int:
        return self._num_spike_queues

    @property
    def spike_window_ms(self) -> int:
        return self._spike_window_ms

    def add_spike(self, spike_id: int, timestamp_ms: float = None) -> None:
        """Add a spike timestamp to the specified queue.

        Args:
            spike_id: The spike ID (0 to N-1).
            timestamp_ms: Timestamp in milliseconds. If None, uses current time.
        """
        if spike_id < 0 or spike_id >= self._num_spike_queues:
            return

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000

        with self._spike_lock:
            self._spike_queues[spike_id].append(timestamp_ms)

    def get_spike_count(self, spike_id: int) -> int:
        """Get the number of spikes in the specified queue within the time window.

        Args:
            spike_id: The spike ID (0 to N-1).

        Returns:
            Number of spikes in the time window.
        """
        if spike_id < 0 or spike_id >= self._num_spike_queues:
            return 0

        with self._spike_lock:
            return len(self._spike_queues[spike_id])

    def get_spike_timestamps(self, spike_id: int) -> List[float]:
        """Get all spike timestamps for the specified queue.

        Args:
            spike_id: The spike ID (0 to N-1).

        Returns:
            List of timestamps in the time window.
        """
        if spike_id < 0 or spike_id >= self._num_spike_queues:
            return []

        with self._spike_lock:
            return list(self._spike_queues[spike_id])

    def get_all_spike_counts(self) -> List[int]:
        """Get spike counts for all queues.

        Returns:
            List of spike counts for each queue.
        """
        with self._spike_lock:
            return [len(q) for q in self._spike_queues]

    def cleanup_old_spikes(self) -> None:
        """Remove spikes older than the time window from all queues."""
        cutoff = time.time() * 1000 - self._spike_window_ms

        with self._spike_lock:
            for queue in self._spike_queues:
                while queue and queue[0] < cutoff:
                    queue.popleft()

    def set_value(self, value_id: str, value: Any) -> None:
        """Set a feature value.

        Args:
            value_id: The feature identifier.
            value: The value to store.
        """
        with self._values_lock:
            self._values[value_id] = value

    def get_value(self, value_id: str, default: Any = None) -> Any:
        """Get a feature value.

        Args:
            value_id: The feature identifier.
            default: Default value if not found.

        Returns:
            The stored value or default.
        """
        with self._values_lock:
            return self._values.get(value_id, default)

    def get_all_values(self) -> Dict[str, Any]:
        """Get all stored feature values.

        Returns:
            Dictionary of all feature values.
        """
        with self._values_lock:
            return dict(self._values)

    def set_coordinate(self, name: str, x: float, y: float, z: float) -> None:
        """Set a coordinate value.

        Args:
            name: The coordinate name.
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        with self._coordinates_lock:
            self._coordinates[name] = {"x": x, "y": y, "z": z}

    def get_coordinate(self, name: str) -> Optional[Dict[str, float]]:
        """Get a coordinate value.

        Args:
            name: The coordinate name.

        Returns:
            Dictionary with x, y, z or None if not found.
        """
        with self._coordinates_lock:
            return self._coordinates.get(name)

    def get_all_coordinates(self) -> Dict[str, Dict[str, float]]:
        """Get all stored coordinates.

        Returns:
            Dictionary of all coordinates.
        """
        with self._coordinates_lock:
            return dict(self._coordinates)

    def set_model(self, model: Any) -> None:
        """Set the received model.

        Args:
            model: The sklearn model.
        """
        with self._model_lock:
            self._model = model

    def get_model(self) -> Any:
        """Get the stored model.

        Returns:
            The sklearn model or None.
        """
        with self._model_lock:
            return self._model


class SpikeCleanupThread(threading.Thread):
    """Background thread that periodically cleans old spikes from the data store."""

    def __init__(self, data_store: DataStore, interval_ms: int = 100):
        """Initialize the cleanup thread.

        Args:
            data_store: The DataStore to clean.
            interval_ms: Cleanup interval in milliseconds.
        """
        super().__init__(daemon=True)
        self._data_store = data_store
        self._interval_s = interval_ms / 1000
        self._running = False

    def run(self) -> None:
        """Run the cleanup loop."""
        self._running = True
        while self._running:
            self._data_store.cleanup_old_spikes()
            time.sleep(self._interval_s)

    def stop(self) -> None:
        """Stop the cleanup thread."""
        self._running = False


class UDPReceiver:
    """UDP Receiver that stores incoming data in a DataStore."""

    def __init__(self, data_store: DataStore, on_message: Callable[[dict], None] = None):
        """Initialize the UDP receiver.

        Args:
            data_store: DataStore object where received data will be stored.
            on_message: Optional callback called for each received message.
        """
        self._config = get_config()
        self._data_store = data_store
        self._on_message = on_message

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._config.receiver_ip, self._config.receiver_port))
        self._socket.settimeout(1.0)

        self._running = False
        self._receive_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[SpikeCleanupThread] = None

    def start(self) -> None:
        """Start the receiver in a background thread."""
        if self._running:
            return

        self._running = True

        self._cleanup_thread = SpikeCleanupThread(
            self._data_store,
            self._config.cleanup_interval_ms
        )
        self._cleanup_thread.start()

        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

    def stop(self) -> None:
        """Stop the receiver."""
        self._running = False

        if self._cleanup_thread:
            self._cleanup_thread.stop()
            self._cleanup_thread = None

        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None

    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self._config.buffer_size)
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
