"""DataStore module for storing and managing received UDP data.

This module is independent from UDP logic and can be used standalone.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class SpikeCleanupThread(threading.Thread):
    """Background thread that periodically cleans old spikes from the data store."""

    def __init__(self, data_store: "DataStore", interval_ms: int = 100):
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

    def __init__(
        self,
        num_spike_queues: int = 5,
        spike_window_ms: int = 1000,
        cleanup_interval_ms: int = 100,
        auto_cleanup: bool = True,
    ):
        """Initialize the data store.

        Args:
            num_spike_queues: Number of spike queues (N).
            spike_window_ms: Time window in milliseconds (W) for keeping spikes.
            cleanup_interval_ms: Interval for cleanup thread in milliseconds.
            auto_cleanup: If True, automatically start cleanup thread.
        """
        self._num_spike_queues = num_spike_queues
        self._spike_window_ms = spike_window_ms
        self._cleanup_interval_ms = cleanup_interval_ms

        self._spike_queues: List[deque] = [deque() for _ in range(num_spike_queues)]
        self._spike_lock = threading.Lock()

        self._values: Dict[str, Any] = {}
        self._values_lock = threading.Lock()

        self._coordinates: Dict[str, Dict[str, float]] = {}
        self._coordinates_lock = threading.Lock()

        self._model: Any = None
        self._model_lock = threading.Lock()

        self._cleanup_thread: Optional[SpikeCleanupThread] = None
        if auto_cleanup:
            self.start_cleanup()

    @property
    def num_spike_queues(self) -> int:
        return self._num_spike_queues

    @property
    def spike_window_ms(self) -> int:
        return self._spike_window_ms

    @property
    def spike_window_s(self) -> float:
        """Get spike window in seconds."""
        return self._spike_window_ms / 1000.0

    def start_cleanup(self) -> None:
        """Start the cleanup thread if not already running."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = SpikeCleanupThread(self, self._cleanup_interval_ms)
            self._cleanup_thread.start()

    def stop_cleanup(self) -> None:
        """Stop the cleanup thread."""
        if self._cleanup_thread:
            self._cleanup_thread.stop()
            self._cleanup_thread = None

    # =========================================================================
    # Spike methods
    # =========================================================================

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

    def get_firing_rate(self, spike_id: int) -> float:
        """Get the firing rate for a spike ID.

        Firing rate = number of spikes in window / window time in seconds.

        Args:
            spike_id: The spike ID (0 to N-1).

        Returns:
            Firing rate in Hz (spikes per second).
        """
        count = self.get_spike_count(spike_id)
        return count / self.spike_window_s

    def get_all_firing_rates(self) -> List[float]:
        """Get firing rates for all spike queues.

        Returns:
            List of firing rates in Hz for each queue.
        """
        counts = self.get_all_spike_counts()
        return [c / self.spike_window_s for c in counts]

    def get_inter_arrival_variance(self, spike_id: int) -> Optional[float]:
        """Get the variance of inter-arrival times for a spike ID.

        Inter-arrival time is the difference between subsequent spike timestamps.

        Args:
            spike_id: The spike ID (0 to N-1).

        Returns:
            Variance of inter-arrival times in ms^2, or None if < 2 spikes.
        """
        timestamps = self.get_spike_timestamps(spike_id)

        if len(timestamps) < 2:
            return None

        # Calculate inter-arrival times
        inter_arrivals = []
        for i in range(1, len(timestamps)):
            inter_arrivals.append(timestamps[i] - timestamps[i - 1])

        if len(inter_arrivals) < 1:
            return None

        # Calculate variance
        mean = sum(inter_arrivals) / len(inter_arrivals)
        variance = sum((x - mean) ** 2 for x in inter_arrivals) / len(inter_arrivals)
        return variance

    def get_all_inter_arrival_variances(self) -> List[Optional[float]]:
        """Get inter-arrival variances for all spike queues.

        Returns:
            List of variances (or None) for each queue.
        """
        return [self.get_inter_arrival_variance(i) for i in range(self._num_spike_queues)]

    # =========================================================================
    # Value methods
    # =========================================================================

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

    # =========================================================================
    # Coordinate methods
    # =========================================================================

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

    # =========================================================================
    # Model methods
    # =========================================================================

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

    # =========================================================================
    # Aggregate methods
    # =========================================================================

    def get_all_features(self) -> Dict[str, Any]:
        """Get all features as a single dictionary for inference.

        Returns a dictionary containing:
        - All stored values (key: value_id)
        - All coordinates flattened (key: "{name}_x", "{name}_y", "{name}_z")
        - Spike firing rates (key: "spike_{id}_rate")
        - Spike inter-arrival variances (key: "spike_{id}_variance")

        Returns:
            Dictionary with all features.
        """
        features = {}

        # Add all values
        features.update(self.get_all_values())

        # Add coordinates (flattened)
        for name, coord in self.get_all_coordinates().items():
            features[f"{name}_x"] = coord["x"]
            features[f"{name}_y"] = coord["y"]
            features[f"{name}_z"] = coord["z"]

        # Add spike features
        for i in range(self._num_spike_queues):
            features[f"spike_{i}_rate"] = self.get_firing_rate(i)
            features[f"spike_{i}_variance"] = self.get_inter_arrival_variance(i)

        return features
