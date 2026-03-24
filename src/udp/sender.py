"""UDP Sender module with utility functions."""

import json
import pickle
import socket
import base64
from typing import Any, Union

from .config import get_config


class UDPSender:
    """UDP Sender that handles all outgoing UDP packets."""

    def __init__(self):
        self._config = get_config()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (self._config.receiver_ip, self._config.receiver_port)

    def send(self, message: Union[str, bytes]) -> None:
        """Send a raw message via UDP.

        Args:
            message: String or bytes to send.
        """
        if isinstance(message, str):
            message = message.encode("utf-8")
        self._socket.sendto(message, self._target)

    def send_json(self, data: dict) -> None:
        """Send a JSON-encoded message via UDP.

        Args:
            data: Dictionary to send as JSON.
        """
        message = json.dumps(data)
        self.send(message)

    def send_spike(self, spike_id: int) -> None:
        """Send a spike event with the given ID.

        Args:
            spike_id: The ID of the spike (0 to N-1).
        """
        data = {
            "type": "spike",
            "id": spike_id
        }
        self.send_json(data)

    def send_model(self, model: Any) -> None:
        """Send a sklearn model via UDP.

        The model is serialized with pickle and base64-encoded for transmission.

        Args:
            model: A sklearn model object to send.
        """
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode("utf-8")
        data = {
            "type": "model",
            "data": model_b64
        }
        self.send_json(data)

    def send_coordinate(self, name: str, x: float, y: float, z: float) -> None:
        """Send a named coordinate (x, y, z) via UDP.

        Args:
            name: Name/identifier for the coordinate.
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        data = {
            "type": "coordinate",
            "name": name,
            "x": x,
            "y": y,
            "z": z
        }
        self.send_json(data)

    def send_value(self, value_id: str, value: Any) -> None:
        """Send a named value via UDP.

        Args:
            value_id: Identifier for the value.
            value: The value to send.
        """
        data = {
            "type": "value",
            "id": value_id,
            "value": value
        }
        self.send_json(data)

    def close(self) -> None:
        """Close the UDP socket."""
        self._socket.close()


_sender: UDPSender = None


def get_sender() -> UDPSender:
    """Get the global UDP sender instance (singleton)."""
    global _sender
    if _sender is None:
        _sender = UDPSender()
    return _sender


def send(message: Union[str, bytes]) -> None:
    """Send a raw message via UDP."""
    get_sender().send(message)


def send_spike(spike_id: int) -> None:
    """Send a spike event with the given ID."""
    get_sender().send_spike(spike_id)


def send_model(model: Any) -> None:
    """Send a sklearn model via UDP."""
    get_sender().send_model(model)


def send_coordinate(name: str, x: float, y: float, z: float) -> None:
    """Send a named coordinate (x, y, z) via UDP."""
    get_sender().send_coordinate(name, x, y, z)


def send_value(value_id: str, value: Any) -> None:
    """Send a named value via UDP."""
    get_sender().send_value(value_id, value)
