"""UDP module for EELISA GreenAI Hand Tracking.

This module provides simple UDP communication for sending and receiving
feature data for inference. The user can simply import this module and
use functions like `udp.send()` without worrying about addresses.

Sender Usage:
    import udp

    # Send raw message
    udp.send("Hello")

    # Send a spike event
    udp.send_spike(0)

    # Send a coordinate
    udp.send_coordinate("palm", 1.0, 2.0, 3.0)

    # Send a value
    udp.send_value("feature_1", 42.5)

    # Send a sklearn model
    udp.send_model(my_model)

Receiver Usage:
    import udp

    # Create a data store (you own this, pass it to receiver)
    data_store = udp.DataStore(num_spike_queues=5, spike_window_ms=1000)

    # Create and start receiver
    receiver = udp.UDPReceiver(data_store)
    receiver.start()

    # Access data from your code
    spike_counts = data_store.get_all_spike_counts()
    palm_coords = data_store.get_coordinate("palm")
    feature_value = data_store.get_value("feature_1")

    # Stop when done
    receiver.stop()
"""

from .config import UDPConfig, load_config, get_config
from .sender import (
    UDPSender,
    get_sender,
    send,
    send_spike,
    send_model,
    send_coordinate,
    send_value,
)
from .receiver import (
    DataStore,
    SpikeCleanupThread,
    UDPReceiver,
)

__all__ = [
    # Config
    "UDPConfig",
    "load_config",
    "get_config",
    # Sender
    "UDPSender",
    "get_sender",
    "send",
    "send_spike",
    "send_model",
    "send_coordinate",
    "send_value",
    # Receiver
    "DataStore",
    "SpikeCleanupThread",
    "UDPReceiver",
]
