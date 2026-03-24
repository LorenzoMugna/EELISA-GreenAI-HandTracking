"""Configuration loader for UDP module."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UDPConfig:
    """Configuration for UDP communication."""
    sender_ip: str
    sender_port: int
    receiver_ip: str
    receiver_port: int
    buffer_size: int
    timeout: int
    num_spike_queues: int
    spike_window_ms: int
    cleanup_interval_ms: int


def load_config(config_path: str = None) -> UDPConfig:
    """Load configuration from config.config file.

    Args:
        config_path: Path to config file. If None, uses config.config in same directory.

    Returns:
        UDPConfig object with loaded settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.config"

    config_dict = {}

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config_dict[key.strip()] = value.strip()

    return UDPConfig(
        sender_ip=config_dict.get("SENDER_IP", "127.0.0.1"),
        sender_port=int(config_dict.get("SENDER_PORT", 5005)),
        receiver_ip=config_dict.get("RECEIVER_IP", "127.0.0.1"),
        receiver_port=int(config_dict.get("RECEIVER_PORT", 5006)),
        buffer_size=int(config_dict.get("BUFFER_SIZE", 65535)),
        timeout=int(config_dict.get("TIMEOUT", 5)),
        num_spike_queues=int(config_dict.get("NUM_SPIKE_QUEUES", 5)),
        spike_window_ms=int(config_dict.get("SPIKE_WINDOW_MS", 1000)),
        cleanup_interval_ms=int(config_dict.get("CLEANUP_INTERVAL_MS", 100)),
    )


_config: UDPConfig = None


def get_config() -> UDPConfig:
    """Get the loaded configuration (singleton pattern)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
