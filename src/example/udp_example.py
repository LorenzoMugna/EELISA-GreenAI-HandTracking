"""Example script to test UDP send/receive functionality.

This script demonstrates both sender and receiver functionality.
Run with --mode sender or --mode receiver on separate machines/terminals.
For local testing, use --mode local to run both in the same process.
"""

import sys
import time
import argparse

sys.path.insert(0, "..")
import udp


def run_sender():
    """Demonstrate sender functionality."""
    print("=== UDP Sender Example ===")
    print(f"Sending to {udp.get_config().receiver_ip}:{udp.get_config().receiver_port}")
    print()

    # Send some spikes
    print("Sending spikes...")
    for i in range(5):
        for spike_id in range(3):
            udp.send_spike(spike_id)
            print(f"  Sent spike id={spike_id}")
        time.sleep(0.1)

    # Send coordinates
    print("\nSending coordinates...")
    udp.send_coordinate("palm", 100.5, 200.3, 50.0)
    print("  Sent palm coordinate")
    udp.send_coordinate("thumb_tip", 120.0, 180.0, 45.0)
    print("  Sent thumb_tip coordinate")

    # Send values
    print("\nSending values...")
    udp.send_value("distance", 42.5)
    print("  Sent distance=42.5")
    udp.send_value("angle", 90.0)
    print("  Sent angle=90.0")

    # Send a simple model (using a mock for demo)
    print("\nSending model...")
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np

        model = LinearRegression()
        model.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))
        udp.send_model(model)
        print("  Sent sklearn LinearRegression model")
    except ImportError:
        print("  sklearn not available, skipping model send")

    print("\nSender done!")


def run_receiver():
    """Demonstrate receiver functionality."""
    print("=== UDP Receiver Example ===")
    config = udp.get_config()
    print(f"Listening on {config.receiver_ip}:{config.receiver_port}")
    print(f"Spike queues: {config.num_spike_queues}")
    print(f"Spike window: {config.spike_window_ms}ms")
    print()

    # Create data store
    data_store = udp.DataStore(
        num_spike_queues=config.num_spike_queues,
        spike_window_ms=config.spike_window_ms
    )

    # Create and start receiver
    def on_message(msg):
        print(f"  Received: {msg}")

    receiver = udp.UDPReceiver(data_store, on_message=on_message)
    receiver.start()
    print("Receiver started. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            print("\n--- Current State ---")
            print(f"Spike counts: {data_store.get_all_spike_counts()}")
            print(f"Values: {data_store.get_all_values()}")
            print(f"Coordinates: {data_store.get_all_coordinates()}")
            model = data_store.get_model()
            print(f"Model: {type(model).__name__ if model else 'None'}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        receiver.stop()
        print("Receiver stopped.")


def run_local_test():
    """Run both sender and receiver locally for testing."""
    print("=== Local UDP Test ===")
    print("Testing with loopback (127.0.0.1)")
    print()

    # Override config for local testing
    udp.config._config = udp.UDPConfig(
        sender_ip="127.0.0.1",
        sender_port=5005,
        receiver_ip="127.0.0.1",
        receiver_port=5005,  # Same port for local test
        buffer_size=65535,
        timeout=5,
        num_spike_queues=5,
        spike_window_ms=1000,
        cleanup_interval_ms=100,
    )

    # Create data store
    data_store = udp.DataStore(num_spike_queues=5, spike_window_ms=1000)

    # Start receiver
    receiver = udp.UDPReceiver(data_store)
    receiver.start()
    print("Receiver started")
    time.sleep(0.1)

    # Send test data
    print("\nSending test data...")

    # Send spikes
    for i in range(10):
        udp.send_spike(i % 5)
    print(f"  Sent 10 spikes")

    # Send coordinates
    udp.send_coordinate("palm", 1.0, 2.0, 3.0)
    udp.send_coordinate("index_tip", 4.0, 5.0, 6.0)
    print(f"  Sent 2 coordinates")

    # Send values
    udp.send_value("feature_a", 100)
    udp.send_value("feature_b", 200)
    print(f"  Sent 2 values")

    # Wait for processing
    time.sleep(0.2)

    # Check results
    print("\n=== Results ===")
    print(f"Spike counts: {data_store.get_all_spike_counts()}")
    print(f"Values: {data_store.get_all_values()}")
    print(f"Coordinates: {data_store.get_all_coordinates()}")

    # Test spike window cleanup
    print("\nWaiting 1.5s to test spike cleanup...")
    time.sleep(1.5)
    print(f"Spike counts after cleanup: {data_store.get_all_spike_counts()}")

    # Cleanup
    receiver.stop()
    print("\nTest complete!")


def main():
    parser = argparse.ArgumentParser(description="UDP module example")
    parser.add_argument(
        "--mode",
        choices=["sender", "receiver", "local"],
        default="local",
        help="Run as sender, receiver, or local test (default: local)"
    )
    args = parser.parse_args()

    if args.mode == "sender":
        run_sender()
    elif args.mode == "receiver":
        run_receiver()
    else:
        run_local_test()


if __name__ == "__main__":
    main()
