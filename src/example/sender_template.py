"""Template script to send UDP messages."""

import sys
import argparse

# Needed only when sending the model
from sklearn.linear_model import LinearRegression
import numpy as np

sys.path.insert(0, "..")
import udp


def whatever_function():
    spike_id = 0
    udp.send_spike(spike_id)
    udp.send_coordinate("palm", 100.5, 200.3, 50.0)
    udp.send_value("distance", 42.5)
        
    model = LinearRegression()
    model.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))
    udp.send_model(model)
    

def main():
    parser = argparse.ArgumentParser(description="UDP module example")
    parser.add_argument(
        "--ip",
        default=None,
        help="Override IP address for sender/receiver"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port for sender/receiver"
    )
    args = parser.parse_args()

    # Apply IP/port overrides if provided
    if args.ip or args.port:
        udp.configure(ip=args.ip, port=args.port)
        print(f"Configured sender target: {args.ip or 'default'}:{args.port or 'default'}")

    whatever_function()


if __name__ == "__main__":
    main()
