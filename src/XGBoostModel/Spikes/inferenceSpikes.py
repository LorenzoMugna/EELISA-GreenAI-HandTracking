import joblib
import xgboost as xgb
import numpy as np
import joblib

import sys
import time
import argparse

sys.path.insert(0, "../..")
import udp


def inference():
    config = udp.get_config()

    # Create data store
    data_store = udp.DataStore(
        num_spike_queues=config.num_spike_queues,
        spike_window_ms=config.spike_window_ms
    )

    # Create and start receiver
    def on_message(msg):
        a = 1#print(f"  Received: {msg}")

    receiver = udp.UDPReceiver(data_store, on_message=on_message)
    receiver.start()
    print("Receiver started. Press Ctrl+C to stop.\n")

    # 1. Initialize an empty Booster
    loaded_model = xgb.Booster()

    # 2. Load the saved weights and configuration
    loaded_model.load_model("PredictionModelWithSpike_NoRot.json")

    # 2b. Load the scaler used during training
    scaler = joblib.load("scalerWithSpike_NoRot.pkl")   

    try:
        while True:
            # DO SOMETHING WITH THE DATA (e.g. inference)
            time.sleep(1)
            # print("\n--- Current State ---")
            # print(f"Spike counts: {data_store.get_all_spike_counts()}")
            # print(f"Firing rates: {data_store.get_all_firing_rates()}")
            # print(f"IAT variances: {data_store.get_all_inter_arrival_variances()}")
            # print(f"Values: {data_store.get_all_values()}")
            # print(f"Coordinates: {data_store.get_all_coordinates()}")
            # model = data_store.get_model()
            # print(f"Model: {type(model).__name__ if model else 'None'}")
            # print(f"\nAll features: {data_store.get_all_features()}")
            # print(f"Bytes received: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")

            # INFERENCE
            # 3. Make predictions (Assuming X_new_gpu is a CuPy array of new data)
            # Note: Native XGBoost expects a DMatrix for prediction
            #labels: ['ch0_rate_hz', 'ch1_rate_hz', 'ch2_rate_hz', 'ch3_rate_hz', 'ch4_rate_hz', 'ch5_rate_hz', 'ch0_var_isi_ms', 'ch1_var_isi_ms', 'ch2_var_isi_ms', 'ch3_var_isi_ms', 'ch4_var_isi_ms', 'ch5_var_isi_ms']
            Data = data_store.get_spikes_features()
            Data = np.array(Data, dtype=np.float32).reshape(1, -1)
            Data = scaler.transform(Data)
            dnew = xgb.DMatrix(Data)
            prediction = loaded_model.predict(dnew)
            index_max_pred = np.argmax(prediction)
            print(f"Input data: {Data.flatten().tolist()}")
            print(f"Predicted class: {index_max_pred}, Probabilities: {prediction}")
            print(prediction)

            
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        receiver.stop()
        print(f"Total data received: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
        print("Receiver stopped.")


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

    inference()


if __name__ == "__main__":
    main()
