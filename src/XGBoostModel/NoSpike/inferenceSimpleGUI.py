"""Inference script with simple matplotlib GUI for real-time visualization."""

# Set matplotlib backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend

import numpy as np
import sys
import time
import argparse
import threading
from pathlib import Path

# Try to import ML dependencies, fall back gracefully if not available
try:
    import joblib
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ML dependencies not available: {e}")
    print("🔄 Will use mock predictor instead")
    ML_AVAILABLE = False

sys.path.insert(0, "../..")
import udp

# Add GUI path to system path
sys.path.insert(0, "../../GUI")
from simple_gui import SimpleHandTrackingGUI, create_model_predictor

# Add common utilities
sys.path.insert(0, "../../common")
from model_utils import load_model_and_scaler, Config


def inference_with_simple_gui():
    """Run inference with simple matplotlib GUI visualization."""
    print("🚀 Starting Hand Tracking Inference with GUI")

    try:
        config = udp.get_config()
        print(f"✅ Configuration loaded: {config.num_spike_queues} spike queues, {config.spike_window_ms}ms window")

        # Create data store
        data_store = udp.DataStore(
            num_spike_queues=config.num_spike_queues,
            spike_window_ms=config.spike_window_ms
        )
        print("✅ DataStore created")

        # Load model and scaler using shared utility
        print("🔧 Loading model and scaler...")
        if ML_AVAILABLE:
            loaded_model, scaler = load_model_and_scaler(
                Config.DEFAULT_MODEL_PATH,
                Config.DEFAULT_SCALER_PATH
            )
        else:
            print("⚠️  ML dependencies not available, skipping model loading")
            loaded_model, scaler = None, None

        # Create model predictor
        print("🤖 Creating predictor...")
        if loaded_model and scaler:
            predictor = create_model_predictor(loaded_model, scaler)
            print("✅ Using real XGBoost model predictor")
        else:
            # Mock predictor for testing
            def mock_predictor(data):
                prediction = int(time.time() * 0.5) % 7  # Cycling predictions like test
                probabilities = np.random.dirichlet(np.ones(7))
                probabilities[prediction] *= 3  # Boost predicted class probability
                probabilities = probabilities / probabilities.sum()
                return prediction, probabilities
            predictor = mock_predictor
            print("⚠️  Using mock predictor (model/scaler not loaded)")

        print(f"📊 Predictor ready: {type(predictor)}")

        # Setup image folder path
        img_folder = Path("../../GUI/img")
        print(f"🖼️  Looking for images in: {img_folder.absolute()}")

        # Create GUI BEFORE starting UDP receiver to avoid threading conflicts
        print("🎨 Creating GUI...")

        # Ensure matplotlib is ready
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode

        gui = SimpleHandTrackingGUI(
            data_store=data_store,
            model_predictor=predictor,
            image_folder=img_folder
        )
        print("✅ GUI created successfully")

        # Start some test data simulation to make GUI visible immediately
        def simulate_test_data():
            """Add some initial test data to make GUI show up with content."""
            for i in range(5):
                data_store.set_value(f"digit_{i}_distance", 10.0 + i * 2)
            data_store.set_value("palm_normal_y", 2.5)

            # Add some spikes and bytes
            for i in range(3):
                data_store.add_spike(i)
            data_store.add_bytes(100, is_spike=True)
            data_store.add_bytes(300, is_spike=False)
            print("✅ Added initial test data for GUI display")

        simulate_test_data()

        # Create and start receiver AFTER GUI is ready
        def on_message(msg):
            pass  # We don't need to print messages in GUI mode

        receiver = udp.UDPReceiver(data_store, on_message=on_message)
        receiver.start()
        print("✅ UDP Receiver started")

        print("\n🚀 The GUI window will show:")
        print("  📊 Top left: Current gesture prediction with confidence scores")
        print("  🖼️  Gesture images (if available)")
        print("  📈 Top middle: Real-time spike timeline visualization")
        print("  📋 Top right: Live UDP data overview")
        print("  📊 Middle: UDP bytes comparison (spikes vs no-spikes)")
        print("  📝 Bottom: Detailed DataStore information")
        print("\n❌ To stop: Close the GUI window or press Ctrl+C")
        print("⏳ Opening GUI window...")

        # Start GUI (this will block until window is closed)
        gui.start()

    except Exception as e:
        print(f"❌ Error in GUI setup: {e}")
        import traceback
        traceback.print_exc()
        return
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        if 'gui' in locals():
            print("🛑 Stopping GUI...")
            gui.stop()
        if 'receiver' in locals():
            print("🛑 Stopping receiver...")
            receiver.stop()
        if 'data_store' in locals():
            print(f"📊 Final stats:")
            print(f"  Total data: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
            print(f"  Spike data: {data_store.get_bytes_spikes_formatted()} ({data_store.get_bytes_spikes()} bytes)")
            print(f"  No-spike data: {data_store.get_bytes_no_spikes_formatted()} ({data_store.get_bytes_no_spikes()} bytes)")
            if data_store.get_bytes_received() > 0:
                spike_ratio = (data_store.get_bytes_spikes() / data_store.get_bytes_received()) * 100
                print(f"  Spike data ratio: {spike_ratio:.1f}%")
        print("✅ Application stopped.")


def inference_cli():
    """Run inference without GUI (command line mode)."""
    if not ML_AVAILABLE:
        print("❌ ML dependencies (joblib, xgboost) required for CLI mode")
        print("🔄 Please install them with: pip install joblib xgboost")
        return

    config = udp.get_config()

    # Create data store
    data_store = udp.DataStore(
        num_spike_queues=config.num_spike_queues,
        spike_window_ms=config.spike_window_ms
    )

    # Create and start receiver
    def on_message(msg):
        a = 1  # Minimal message handler

    receiver = udp.UDPReceiver(data_store, on_message=on_message)
    receiver.start()
    print("Receiver started. Press Ctrl+C to stop.\n")

    # Load model and scaler
    try:
        loaded_model = xgb.Booster()
        loaded_model.load_model("PredictionModelNoSpike.json")

        scaler = joblib.load("scalerNoSpike.pkl")
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        return

    try:
        while True:
            # DO SOMETHING WITH THE DATA (e.g. inference)
            time.sleep(1)
            print("\n--- Current State ---")
            print(f"Spike counts: {data_store.get_all_spike_counts()}")
            print(f"Firing rates: {data_store.get_all_firing_rates()}")
            print(f"IAT variances: {data_store.get_all_inter_arrival_variances()}")
            print(f"Values: {data_store.get_all_values()}")
            print(f"Coordinates: {data_store.get_all_coordinates()}")
            model = data_store.get_model()
            print(f"Model: {type(model).__name__ if model else 'None'}")
            print(f"\nAll features: {data_store.get_all_features()}")
            print(f"Bytes received: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
            print(f"Spike bytes: {data_store.get_bytes_spikes_formatted()} ({data_store.get_bytes_spikes()} bytes)")
            print(f"No-spike bytes: {data_store.get_bytes_no_spikes_formatted()} ({data_store.get_bytes_no_spikes()} bytes)")

            # INFERENCE
            # 3. Make predictions (Assuming X_new_gpu is a CuPy array of new data)
            # Note: Native XGBoost expects a DMatrix for prediction
            Data = data_store.get_digit_distances()
            Data = np.array(Data, dtype=np.float32).reshape(1, -1)
            Data = scaler.transform(Data)
            dnew = xgb.DMatrix(Data)
            prediction = loaded_model.predict(dnew)
            index_max_pred = np.argmax(prediction)
            confidence = np.max(prediction)
            print(f"Input data: {Data.flatten().tolist()}")
            print(f"Predicted class: {index_max_pred}, Confidence: {confidence:.3f}")
            print(f"Probabilities: {prediction}")
    except KeyboardInterrupt:
        print("\nStopping receiver...")
        receiver.stop()
        print(f"Total data received: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
        print(f"Spike data received: {data_store.get_bytes_spikes_formatted()} ({data_store.get_bytes_spikes()} bytes)")
        print(f"No-spike data received: {data_store.get_bytes_no_spikes_formatted()} ({data_store.get_bytes_no_spikes()} bytes)")
        print("Receiver stopped.")


def main():
    parser = argparse.ArgumentParser(description="Hand Tracking Inference with Simple GUI")
    parser.add_argument(
        "--ip",
        default=None,
        help="Override IP address for receiver"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port for receiver"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (command line mode)"
    )
    args = parser.parse_args()

    # Apply IP/port overrides if provided
    if args.ip or args.port:
        # Note: For receiver, we would need to modify the receiver creation
        print(f"IP/port override not implemented for receiver yet")

    if args.no_gui:
        print("🖥️  Starting in command line mode (no GUI)")
        inference_cli()
    else:
        print("🎨 Starting with GUI interface")
        try:
            # Test matplotlib availability before proceeding
            import matplotlib.pyplot as plt
            print("✅ Matplotlib available")
            inference_with_simple_gui()
        except ImportError as e:
            print(f"❌ GUI dependencies not available: {e}")
            print("🔄 Falling back to command line mode...")
            inference_cli()
        except Exception as e:
            print(f"❌ GUI failed with error: {e}")
            print("🔄 Falling back to command line mode...")
            inference_cli()


if __name__ == "__main__":
    main()