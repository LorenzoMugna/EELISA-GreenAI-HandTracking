"""Inference script with simple matplotlib GUI for real-time visualization - SPIKE VERSION."""

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
from simple_gui import SimpleHandTrackingGUI

# Add common utilities
sys.path.insert(0, "../../common")
from model_utils import MessageTypes


# Spike-specific configuration
class SpikeConfig:
    """Configuration constants for spike model."""
    DEFAULT_MODEL_PATH = "PredictionModelWithSpike_NoRot.json"
    DEFAULT_SCALER_PATH = "scalerWithSpike_NoRot.pkl"
    DEFAULT_IMAGE_SIZE = (200, 200)
    DEFAULT_UPDATE_INTERVAL_MS = 500
    DEFAULT_SPIKE_TIMELINE_SECONDS = 5


def load_spike_model_and_scaler(model_path: str = "PredictionModelWithSpike_NoRot.json",
                                scaler_path: str = "scalerWithSpike_NoRot.pkl"):
    """Load XGBoost spike model and scaler with error handling.

    Args:
        model_path: Path to the XGBoost spike model file
        scaler_path: Path to the scaler pickle file

    Returns:
        tuple: (loaded_model, scaler) or (None, None) if failed
    """
    if not ML_AVAILABLE:
        print("⚠️  ML dependencies not available")
        return None, None

    try:
        # Load XGBoost model
        loaded_model = xgb.Booster()
        loaded_model.load_model(model_path)
        print(f"✅ Spike model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading spike model from {model_path}: {e}")
        return None, None

    try:
        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f"✅ Spike scaler loaded successfully from {scaler_path}")
    except Exception as e:
        print(f"❌ Error loading spike scaler from {scaler_path}: {e}")
        return loaded_model, None

    return loaded_model, scaler


def create_spike_model_predictor(model, scaler):
    """Create a spike model predictor function.

    Args:
        model: Trained XGBoost spike model
        scaler: Fitted scaler for spike data preprocessing

    Returns:
        Function that takes spike data and returns (prediction, probabilities)
    """
    def predict(data_store):
        try:
            # Extract spike features: [ch0_rate_hz, ..., ch5_rate_hz, ch0_var_isi_ms, ..., ch5_var_isi_ms]
            spike_features = data_store.get_spikes_features()
            data_array = np.array(spike_features, dtype=np.float32).reshape(1, -1)

            # Scale the data if scaler available
            if scaler:
                data_array = scaler.transform(data_array)

            # Create DMatrix and predict
            dmatrix = xgb.DMatrix(data_array)
            probabilities = model.predict(dmatrix)[0]  # Get first (and only) prediction
            prediction = np.argmax(probabilities)

            return int(prediction), probabilities
        except Exception as e:
            print(f"🔥 Spike prediction error: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return predict


def inference_with_simple_gui():
    """Run spike inference with simple matplotlib GUI visualization."""
    print("🔥 Starting Hand Tracking Inference with GUI - SPIKE VERSION")

    try:
        config = udp.get_config()
        print(f"✅ Configuration loaded: {config.num_spike_queues} spike queues, {config.spike_window_ms}ms window")

        # Create data store
        data_store = udp.DataStore(
            num_spike_queues=config.num_spike_queues,
            spike_window_ms=config.spike_window_ms
        )
        print("✅ DataStore created")

        # Load spike model and scaler
        print("🔧 Loading spike model and scaler...")
        if ML_AVAILABLE:
            loaded_model, scaler = load_spike_model_and_scaler(
                SpikeConfig.DEFAULT_MODEL_PATH,
                SpikeConfig.DEFAULT_SCALER_PATH
            )
        else:
            print("⚠️  ML dependencies not available, skipping model loading")
            loaded_model, scaler = None, None

        # Create model predictor
        print("🤖 Creating spike predictor...")
        if loaded_model and scaler:
            # Spike predictor that works with GUI interface
            def spike_predictor(data):
                """Spike predictor that ignores the input data parameter and uses spike features."""
                try:
                    # Extract spike features: [ch0_rate_hz, ..., ch5_rate_hz, ch0_var_isi_ms, ..., ch5_var_isi_ms]
                    spike_features = data_store.get_spikes_features()

                    # Check if we have valid spike data
                    if all(f == 0.0 for f in spike_features):
                        return -1, np.zeros(7)  # No data signal

                    data_array = np.array(spike_features, dtype=np.float32).reshape(1, -1)

                    # Scale the data
                    data_array = scaler.transform(data_array)

                    # Create DMatrix and predict
                    dmatrix = xgb.DMatrix(data_array)
                    probabilities = loaded_model.predict(dmatrix)[0]  # Get first prediction
                    prediction = np.argmax(probabilities)

                    return int(prediction), probabilities
                except Exception as e:
                    print(f"🔥 Spike prediction error: {e}")
                    return -1, np.zeros(7)  # No data signal on error

            predictor = spike_predictor
            print("✅ Using real XGBoost spike model predictor")
        else:
            # Mock predictor for testing
            def mock_spike_predictor(data):
                """Mock spike predictor for testing when model not available."""
                try:
                    # Check if we have any spike data
                    spike_counts = data_store.get_all_spike_counts()
                    if all(count == 0 for count in spike_counts):
                        return -1, np.zeros(7)  # No data signal

                    prediction = int(time.time() * 0.5) % 7  # Cycling predictions like test
                    probabilities = np.random.dirichlet(np.ones(7))
                    probabilities[prediction] *= 3  # Boost predicted class probability
                    probabilities = probabilities / probabilities.sum()
                    return prediction, probabilities
                except Exception as e:
                    print(f"🔥 Mock spike prediction error: {e}")
                    return -1, np.zeros(7)

            predictor = mock_spike_predictor
            print("⚠️  Using mock spike predictor (model/scaler not loaded)")

        print(f"📊 Spike predictor ready: {type(predictor)}")

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

        # Start some test spike data simulation to make GUI visible immediately
        def simulate_spike_test_data():
            """Add some initial spike test data to make GUI show up with content."""
            # Add some spikes to different channels
            for i in range(6):
                for _ in range(np.random.poisson(3)):  # Random spikes per channel
                    data_store.add_spike(i)

            # Add some bytes - more spike data in spike version
            data_store.add_bytes(200, is_spike=True)   # More spike data
            data_store.add_bytes(150, is_spike=False)  # Less coordinate data
            print("✅ Added initial spike test data for GUI display")

        simulate_spike_test_data()

        # Create and start receiver AFTER GUI is ready
        def on_message(msg):
            pass  # We don't need to print messages in GUI mode

        receiver = udp.UDPReceiver(data_store, on_message=on_message)
        receiver.start()
        print("✅ UDP Receiver started")

        print("\n🔥 The SPIKE VERSION GUI window will show:")
        print("  📊 Top left: Current gesture prediction with confidence scores")
        print("  🖼️  Gesture images (if available)")
        print("  📈 Top middle: Real-time spike timeline visualization")
        print("  📋 Top right: Live UDP data overview")
        print("  📊 Middle: UDP bytes comparison (spikes vs no-spikes)")
        print("  📝 Bottom: Detailed DataStore information with SPIKE FEATURES")
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
            print(f"📊 Final spike stats:")
            print(f"  Total data: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
            print(f"  Spike data: {data_store.get_bytes_spikes_formatted()} ({data_store.get_bytes_spikes()} bytes)")
            print(f"  No-spike data: {data_store.get_bytes_no_spikes_formatted()} ({data_store.get_bytes_no_spikes()} bytes)")
            if data_store.get_bytes_received() > 0:
                spike_ratio = (data_store.get_bytes_spikes() / data_store.get_bytes_received()) * 100
                print(f"  Spike data ratio: {spike_ratio:.1f}%")
            print(f"  🔥 Spike features: {data_store.get_spikes_features()}")
        print("✅ Spike application stopped.")


def inference_cli():
    """Run spike inference without GUI (command line mode)."""
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
    print("Spike receiver started. Press Ctrl+C to stop.\n")

    # Load spike model and scaler
    try:
        loaded_model = xgb.Booster()
        loaded_model.load_model("PredictionModelWithSpike_NoRot.json")

        scaler = joblib.load("scalerWithSpike_NoRot.pkl")
        print("Spike model and scaler loaded successfully")
    except Exception as e:
        print(f"Error loading spike model/scaler: {e}")
        return

    try:
        while True:
            # DO SOMETHING WITH THE SPIKE DATA (e.g. inference)
            time.sleep(1)
            print("\n--- Current Spike State ---")
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

            # SPIKE INFERENCE
            # Extract spike features (firing rates + ISI variances)
            spike_data = data_store.get_spikes_features()
            spike_array = np.array(spike_data, dtype=np.float32).reshape(1, -1)
            spike_array = scaler.transform(spike_array)
            dnew = xgb.DMatrix(spike_array)
            prediction = loaded_model.predict(dnew)
            index_max_pred = np.argmax(prediction)
            confidence = np.max(prediction)
            print(f"🔥 Spike input data: {spike_array.flatten().tolist()}")
            print(f"🔥 Predicted class: {index_max_pred}, Confidence: {confidence:.3f}")
            print(f"🔥 Probabilities: {prediction}")
    except KeyboardInterrupt:
        print("\n🔥 Stopping spike receiver...")
        receiver.stop()
        print(f"Total data received: {data_store.get_bytes_received_formatted()} ({data_store.get_bytes_received()} bytes)")
        print(f"Spike data received: {data_store.get_bytes_spikes_formatted()} ({data_store.get_bytes_spikes()} bytes)")
        print(f"No-spike data received: {data_store.get_bytes_no_spikes_formatted()} ({data_store.get_bytes_no_spikes()} bytes)")
        print("🔥 Spike receiver stopped.")


def main():
    parser = argparse.ArgumentParser(description="Hand Tracking Spike Inference with Simple GUI")
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
        print("🖥️  Starting spike inference in command line mode (no GUI)")
        inference_cli()
    else:
        print("🎨 Starting spike inference with GUI interface")
        try:
            # Test matplotlib availability before proceeding
            import matplotlib.pyplot as plt
            print("✅ Matplotlib available")
            inference_with_simple_gui()
        except ImportError as e:
            print(f"❌ GUI dependencies not available: {e}")
            print("🔄 Falling back to spike command line mode...")
            inference_cli()
        except Exception as e:
            print(f"❌ Spike GUI failed with error: {e}")
            print("🔄 Falling back to spike command line mode...")
            inference_cli()


if __name__ == "__main__":
    main()