"""GUI for real-time visualization of UDP data and neural network predictions."""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from collections import deque
from typing import List, Optional
import os
from pathlib import Path


class HandTrackingGUI:
    """Real-time GUI for hand tracking inference visualization."""

    def __init__(self, data_store, model_predictor=None, image_folder=None):
        """Initialize the GUI.

        Args:
            data_store: DataStore instance for accessing UDP data
            model_predictor: Function that takes data and returns (prediction, probabilities)
            image_folder: Path to folder containing gesture images (0.webp, 1.jpg.jpeg, etc.)
        """
        self.data_store = data_store
        self.model_predictor = model_predictor
        self.image_folder = Path(image_folder) if image_folder else None

        self.root = tk.Tk()
        self.root.title("Hand Tracking - Spike vs No-Spike Comparison")
        self.root.geometry("1400x800")

        # Data for visualization
        self.spike_timeline_data = [deque(maxlen=1000) for _ in range(6)]  # 6 spike channels
        self.bytes_timeline = deque(maxlen=100)
        self.spike_bytes_timeline = deque(maxlen=100)
        self.no_spike_bytes_timeline = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)

        # Load gesture images
        self.gesture_images = self._load_gesture_images()

        # GUI components
        self.setup_gui()

        # Update thread control
        self.running = False
        self.update_thread = None

    def _load_gesture_images(self) -> dict:
        """Load gesture images from the img folder."""
        images = {}
        if not self.image_folder or not self.image_folder.exists():
            return images

        # Look for images with different extensions and naming patterns
        for i in range(10):  # Support 0-9 gestures
            img_path = None

            # Try different file patterns
            patterns = [
                f"{i}.webp",
                f"{i}.jpg.jpeg",
                f"{i}.jpg",
                f"{i}.jpeg",
                f"{i}.png"
            ]

            for pattern in patterns:
                test_path = self.image_folder / pattern
                if test_path.exists():
                    img_path = test_path
                    break

            if img_path:
                try:
                    img = Image.open(img_path)
                    img = img.resize((200, 200), Image.Resampling.LANCZOS)
                    images[i] = ImageTk.PhotoImage(img)
                    print(f"Loaded gesture {i} from {img_path.name}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        return images

    def setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top row: Prediction display and spike timeline
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Prediction display
        prediction_frame = ttk.LabelFrame(top_frame, text="Current Prediction", padding=10)
        prediction_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        self.prediction_label = ttk.Label(prediction_frame, text="Prediction: --", font=("Arial", 14, "bold"))
        self.prediction_label.pack(pady=(0, 10))

        self.image_label = ttk.Label(prediction_frame, text="No image")
        self.image_label.pack(pady=(0, 10))

        self.confidence_label = ttk.Label(prediction_frame, text="Confidence: --", font=("Arial", 10))
        self.confidence_label.pack()

        # Right: Spike timeline
        spike_frame = ttk.LabelFrame(top_frame, text="Spike Timeline", padding=5)
        spike_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_spike_timeline(spike_frame)

        # Bottom row: Bytes comparison graph
        bottom_frame = ttk.LabelFrame(main_frame, text="UDP Bytes Transmitted - Spike vs No-Spike", padding=5)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.setup_bytes_graph(bottom_frame)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(status_frame, text="Status: Stopped", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)

        # Control buttons
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_visualization)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_visualization, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

    def setup_spike_timeline(self, parent):
        """Setup the spike timeline visualization."""
        self.spike_fig = Figure(figsize=(8, 4), dpi=100, facecolor='white')
        self.spike_ax = self.spike_fig.add_subplot(111)
        self.spike_ax.set_title("Spike Events Timeline (Last 10 seconds)")
        self.spike_ax.set_xlabel("Time (seconds)")
        self.spike_ax.set_ylabel("Digit/Channel")

        # Set up y-axis for 6 channels (palm + 5 digits)
        self.spike_ax.set_ylim(-0.5, 5.5)
        self.spike_ax.set_yticks(range(6))
        self.spike_ax.set_yticklabels(['Palm', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
        self.spike_ax.grid(True, alpha=0.3)

        self.spike_canvas = FigureCanvasTkAgg(self.spike_fig, parent)
        self.spike_canvas.draw()
        self.spike_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_bytes_graph(self, parent):
        """Setup the bytes comparison graph."""
        self.bytes_fig = Figure(figsize=(12, 3), dpi=100, facecolor='white')
        self.bytes_ax = self.bytes_fig.add_subplot(111)
        self.bytes_ax.set_title("UDP Bytes Transmission Comparison")
        self.bytes_ax.set_xlabel("Time")
        self.bytes_ax.set_ylabel("Bytes")
        self.bytes_ax.grid(True, alpha=0.3)

        self.bytes_canvas = FigureCanvasTkAgg(self.bytes_fig, parent)
        self.bytes_canvas.draw()
        self.bytes_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_visualization(self):
        """Start the real-time visualization."""
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running")

            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop_visualization(self):
        """Stop the real-time visualization."""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

    def _update_loop(self):
        """Main update loop running in separate thread."""
        while self.running:
            try:
                self._update_prediction()
                self._update_spike_timeline()
                self._update_bytes_graph()
                time.sleep(0.1)  # Update every 100ms
            except Exception as e:
                print(f"GUI update error: {e}")

    def _update_prediction(self):
        """Update the prediction display."""
        if self.model_predictor:
            try:
                data = self.data_store.get_digit_distances()
                if any(d != 0.0 for d in data):  # Only predict if we have data
                    prediction, probabilities = self.model_predictor(data)

                    # Update prediction text
                    confidence = np.max(probabilities) if len(probabilities) > 0 else 0.0

                    def update_ui():
                        self.prediction_label.config(text=f"Prediction: {prediction}")
                        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")

                        # Update image if available
                        if prediction in self.gesture_images:
                            self.image_label.config(image=self.gesture_images[prediction], text="")
                        else:
                            self.image_label.config(image="", text=f"Gesture {prediction}")

                    self.root.after(0, update_ui)

            except Exception as e:
                print(f"Prediction update error: {e}")

    def _update_spike_timeline(self):
        """Update the spike timeline visualization."""
        current_time = time.time()

        # Collect spike data from last 10 seconds
        spike_events = []
        for channel in range(6):
            timestamps = self.data_store.get_spike_timestamps(channel)
            # Convert to relative time (seconds ago)
            for ts in timestamps:
                relative_time = (current_time * 1000 - ts) / 1000  # Convert to seconds ago
                if relative_time <= 10:  # Only show last 10 seconds
                    spike_events.append((10 - relative_time, channel))  # Flip time axis

        def update_spike_plot():
            self.spike_ax.clear()
            self.spike_ax.set_title("Spike Events Timeline (Last 10 seconds)")
            self.spike_ax.set_xlabel("Time (seconds ago)")
            self.spike_ax.set_ylabel("Channel")
            self.spike_ax.set_xlim(0, 10)
            self.spike_ax.set_ylim(-0.5, 5.5)
            self.spike_ax.set_yticks(range(6))
            self.spike_ax.set_yticklabels(['Palm', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
            self.spike_ax.grid(True, alpha=0.3)

            # Plot spike events as vertical lines
            for time_pos, channel in spike_events:
                self.spike_ax.axvline(x=10-time_pos, ymin=(channel-0.4)/6, ymax=(channel+0.4)/6,
                                    color='red', linewidth=2, alpha=0.7)

            self.spike_canvas.draw()

        self.root.after(0, update_spike_plot)

    def _update_bytes_graph(self):
        """Update the bytes comparison graph."""
        # Collect current byte counts
        total_bytes = self.data_store.get_bytes_received()
        spike_bytes = self.data_store.get_bytes_spikes()
        no_spike_bytes = self.data_store.get_bytes_no_spikes()

        # Add to timeline
        self.bytes_timeline.append(total_bytes)
        self.spike_bytes_timeline.append(spike_bytes)
        self.no_spike_bytes_timeline.append(no_spike_bytes)

        def update_bytes_plot():
            self.bytes_ax.clear()
            self.bytes_ax.set_title("UDP Bytes Transmission Comparison")
            self.bytes_ax.set_xlabel("Time (samples)")
            self.bytes_ax.set_ylabel("Bytes")
            self.bytes_ax.grid(True, alpha=0.3)

            x_data = range(len(self.bytes_timeline))

            if len(self.bytes_timeline) > 0:
                self.bytes_ax.plot(x_data, list(self.spike_bytes_timeline),
                                 'r-', label='Spike Data', linewidth=2)
                self.bytes_ax.plot(x_data, list(self.no_spike_bytes_timeline),
                                 'b-', label='No-Spike Data', linewidth=2)
                self.bytes_ax.plot(x_data, list(self.bytes_timeline),
                                 'g--', label='Total', linewidth=2, alpha=0.7)

                self.bytes_ax.legend()

                # Add text showing current values
                self.bytes_ax.text(0.02, 0.98,
                                 f"Spike: {self.data_store.get_bytes_spikes_formatted()}\n"
                                 f"No-Spike: {self.data_store.get_bytes_no_spikes_formatted()}\n"
                                 f"Total: {self.data_store.get_bytes_received_formatted()}",
                                 transform=self.bytes_ax.transAxes,
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.bytes_canvas.draw()

        self.root.after(0, update_bytes_plot)

    def run(self):
        """Run the GUI main loop."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):
        """Handle window closing."""
        self.stop_visualization()
        self.root.destroy()


def create_model_predictor(model, scaler):
    """Create a model predictor function.

    Args:
        model: Trained XGBoost model
        scaler: Fitted scaler for data preprocessing

    Returns:
        Function that takes data and returns (prediction, probabilities)
    """
    def predict(data):
        try:
            # Convert to numpy array and reshape
            data_array = np.array(data, dtype=np.float32).reshape(1, -1)

            # Scale the data
            if scaler:
                data_array = scaler.transform(data_array)

            # Create DMatrix and predict
            import xgboost as xgb
            dmatrix = xgb.DMatrix(data_array)
            probabilities = model.predict(dmatrix)[0]  # Get first (and only) prediction
            prediction = np.argmax(probabilities)

            return int(prediction), probabilities
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return predict