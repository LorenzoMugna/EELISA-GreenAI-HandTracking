"""Simple matplotlib-based GUI for real-time hand tracking visualization."""

import matplotlib
# Try to use an interactive backend, fall back to Agg if needed
try:
    matplotlib.use('TkAgg')  # Preferred interactive backend
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Alternative interactive backend
    except ImportError:
        matplotlib.use('Agg')  # Non-interactive fallback
        print("Warning: Using non-interactive matplotlib backend")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
import threading
import time
from collections import deque
from typing import List, Optional
import os
from pathlib import Path


class SimpleHandTrackingGUI:
    """Simple matplotlib-based GUI for hand tracking inference visualization."""

    def __init__(self, data_store, model_predictor=None, image_folder=None):
        """Initialize the simple GUI.

        Args:
            data_store: DataStore instance for accessing UDP data
            model_predictor: Function that takes data and returns (prediction, probabilities)
            image_folder: Path to folder containing gesture images
        """
        self.data_store = data_store
        self.model_predictor = model_predictor
        self.image_folder = Path(image_folder) if image_folder else None

        # Data for visualization
        self.spike_timeline_data = [deque(maxlen=200) for _ in range(6)]
        self.bytes_timeline = deque(maxlen=100)
        self.spike_bytes_timeline = deque(maxlen=100)
        self.no_spike_bytes_timeline = deque(maxlen=100)

        # Current state
        self.current_prediction = 0
        self.current_probabilities = np.zeros(7)
        self.current_confidence = 0.0

        # Statistics tracking
        self.start_time = time.time()
        self.prediction_count = 0
        self.last_prediction_time = time.time()

        # Determine data mode based on folder location
        self.data_mode = self._determine_data_mode()

        # Load gesture images
        self.gesture_images = self._load_gesture_images()

        # Setup matplotlib
        plt.style.use('default')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Hand Tracking - Spike vs No-Spike Comparison', fontsize=16, fontweight='bold')

        # Add close event handler
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # Create subplots
        self.setup_layout()

        # Animation
        self.running = False
        self.animation = None

    def _determine_data_mode(self):
        """Determine data mode based on current working directory."""
        current_path = os.getcwd()

        if "Spikes" in current_path or "/Spikes/" in current_path:
            print("📊 Folder-based detection: Using SPIKE FEATURES mode")
            return "spikes"
        else:
            print("📊 Folder-based detection: Using DIGIT DISTANCES mode")
            return "digits"

    def _on_close(self, event):
        """Handle window close event."""
        print("GUI window closed by user")
        self.stop()

    def _load_gesture_images(self) -> dict:
        """Load gesture images from the img folder."""
        images = {}
        if not self.image_folder or not self.image_folder.exists():
            print("No image folder found, using text labels")
            return images

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
                    img = img.resize((150, 150), Image.Resampling.LANCZOS)
                    images[i] = np.array(img)
                    print(f"Loaded gesture {i} from {img_path.name}")
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        return images

    def setup_layout(self):
        """Setup the matplotlib layout."""
        # Create a grid layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 2, 1])

        # Top left: Prediction with image
        self.ax_prediction = self.fig.add_subplot(gs[0, 0])
        self.ax_prediction.set_title('Current Prediction', fontweight='bold')
        self.ax_prediction.axis('off')

        # Top middle: Spike timeline
        self.ax_spikes = self.fig.add_subplot(gs[0, 1])
        self.ax_spikes.set_title('Spike Events Timeline (Last 5 seconds)', fontweight='bold')
        self.ax_spikes.set_xlabel('Time (seconds ago)')
        self.ax_spikes.set_ylabel('Channel')
        self.ax_spikes.set_ylim(-0.5, 5.5)
        self.ax_spikes.set_yticks(range(6))
        self.ax_spikes.set_yticklabels(['Palm', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
        self.ax_spikes.grid(True, alpha=0.3)

        # Top right: Data overview
        self.ax_data = self.fig.add_subplot(gs[0, 2])
        self.ax_data.set_title('Live Data Overview', fontweight='bold')
        self.ax_data.axis('off')

        # Middle: Bytes comparison graph
        self.ax_bytes = self.fig.add_subplot(gs[1, :])
        self.ax_bytes.set_title('UDP Bytes Transmission - Spike vs No-Spike', fontweight='bold')
        self.ax_bytes.set_xlabel('Time (samples)')
        self.ax_bytes.set_ylabel('Cumulative Bytes')
        self.ax_bytes.grid(True, alpha=0.3)

        # Bottom: Detailed data text
        self.ax_text = self.fig.add_subplot(gs[2, :])
        self.ax_text.set_title('Detailed DataStore Information', fontweight='bold')
        self.ax_text.axis('off')

        plt.tight_layout()

    def update_prediction(self):
        """Update the prediction display using auto-detected data mode."""
        if self.model_predictor:
            try:
                # Auto-detect and extract appropriate data
                if self.data_mode == "spikes":
                    # Use spike features
                    spike_counts = self.data_store.get_all_spike_counts()
                    has_data = any(count > 0 for count in spike_counts)
                    data = None  # Spike predictors usually ignore this parameter
                else:
                    # Use digit distances
                    data = self.data_store.get_digit_distances()
                    has_data = any(d != 0.0 for d in data)

                if has_data:
                    prediction, probabilities = self.model_predictor(data)

                    if prediction >= 0:  # Valid prediction
                        self.current_prediction = prediction
                        self.current_probabilities = probabilities
                        self.current_confidence = np.max(probabilities) if len(probabilities) > 0 else 0.0
                        self.prediction_count += 1
                        self.last_prediction_time = time.time()
                    else:
                        # Invalid prediction
                        self.current_prediction = -1
                        self.current_confidence = 0.0
                else:
                    # No data available
                    self.current_prediction = -1
                    self.current_confidence = 0.0
            except Exception as e:
                print(f"Prediction update error: {e}")
                self.current_prediction = -1
                self.current_confidence = 0.0

        # Update prediction display
        self.ax_prediction.clear()

        # Dynamic title based on data mode
        if self.data_mode == "spikes":
            title = '🔥 Spike-Based Prediction'
        else:
            title = '📏 Distance-Based Prediction'
        self.ax_prediction.set_title(title, fontweight='bold')
        self.ax_prediction.axis('off')

        if self.current_prediction >= 0:
            # Show gesture image if available
            if self.current_prediction in self.gesture_images:
                self.ax_prediction.imshow(self.gesture_images[self.current_prediction])
                self.ax_prediction.text(0.5, -0.1, f'Gesture {self.current_prediction}',
                                      transform=self.ax_prediction.transAxes,
                                      ha='center', fontsize=14, fontweight='bold')
                self.ax_prediction.text(0.5, -0.2, f'Confidence: {self.current_confidence*100:.1f}%',
                                      transform=self.ax_prediction.transAxes,
                                      ha='center', fontsize=12,
                                      color='green' if self.current_confidence > 0.7 else 'orange' if self.current_confidence > 0.5 else 'red')
            else:
                # Text-only display
                self.ax_prediction.text(0.5, 0.7, f'Gesture {self.current_prediction}',
                                      transform=self.ax_prediction.transAxes,
                                      ha='center', fontsize=20, fontweight='bold')
                self.ax_prediction.text(0.5, 0.3, f'Confidence:\n{self.current_confidence*100:.1f}%',
                                      transform=self.ax_prediction.transAxes,
                                      ha='center', fontsize=16,
                                      color='green' if self.current_confidence > 0.7 else 'orange' if self.current_confidence > 0.5 else 'red')
        else:
            no_data_msg = f'No {self.data_mode.title()}\nData Available'
            self.ax_prediction.text(0.5, 0.5, no_data_msg,
                                  transform=self.ax_prediction.transAxes,
                                  ha='center', fontsize=16, color='gray')

    def update_spike_timeline(self):
        """Update the spike timeline visualization."""
        current_time = time.time()

        # Collect spike data from last 5 seconds
        spike_events = []
        for channel in range(6):
            timestamps = self.data_store.get_spike_timestamps(channel)
            for ts in timestamps:
                relative_time = (current_time * 1000 - ts) / 1000  # Convert to seconds ago
                if relative_time <= 5:  # Show last 5 seconds
                    spike_events.append((5 - relative_time, channel))

        # Update spike plot
        self.ax_spikes.clear()
        self.ax_spikes.set_title('Spike Events Timeline (Last 5 seconds)', fontweight='bold')
        self.ax_spikes.set_xlabel('Time (seconds ago)')
        self.ax_spikes.set_ylabel('Channel')
        self.ax_spikes.set_xlim(0, 5)
        self.ax_spikes.set_ylim(-0.5, 5.5)
        self.ax_spikes.set_yticks(range(6))
        self.ax_spikes.set_yticklabels(['Palm', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'])
        self.ax_spikes.grid(True, alpha=0.3)

        # Plot spike events as vertical lines
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for time_pos, channel in spike_events:
            self.ax_spikes.axvline(x=5-time_pos, ymin=(channel-0.4)/6, ymax=(channel+0.4)/6,
                                 color=colors[channel], linewidth=3, alpha=0.8)

    def update_data_overview(self):
        """Update the data overview panel with enhanced statistics."""
        self.ax_data.clear()
        self.ax_data.set_title('Live Data Overview', fontweight='bold')
        self.ax_data.axis('off')

        # Get current data
        spike_counts = self.data_store.get_all_spike_counts()
        firing_rates = self.data_store.get_all_firing_rates()
        total_bytes = self.data_store.get_bytes_received()
        spike_bytes = self.data_store.get_bytes_spikes()
        no_spike_bytes = self.data_store.get_bytes_no_spikes()

        # Calculate new statistics
        current_time = time.time()
        runtime_seconds = current_time - self.start_time
        runtime_minutes = runtime_seconds / 60

        # Bytes per second (throughput)
        if runtime_seconds > 0:
            bytes_per_second = total_bytes / runtime_seconds
            bytes_per_minute = total_bytes / runtime_minutes if runtime_minutes > 0 else 0
        else:
            bytes_per_second = 0
            bytes_per_minute = 0

        # Bytes per prediction
        if self.prediction_count > 0:
            bytes_per_prediction = total_bytes / self.prediction_count
            predictions_per_second = self.prediction_count / runtime_seconds if runtime_seconds > 0 else 0
        else:
            bytes_per_prediction = 0
            predictions_per_second = 0

        # Create enhanced overview text
        overview_text = f"""
Mode: {self.data_mode.upper()}

Spike Counts: {spike_counts}
Total Spikes: {sum(spike_counts)}

Firing Rates (Hz):
{[f'{rate:.1f}' for rate in firing_rates]}

UDP Bytes:
Total: {self.data_store.get_bytes_received_formatted()}
Spikes: {self.data_store.get_bytes_spikes_formatted()}
No-Spikes: {self.data_store.get_bytes_no_spikes_formatted()}

Performance Stats:
Runtime: {runtime_minutes:.1f}m ({runtime_seconds:.1f}s)
Predictions: {self.prediction_count}
Throughput: {bytes_per_second:.1f} B/s
Efficiency: {bytes_per_prediction:.1f} B/pred
Pred Rate: {predictions_per_second:.1f} pred/s

Data Efficiency:
{spike_bytes/(total_bytes+1)*100:.1f}% spike data
        """.strip()

        self.ax_data.text(0.05, 0.95, overview_text, transform=self.ax_data.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=9)

    def update_bytes_graph(self):
        """Update the bytes comparison graph."""
        # Add current data to timeline
        total_bytes = self.data_store.get_bytes_received()
        spike_bytes = self.data_store.get_bytes_spikes()
        no_spike_bytes = self.data_store.get_bytes_no_spikes()

        self.bytes_timeline.append(total_bytes)
        self.spike_bytes_timeline.append(spike_bytes)
        self.no_spike_bytes_timeline.append(no_spike_bytes)

        # Update bytes plot
        self.ax_bytes.clear()
        self.ax_bytes.set_title('UDP Bytes Transmission - Spike vs No-Spike', fontweight='bold')
        self.ax_bytes.set_xlabel('Time (samples)')
        self.ax_bytes.set_ylabel('Cumulative Bytes')
        self.ax_bytes.grid(True, alpha=0.3)

        if len(self.bytes_timeline) > 1:
            x_data = range(len(self.bytes_timeline))

            self.ax_bytes.plot(x_data, list(self.spike_bytes_timeline),
                             'r-', label='Spike Data', linewidth=2)
            self.ax_bytes.plot(x_data, list(self.no_spike_bytes_timeline),
                             'b-', label='No-Spike Data', linewidth=2)
            self.ax_bytes.plot(x_data, list(self.bytes_timeline),
                             'g--', label='Total', linewidth=2, alpha=0.7)

            self.ax_bytes.legend(loc='upper left')

    def update_detailed_text(self):
        """Update the detailed text information with enhanced statistics."""
        self.ax_text.clear()
        self.ax_text.set_title('Detailed DataStore Information', fontweight='bold')
        self.ax_text.axis('off')

        # Get all detailed data
        spike_counts = self.data_store.get_all_spike_counts()
        firing_rates = self.data_store.get_all_firing_rates()
        iat_variances = self.data_store.get_all_inter_arrival_variances()
        values = self.data_store.get_all_values()
        coordinates = self.data_store.get_all_coordinates()
        model = self.data_store.get_model()

        # Get mode-specific features
        if self.data_mode == "spikes":
            features_data = self.data_store.get_spikes_features()
            features_label = "Spike features"
        else:
            features_data = self.data_store.get_digit_distances()
            features_label = "Digit distances"

        # Calculate performance metrics
        current_time = time.time()
        runtime_seconds = current_time - self.start_time
        total_bytes = self.data_store.get_bytes_received()

        bytes_per_second = total_bytes / runtime_seconds if runtime_seconds > 0 else 0
        bytes_per_prediction = total_bytes / self.prediction_count if self.prediction_count > 0 else 0
        predictions_per_second = self.prediction_count / runtime_seconds if runtime_seconds > 0 else 0

        # Format the data with performance stats
        text_content = f"""
--- Current State ({self.data_mode.upper()} Mode) ---
Spike counts: {spike_counts}
Firing rates: {[f'{rate:.2f}' for rate in firing_rates]}
IAT variances: {[f'{var:.2e}' if var is not None else 'None' for var in iat_variances]}
Values: {values}
Coordinates: {coordinates}
Model: {type(model).__name__ if model else 'None'}

{features_label}: {[f'{f:.3f}' for f in features_data]}

Performance Metrics:
Runtime: {runtime_seconds:.1f}s
Total predictions: {self.prediction_count}
Throughput: {bytes_per_second:.1f} bytes/s
Efficiency: {bytes_per_prediction:.1f} bytes/prediction
Prediction rate: {predictions_per_second:.1f} predictions/s

UDP Transmission:
Bytes received: {self.data_store.get_bytes_received_formatted()} ({self.data_store.get_bytes_received()} bytes)
Spike bytes: {self.data_store.get_bytes_spikes_formatted()} ({self.data_store.get_bytes_spikes()} bytes)
No-spike bytes: {self.data_store.get_bytes_no_spikes_formatted()} ({self.data_store.get_bytes_no_spikes()} bytes)
Spike efficiency: {self.data_store.get_bytes_spikes()/(total_bytes+1)*100:.1f}% of total data
        """.strip()

        self.ax_text.text(0.02, 0.98, text_content, transform=self.ax_text.transAxes,
                         verticalalignment='top', fontfamily='monospace', fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    def update_all(self, frame):
        """Update all components of the GUI."""
        if not self.running:
            return

        try:
            self.update_prediction()
            self.update_spike_timeline()
            self.update_data_overview()
            self.update_bytes_graph()
            self.update_detailed_text()
        except Exception as e:
            print(f"GUI update error: {e}")

        return []

    def start(self):
        """Start the GUI animation."""
        self.running = True

        # Ensure the figure is properly configured for display
        self.fig.canvas.manager.set_window_title('Hand Tracking - Spike vs No-Spike')

        self.animation = FuncAnimation(
            self.fig,
            self.update_all,
            interval=500,
            blit=False,
            cache_frame_data=False
        )

        print("GUI started. Close the window to stop.")

        # Force the window to appear and stay interactive
        plt.ion()  # Turn on interactive mode
        self.fig.show()

        # Keep the GUI running with proper event handling
        try:
            while self.running:
                self.fig.canvas.flush_events()
                plt.pause(0.1)
        except KeyboardInterrupt:
            print("\nGUI interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the GUI animation."""
        self.running = False
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
        plt.close(self.fig)


def create_model_predictor(model, scaler):
    """Create a model predictor function."""
    def predict(data):
        try:
            data_array = np.array(data, dtype=np.float32).reshape(1, -1)

            if scaler:
                data_array = scaler.transform(data_array)

            import xgboost as xgb
            dmatrix = xgb.DMatrix(data_array)
            probabilities = model.predict(dmatrix)[0]
            prediction = np.argmax(probabilities)

            return int(prediction), probabilities
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return predict