"""
The live display, and the keyboard controls for recording training data.

The window shows the FFT of the newest audio above a scrolling spectrogram, with
each detected event marked on the spectrogram in its predicted class colour.
Below the spectrum sit the recording status and the last prediction, so every
stage of the pipeline is visible at once.

Press 'f' for a second window holding one panel per feature of the last event,
drawn by each feature's own plot function. That is the point of it: a feature
added to features.FEATURES shows up there with no change to this file.

The main window is blitted, so its artists and axis limits are fixed. The
feature window is redrawn from scratch, which is why it lives in a window of its
own -- and it costs nothing here, because it only redraws when an event happens.
"""

import warnings
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from acousticsensing.audio import drain
from acousticsensing.config import CLASS_LABELS, Config
from acousticsensing.dataset import DATA_DIR, save_clip
from acousticsensing.features import FEATURES, frequency_bins, spectrum_db
from acousticsensing.pipeline import AcousticPipeline, Event

# Marker class for an event detected while no model is loaded
UNCLASSIFIED = -1

# Fixed y-axis range for the dB spectrum. A full-scale sine peaks at about -6 dB.
MIN_DISPLAY_DB = -120
MAX_DISPLAY_DB = 0

# Colour range for the spectrogram image
SPECTROGRAM_MIN_DB = -90
SPECTROGRAM_MAX_DB = -20

# Colours the class readouts cycle through, one per class in CLASS_LABELS order
CLASS_COLORMAP = "tab10"

# Panels are stacked in one column until there are more than this many
FEATURE_PANEL_ROWS_PER_COLUMN = 4


def class_color(label: int) -> str:
    """
    Pick the colour of one class.

    Colours come from a colormap rather than a hand-written dict, so adding a
    class to CLASS_LABELS needs no change here.

    :param label: The class label.
    :return: A matplotlib colour. Black for a label absent from CLASS_LABELS,
        which means a model predicting a class that has since been retired.
    """
    order = list(CLASS_LABELS)
    if label not in order:
        return "black"

    colors = plt.get_cmap(CLASS_COLORMAP).colors
    return colors[order.index(label) % len(colors)]


def _release_key(action: str, key: str) -> None:
    """
    Free a key matplotlib binds to a toolbar action, so the app can use it.

    :param action: The rcParams keymap entry, e.g. "keymap.home".
    :param key: The key to remove from it.
    """
    plt.rcParams[action] = [k for k in plt.rcParams[action] if k != key]


_release_key("keymap.home", "r")  # 'r' records
_release_key("keymap.fullscreen", "f")  # 'f' toggles the feature window


class FeaturePanels:
    """
    A separate window showing the features of the most recent event.

    Each panel is drawn by its feature's own plot function, so this window shows
    exactly what the classifier was given, in whatever form the feature author
    chose to draw it.
    """

    def __init__(self, config: Config):
        """
        :param config: Passed to each feature's plot function.
        """
        self.config = config
        self.fig = None
        self.axes: dict[str, Axes] = {}
        self.pending: Event | None = None

    @property
    def visible(self) -> bool:
        """Whether the feature window is open."""
        return self.fig is not None

    def toggle(self, names: tuple[str, ...]) -> None:
        """
        Open the window, or close it if it is already open.

        :param names: Features to give a panel to, in vector order.
        """
        if self.visible:
            self.close()
        else:
            self.open(names)

    def open(self, names: tuple[str, ...]) -> None:
        """
        Create the window with one empty panel per feature.

        :param names: Features to give a panel to, in vector order.
        """
        if self.visible or not names:
            return

        columns = ceil(len(names) / FEATURE_PANEL_ROWS_PER_COLUMN)
        rows = ceil(len(names) / columns)

        self.fig = plt.figure(figsize=(7 * columns, 2.2 * rows), layout="constrained")
        grid = self.fig.subplots(rows, columns, squeeze=False).ravel(order="F")

        self.axes = dict(zip(names, grid))
        for name, ax in self.axes.items():
            ax.set_title(FEATURES[name].title)
        # Panels fill each column in turn, so a short last column has spares
        for ax in grid[len(names) :]:
            ax.set_axis_off()

        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Features of the last event")
        self.fig.canvas.mpl_connect("close_event", lambda event: self._forget())

        with warnings.catch_warnings():
            # This figure is created while the event loop is already running, so
            # it has to be shown by hand. A non-interactive backend has no window
            # to show and says so; nothing can be done about that here.
            warnings.simplefilter("ignore", UserWarning)
            self.fig.show()

    def close(self) -> None:
        """Close the feature window."""
        if self.fig is not None:
            plt.close(self.fig)
        self._forget()

    def show(self, event: Event) -> None:
        """
        Queue an event to be drawn on the next update.

        :param event: The event whose features to draw.
        """
        self.pending = event

    def update(self) -> None:
        """Redraw the panels, if an event is waiting and the window is open."""
        if not self.visible or self.pending is None:
            return

        event, self.pending = self.pending, None
        title = str(event.prediction) if event.prediction else "unclassified"

        for name, ax in self.axes.items():
            if name not in event.features:
                continue
            ax.clear()
            ax.set_title(f"{FEATURES[name].title} - {title}")
            FEATURES[name].plot(ax, event.features[name], self.config)

        self.fig.canvas.draw_idle()

    def _forget(self) -> None:
        """Drop the figure, so the next open() builds it again."""
        self.fig = None
        self.axes = {}


class Visualizer:
    """
    Draws the live audio and drives the pipeline from the audio queue.

    Every block goes into the pipeline, so detection and classification happen
    here, on the animation's cadence.
    """

    def __init__(
        self,
        pipeline: AcousticPipeline,
        blocks,
        display_seconds: int = 10,
        max_frequency: float | None = None,
        data_dir: Path = DATA_DIR,
    ):
        """
        :param pipeline: The wired pipeline. Blocks are pushed into it and the
            events it returns are drawn.
        :param blocks: Queue of audio blocks written by the audio callback.
        :param display_seconds: How much history the spectrogram shows.
        :param max_frequency: Highest frequency drawn, in Hz. Defaults to Nyquist.
        :param data_dir: Where recorded clips are saved.
        :raises ValueError: If no class is defined.
        """
        if not CLASS_LABELS:
            raise ValueError("CLASS_LABELS is empty; define at least one class.")

        self.pipeline = pipeline
        self.config = pipeline.config
        self.blocks = blocks
        self.display_seconds = display_seconds
        self.data_dir = data_dir

        self.max_frequency = max_frequency or self.config.nyquist

        # Recording state. The class is chosen before recording starts, so the
        # display never has to block on input.
        self.recording = False
        self.current_label = next(iter(CLASS_LABELS))
        self.recorded_blocks: list[np.ndarray] = []
        self.last_saved: Path | None = None

        # Newest events, as (sample index, class label), for the markers
        self.recent_events: list[tuple[int, int]] = []
        self.samples_seen = 0
        self.last_event: Event | None = None

        # Rolling buffer holding the newest fft_size samples, since audio blocks
        # are smaller than one FFT frame
        self.sample_buffer = np.zeros(self.config.fft_size, dtype=np.float32)

        bins = frequency_bins(self.config)
        self.shown_bins = bins <= self.max_frequency
        self.shown_frequencies = bins[self.shown_bins]

        # One spectrogram column per audio block
        columns = max(
            1, int(display_seconds * self.config.sample_rate / self.config.block_size)
        )
        self.spectrogram_data = np.full(
            (len(self.shown_frequencies), columns), SPECTROGRAM_MIN_DB, dtype=np.float32
        )
        self.latest_spectrum = np.full(
            len(self.shown_frequencies), MIN_DISPLAY_DB, dtype=np.float32
        )

        self._build_figure()

    # -------------------------------
    # Setup
    # -------------------------------

    def _build_figure(self) -> None:
        """Lay out the window: spectrum above spectrogram, with the readouts."""
        axes = plt.figure(figsize=(13, 8), layout="constrained").subplot_mosaic(
            [["fft"], ["spectrogram"]], height_ratios=[1, 2]
        )
        self.fft_ax = axes["fft"]
        self.spectrogram_ax = axes["spectrogram"]
        self.fig = self.fft_ax.figure
        self.fig.canvas.mpl_connect("key_press_event", self.handle_key_press)

        (self.line,) = self.fft_ax.plot([], [], lw=1)
        self.fft_ax.set_xlim(0, self.max_frequency)
        self.fft_ax.set_ylim(MIN_DISPLAY_DB, MAX_DISPLAY_DB)
        self.fft_ax.set_xlabel("Frequency (Hz)")
        self.fft_ax.set_ylabel("Magnitude (dB)")
        self.fft_ax.set_title("Live spectrum")

        self.image = self.spectrogram_ax.imshow(
            self.spectrogram_data,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[-self.display_seconds, 0, 0, self.max_frequency],
            cmap="magma",
            vmin=SPECTROGRAM_MIN_DB,
            vmax=SPECTROGRAM_MAX_DB,
        )
        self.spectrogram_ax.set_title("Spectrogram and detected events")
        self.spectrogram_ax.set_xlabel("Time relative to now (seconds)")
        self.spectrogram_ax.set_ylabel("Frequency (Hz)")
        self.fig.colorbar(self.image, ax=self.spectrogram_ax).set_label(
            "Magnitude (dB)"
        )

        # One marker line per class, so a marker's colour names its class. Blitting
        # needs a fixed set of artists, which rules out drawing them per event.
        # The extra line is for events detected while no model is loaded.
        self.event_markers = {}
        for label, name in [*CLASS_LABELS.items(), (UNCLASSIFIED, "unclassified")]:
            (marker,) = self.spectrogram_ax.plot(
                [],
                [],
                marker="v",
                ls="none",
                ms=10,
                color=class_color(label),
                label=name,
            )
            self.event_markers[label] = marker
        self.spectrogram_ax.legend(loc="upper left", ncol=len(self.event_markers))

        self.status_text = self.fft_ax.text(
            0.01,
            0.95,
            "",
            transform=self.fft_ax.transAxes,
            va="top",
            family="monospace",
        )
        self.prediction_text = self.fft_ax.text(
            0.99,
            0.95,
            "Listening...",
            transform=self.fft_ax.transAxes,
            va="top",
            ha="right",
            family="monospace",
            fontsize=13,
            fontweight="bold",
            color="grey",
        )
        self._update_status()

        self.feature_panels = FeaturePanels(self.config)

    def _animated_artists(self) -> tuple:
        """Every artist that changes between frames; blit redraws only these."""
        return (
            self.line,
            self.image,
            self.status_text,
            self.prediction_text,
            *self.event_markers.values(),
        )

    # -------------------------------
    # Controls
    # -------------------------------

    def handle_key_press(self, event) -> None:
        """
        Act on a key press.

        :param event: The matplotlib key press event.
        """
        if event.key == "q":
            self.stop()
        elif event.key in {str(label) for label in CLASS_LABELS}:
            if self.recording:
                print("Stop recording before changing the class.")
                return
            self.current_label = int(event.key)
            print(f"Recording class set to {CLASS_LABELS[self.current_label]}")
        elif event.key == "r":
            self._toggle_recording()
        elif event.key == "u":
            self._undo_last_save()
        elif event.key == "f":
            self.feature_panels.toggle(self.pipeline.extractor.names)
            if self.last_event is not None:
                self.feature_panels.show(self.last_event)

        self._update_status()

    def _toggle_recording(self) -> None:
        """Start recording, or stop and save what was recorded."""
        if self.recording:
            self._save_recording()
        else:
            self.recorded_blocks = []
            print(f"Recording {CLASS_LABELS[self.current_label]}...")
        self.recording = not self.recording

    def _save_recording(self) -> None:
        """Write the recorded audio as one clip under the current class."""
        if not self.recorded_blocks:
            print("Nothing recorded; no clip saved.")
            return

        audio = np.concatenate(self.recorded_blocks)
        self.last_saved = save_clip(
            audio, self.current_label, self.config, self.data_dir
        )
        duration = len(audio) / self.config.sample_rate
        print(
            f"Saved {duration:.2f}s as {CLASS_LABELS[self.current_label]} "
            f"-> {self.last_saved}"
        )

    def _undo_last_save(self) -> None:
        """Delete the most recently saved clip."""
        if self.last_saved is None:
            print("No saved clip to undo.")
            return

        self.last_saved.unlink(missing_ok=True)
        print(f"Deleted {self.last_saved}")
        self.last_saved = None

    # -------------------------------
    # Animation
    # -------------------------------

    def _process_block(self, block: np.ndarray) -> None:
        """
        Draw one audio block and run it through the pipeline.

        :param block: One block from the microphone.
        """
        block = np.ravel(block)
        self.sample_buffer = np.concatenate([self.sample_buffer, block])[
            -self.config.fft_size :
        ]
        self.samples_seen += len(block)

        if self.recording:
            # Keep the raw block, not the rolling buffer, so no sample is stored twice
            self.recorded_blocks.append(block.copy())

        spectrum = spectrum_db(self.sample_buffer, self.config)[self.shown_bins]
        # Scroll left by one column and put the newest spectrum on the right
        self.spectrogram_data[:, :-1] = self.spectrogram_data[:, 1:]
        self.spectrogram_data[:, -1] = spectrum
        self.latest_spectrum = spectrum

        for event in self.pipeline.push(block):
            self._show_event(event)

    def _show_event(self, event: Event) -> None:
        """
        Record one detected event for display.

        :param event: The event the pipeline produced.
        """
        self.last_event = event
        self.feature_panels.show(event)

        label = event.prediction.label if event.prediction else UNCLASSIFIED
        self.recent_events.append((event.segment.start_sample, label))

        if event.prediction is None:
            self.prediction_text.set_text("Event (no model)")
            self.prediction_text.set_color("grey")
        else:
            self.prediction_text.set_text(str(event.prediction))
            self.prediction_text.set_color(class_color(event.prediction.label))

    def _update_markers(self) -> None:
        """Place the event markers on the spectrogram's time axis."""
        oldest = self.samples_seen - self.display_seconds * self.config.sample_rate
        self.recent_events = [
            (sample, label) for sample, label in self.recent_events if sample >= oldest
        ]

        height = self.max_frequency * 0.97
        for label, marker in self.event_markers.items():
            times = [
                (sample - self.samples_seen) / self.config.sample_rate
                for sample, event_label in self.recent_events
                if event_label == label
            ]
            marker.set_data(times, [height] * len(times))

    def _update_status(self) -> None:
        """Refresh the status line."""
        # Emoji are missing from matplotlib's default fonts, so use plain symbols
        state = "● REC" if self.recording else "○ idle"
        keys = "  ".join(f"[{k}] {v}" for k, v in CLASS_LABELS.items())
        self.status_text.set_text(
            f"Class: {CLASS_LABELS[self.current_label]} | {state}\n"
            f"{keys}  [r] rec  [u] undo  [f] features  [q] quit"
        )
        self.status_text.set_color("red" if self.recording else "black")

    def update_plot(self, frame) -> tuple:
        """
        Advance one animation frame.

        :param frame: Frame number, unused.
        :return: The artists blitting must redraw.
        """
        blocks = drain(self.blocks)
        for block in blocks:
            self._process_block(block)

        if blocks:
            self.line.set_data(self.shown_frequencies, self.latest_spectrum)
            self.image.set_data(self.spectrogram_data)
            self._update_markers()
            self.feature_panels.update()

        return self._animated_artists()

    def start(self) -> None:
        """Run the display until the window is closed."""
        self.animation = FuncAnimation(
            self.fig,
            self.update_plot,
            blit=True,
            interval=50,
            cache_frame_data=False,
        )
        plt.show()

    def stop(self) -> None:
        """Close both windows."""
        self.feature_panels.close()
        plt.close(self.fig)
