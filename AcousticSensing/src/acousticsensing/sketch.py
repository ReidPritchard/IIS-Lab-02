"""
Sketch mode: a p5py window the pipeline drives, instead of the analysis display.

The analysis display (visuals.py) shows every stage of the pipeline so it can be
debugged and trained. This window shows none of that: it is a canvas that reacts
to what the pipeline hears, which is what the finished interaction looks like.

Each frame drains the microphone queue, pushes the blocks through the same
AcousticPipeline the analysis display uses, and hands every Event to on_event.
Drawing then happens from state those calls left behind. To make the sketch do
something new, change on_event (what an event leaves behind) and render (how that
is drawn); nothing else in the pipeline needs to know.

p5 draws from module-level functions and globals, so it can run one sketch per
process. The skia renderer is used because its loop returns when the window is
closed, letting app.py close the microphone afterwards.
Press Escape or close the window to quit.
"""

import builtins
import queue
import time

import glfw
import numpy as np
import p5
from p5.core import p5 as p5_state

from acousticsensing.audio import drain, play
from acousticsensing.config import CLASS_LABELS
from acousticsensing.pipeline import AcousticPipeline, Event
from acousticsensing.visuals import class_color

WINDOW_TITLE = "Acoustic Sensing"

# Block RMS shown as a full level meter
METER_MAX_RMS = 0.2

# How quickly the level meter falls back after a loud block, per frame
METER_DECAY = 0.85

# Predictions below this confidence are shown but not added to the loop
MIN_CONFIDENCE = 0.6

# Seconds the background flashes after an event is added to the loop
FLASH_SECONDS = 0.25

# Extra seconds of detection ignored after playback, on top of the sample length
PLAYBACK_MARGIN_SECONDS = 0.1

# Colour of an event detected while no model is loaded
UNCLASSIFIED_COLOR = (160, 160, 160)


def event_color(event: Event) -> tuple[int, int, int]:
    """
    Pick the colour of one event: its class colour, or grey with no model.

    :param event: The event the pipeline produced.
    :return: RGB, 0 to 255.
    """
    if event.prediction is None:
        return UNCLASSIFIED_COLOR
    r, g, b = class_color(event.prediction.label)[:3]
    return round(float(r) * 255), round(float(g) * 255), round(float(b) * 255)


class Sketch:
    """
    A p5 window fed by the audio analysis pipeline. This sketch is an "Acoustic Gesture Looper"
    Imagine an 8- or 16-beat loop running continuously. You perform sounds on the desk
    and each recognized gesture gets placed into the current position in the loop.

    :ivar level: Smoothed block RMS, for the level meter.
    :ivar last_event: The most recent event, for the caption.
    :ivar flash_until: time.monotonic() at which the flash ends.
    """

    def __init__(
        self,
        pipeline: AcousticPipeline,
        blocks: queue.Queue,
        size: tuple[int, int] = (900, 600),
        frame_rate: int = 60,
    ):
        """
        :param pipeline: Every stage from segmentation to classification.
        :param blocks: Queue the microphone callback fills.
        :param size: Window size in pixels.
        :param frame_rate: Frames drawn per second.
        """
        self.pipeline = pipeline
        self.blocks = blocks
        self.size = size
        self.frame_rate = frame_rate

        self.level = 0.0
        self.last_event: Event | None = None
        self.flash_until = 0.0

        # Looper state
        self.loop_length = 8  # Number of beats in the loop
        self.current_beat = 0  # Current beat position in the loop
        # One row per beat, one slot per class label, so a beat can hold several
        # classes at once. Built per beat so the rows are not one shared list.
        self.loop: list[list[Event | None]] = [
            [None] * (max(CLASS_LABELS) + 1) for _ in range(self.loop_length)
        ]
        self.last_beat_time = time.monotonic()  # Track time for beat progression
        self.bpm = 120  # Beats per minute for the loop
        # The microphone hears the loop playing back; events before this time are
        # that echo and must not be recorded into the loop again
        self.mute_until = 0.0

    def start(self) -> None:
        """Run the sketch until the window is closed or Escape is pressed."""
        p5.run(
            sketch_setup=self.setup,
            sketch_draw=self.draw,
            frame_rate=self.frame_rate,
            renderer="skia",
        )

    def setup(self) -> None:
        """Called once by p5 when the window opens."""
        p5.size(*self.size)
        glfw.set_window_title(p5_state.sketch.window, WINDOW_TITLE)

    def draw(self) -> None:
        """Called by p5 every frame: advance the pipeline, then draw."""
        for block in drain(self.blocks):
            self.level = max(self.level, float(np.sqrt(np.mean(block**2))))
            for event in self.pipeline.push(block):
                self.on_event(event)

        self.render(time.monotonic())
        self.level *= METER_DECAY

    def on_event(self, event: Event) -> None:
        """
        React to one event from the pipeline.

        :param event: The event, with whatever the later stages decided.
        """
        self.last_event = event

        # A confident prediction goes into the current beat of the loop, keeping
        # its audio for playback
        prediction = event.prediction
        if prediction is None or prediction.confidence < MIN_CONFIDENCE:
            return
        if time.monotonic() < self.mute_until:
            return

        beat = self.loop[self.current_beat]
        # A model trained on since-retired classes can name a label with no slot
        # Class 0 is the "no event" class, which we want to ignore for the loop
        if 0 < prediction.label < len(beat):
            beat[prediction.label] = event
            self.flash_until = time.monotonic() + FLASH_SECONDS
            print(f"Beat {self.current_beat + 1}: {prediction}")

    def render(self, now: float) -> None:
        """
        Draw one frame from the current state.

        :param now: Current time, from time.monotonic().
        """
        width, height = builtins.width, builtins.height

        if now < self.flash_until and self.last_event is not None:
            r, g, b = event_color(self.last_event)
            p5.background(r * 0.3, g * 0.3, b * 0.3)
        else:
            p5.background(15)

        self._render_meter(width, height)
        self._render_caption()
        self._render_sequencer(width, height)

        # Update loop state, playing each event in a beat once as the beat starts
        beat_duration = 60.0 / self.bpm
        if now - self.last_beat_time >= beat_duration:
            self.current_beat = (self.current_beat + 1) % self.loop_length
            self.last_beat_time += beat_duration

            self._play_beat(now)

    def _play_beat(self, now: float) -> None:
        """
        Play the recorded samples in the current beat, mixed together.

        :param now: Current time, from time.monotonic().
        """
        samples = [
            event.segment.audio
            for event in self.loop[self.current_beat]
            if event is not None
        ]
        if not samples:
            return

        # Every segment has the same length, so they sum sample for sample
        play(np.clip(np.sum(samples, axis=0), -1.0, 1.0), self.pipeline.config)
        self.mute_until = (
            now + self.pipeline.config.segment_seconds + PLAYBACK_MARGIN_SECONDS
        )

    def _render_meter(self, width: float, height: float) -> None:
        """Draw the input level along the bottom, with the onset threshold."""
        bar_height = 12
        top = height - bar_height

        p5.no_stroke()
        p5.fill(40)
        p5.rect(0, top, width, bar_height)
        p5.fill(90, 200, 120)
        p5.rect(0, top, width * min(self.level / METER_MAX_RMS, 1.0), bar_height)

        # Blocks louder than this line start an event
        threshold = width * self.pipeline.config.onset_rms / METER_MAX_RMS
        p5.stroke(255)
        p5.stroke_weight(2)
        p5.line(threshold, top, threshold, height)

    def _render_caption(self) -> None:
        """Name the last event in the top left corner."""
        if self.last_event is None:
            caption = "Listening..."
        elif self.last_event.prediction is None:
            caption = "Event (no model)"
        else:
            caption = str(self.last_event.prediction)

        p5.no_stroke()
        p5.fill(230)
        p5.text_size(24)
        p5.text(caption, 20, 20)

    def _render_sequencer(self, width: float, height: float) -> None:
        """Draw the loop sequencer at the bottom of the window."""
        # Width is full window, height is a row for each potential class
        beat_width = width / self.loop_length
        slot_height = 40
        top = height / 2

        p5.no_stroke()

        for i, beat in enumerate(self.loop):
            x = i * beat_width
            for label, event in enumerate(beat):
                if event is not None:
                    r, g, b = event_color(event)
                    p5.fill(r, g, b)
                else:
                    p5.fill(50)
                p5.rect(x, top + label * slot_height, beat_width - 2, slot_height - 2)

        # Highlight the current beat, across every class row
        p5.stroke(255)
        p5.stroke_weight(3)
        p5.no_fill()
        p5.rect(
            self.current_beat * beat_width,
            top,
            beat_width - 2,
            len(self.loop[0]) * slot_height - 2,
        )
