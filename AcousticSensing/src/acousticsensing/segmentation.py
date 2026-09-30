"""
Stage 2: event detection and segmentation.

Continuous audio in, one fixed-length segment per acoustic event out. Every
later stage sees only segments, which is what makes them simple: a segment is
always the same length and always holds an event, so features have a fixed size
and the classifier is never asked about silence.

The same StreamSegmenter runs live and over recorded clips (see segment_audio),
so the model is trained on exactly the kind of segment it is later given.
"""

from collections import deque
from dataclasses import dataclass
from math import ceil

import numpy as np

from acousticsensing.config import Config


def rms(audio: np.ndarray) -> float:
    """
    Root-mean-square level of a block of audio.

    :param audio: Audio as a 1D numpy array.
    :return: The RMS level, 0.0 for empty input.
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


@dataclass(frozen=True)
class Segment:
    """
    One detected acoustic event.

    :param audio: Exactly config.segment_samples of audio, starting
        config.preroll_samples before the onset.
    :param start_sample: Index of the segment's first sample in the stream it
        was cut from. Used to place the event on the display's time axis.
    :param level: RMS of the block that triggered detection, i.e. how loud the
        event was.
    """

    audio: np.ndarray
    start_sample: int
    level: float


class StreamSegmenter:
    """
    Cuts a continuous stream into one segment per event.

    Three states, in order:

    1. Idle. Each block is kept in a short pre-roll buffer and tested against
       config.onset_rms.
    2. Collecting. A block crossed the threshold, so that block and the pre-roll
       start a segment; further blocks are appended until the segment is full.
    3. Refractory. For config.refractory_seconds afterwards, no new onset is
       accepted, so one physical clap produces one segment rather than one per
       block of its decay.

    Detection is per block, so onsets are located to within one block
    (about 21 ms at 96 kHz with a 2048-sample block). The pre-roll covers the
    part of the attack that lands in the previous block.
    """

    def __init__(self, config: Config):
        """
        :param config: Segment length, pre-roll, threshold and refractory period.
        """
        self.config = config
        self.segment_samples = config.segment_samples
        # Pre-roll is held as whole blocks, since blocks are the unit that arrives
        self.preroll_blocks = max(1, ceil(config.preroll_samples / config.block_size))
        self.refractory_blocks = ceil(config.refractory_samples / config.block_size)
        self.reset()

    def reset(self) -> None:
        """Forget all buffered audio and return to the idle state."""
        self._preroll: deque[np.ndarray] = deque(maxlen=self.preroll_blocks)
        self._collecting: list[np.ndarray] | None = None
        self._segment_start = 0
        self._segment_level = 0.0
        self._cooldown_blocks = 0
        self._samples_seen = 0

    def push(self, block: np.ndarray) -> list[Segment]:
        """
        Feed one block of audio and collect any segment it completes.

        :param block: One block of audio, usually config.block_size samples.
        :return: The segments finished by this block. A list rather than an
            optional value so the caller's loop reads the same whether zero or
            one event landed here.
        """
        block = np.ravel(np.asarray(block, dtype=np.float32))
        block_start = self._samples_seen
        self._samples_seen += len(block)

        if self._collecting is not None:
            self._collecting.append(block)
            return self._finish_if_complete()

        if self._cooldown_blocks > 0:
            self._cooldown_blocks -= 1
            self._preroll.append(block)
            return []

        level = rms(block)
        if level < self.config.onset_rms:
            self._preroll.append(block)
            return []

        # Onset. The segment starts at the front of the pre-roll, which is why
        # the start index counts backwards from this block.
        preroll = list(self._preroll)
        self._preroll.clear()
        self._segment_start = block_start - sum(len(b) for b in preroll)
        self._segment_level = level
        self._collecting = [*preroll, block]

        # A long pre-roll can already complete the segment on the onset block
        return self._finish_if_complete()

    def flush(self) -> list[Segment]:
        """
        End a segment that is still being collected, padding it with silence.

        Only useful offline, where the audio stops mid-event: a clip whose clap
        sits near its end would otherwise be dropped. A live stream keeps
        supplying blocks, so it never needs this.

        :return: The padded segment, or an empty list if none was in progress.
        """
        return [self._finish()] if self._collecting is not None else []

    def _collected_samples(self) -> int:
        """Samples gathered so far for the segment being collected."""
        return sum(len(block) for block in self._collecting or ())

    def _finish_if_complete(self) -> list[Segment]:
        """Close the segment being collected, if it has enough audio yet."""
        if self._collected_samples() < self.segment_samples:
            return []
        return [self._finish()]

    def _finish(self) -> Segment:
        """
        Close the segment being collected and enter the refractory period.

        :return: The finished segment, trimmed or zero-padded to the exact length.
        """
        audio = np.concatenate(self._collecting)[: self.segment_samples]
        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))

        self._collecting = None
        self._cooldown_blocks = self.refractory_blocks
        return Segment(audio, self._segment_start, self._segment_level)


def segment_audio(audio: np.ndarray, config: Config) -> list[Segment]:
    """
    Run a recorded clip through the live segmenter.

    The clip is fed in blocks of config.block_size, exactly as the microphone
    would deliver it, so a training segment and a live segment are cut the same
    way.

    :param audio: The clip as a 1D numpy array.
    :param config: Same settings the live pipeline uses.
    :return: One segment per detected event, in order.
    """
    segmenter = StreamSegmenter(config)
    segments = []
    for start in range(0, len(audio), config.block_size):
        segments.extend(segmenter.push(audio[start : start + config.block_size]))
    segments.extend(segmenter.flush())
    return segments


def loudest_segment(audio: np.ndarray, config: Config) -> Segment:
    """
    Cut a segment around the loudest point of a clip, ignoring the threshold.

    Training needs a segment from every labelled clip, including clips the
    detector rejects: a quiet "No Event" recording never crosses onset_rms, yet
    it is exactly the negative example the classifier needs.

    :param audio: The clip as a 1D numpy array.
    :param config: Segment length and pre-roll.
    :return: A segment centred on the peak sample, zero-padded if the clip is
        shorter than one segment.
    """
    if len(audio) == 0:
        return Segment(np.zeros(config.segment_samples, dtype=np.float32), 0, 0.0)

    peak = int(np.argmax(np.abs(audio)))
    start = max(
        0, min(peak - config.preroll_samples, len(audio) - config.segment_samples)
    )
    window = audio[start : start + config.segment_samples]
    if len(window) < config.segment_samples:
        window = np.pad(window, (0, config.segment_samples - len(window)))

    return Segment(window.astype(np.float32), start, rms(window))
