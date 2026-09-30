"""
Settings shared by every stage of the pipeline.

One Config instance is built at startup and handed to each stage, so the
microphone, the segmenter, the feature extractor and the training script cannot
drift apart. Features extracted with one sample rate are not comparable with
features extracted at another, and a model is only valid for the Config it was
trained under.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Where the trainer writes the model and the app looks for it
MODEL_PATH = Path("svm_clap_snap_model.pkl")

# The one place classes are defined. The integer is the identity: it is stored in
# every recorded clip and predicted by the model. A trained model keeps its own
# copy of this map, so renaming a class here does not silently rename old models'
# predictions -- it only affects clips recorded and models trained from now on.
CLASS_LABELS = {0: "No Event", 1: "Clap", 2: "Snap", 3: "Knock", 4: "Finger Tap"}


@dataclass(frozen=True)
class Config:
    """
    Every number the signal chain depends on.

    :param sample_rate: Samples per second captured from the microphone.
    :param block_size: Samples per block delivered by the audio callback. Also
        the resolution of onset detection, since the segmenter decides block by
        block.
    :param segment_seconds: Length of the segment cut around each onset. Long
        enough to cover the whole transient, short enough that one clap cannot
        contain two.
    :param preroll_seconds: How much audio before the onset the segment keeps.
        The block that crosses the threshold already contains the attack, but the
        very start of it usually sits in the block before.
    :param onset_rms: Block RMS that counts as an event. Below it the pipeline
        stays idle, so silence never reaches the classifier.
    :param refractory_seconds: Quiet period after a segment during which no new
        onset is accepted, so one physical clap yields one segment.
    :param fft_size: Samples per analysis frame inside a segment.
    :param hop_size: Samples between analysis frames.
    :param n_mfcc: MFCCs kept per frame.
    :param n_mels: Mel bands used by the log-mel feature.
    :param n_bands: Frequency bands the band-energy feature splits the spectrum
        into. The edges are log-spaced, because pitch is.
    :param band_low_hz: Bottom edge of the lowest band. Below it is rumble and
        mains hum rather than anything a clap or a snap produces.
    :param envelope_frame_size: Samples per frame of the amplitude envelope used
        to time the attack. Much shorter than fft_size, because an attack lasts
        a few milliseconds and a 4096-sample frame would swallow it whole.
    :param envelope_hop_size: Samples between envelope frames, and so the
        resolution of attack time and event duration.
    :param envelope_floor: Fraction of the envelope peak that counts as the
        event being under way. Sets where the attack starts and where the event
        is judged to have ended.
    """

    sample_rate: int = 96_000
    block_size: int = 2048

    segment_seconds: float = 0.15
    preroll_seconds: float = 0.02
    onset_rms: float = 0.02
    refractory_seconds: float = 0.25

    fft_size: int = 4096
    hop_size: int = 1024
    n_mfcc: int = 13
    n_mels: int = 40

    n_bands: int = 6
    band_low_hz: float = 50.0

    envelope_frame_size: int = 512
    envelope_hop_size: int = 128
    envelope_floor: float = 0.1

    @property
    def segment_samples(self) -> int:
        """Length of one segment in samples."""
        return int(self.segment_seconds * self.sample_rate)

    @property
    def preroll_samples(self) -> int:
        """Pre-onset audio kept in a segment, in samples."""
        return int(self.preroll_seconds * self.sample_rate)

    @property
    def refractory_samples(self) -> int:
        """Quiet period after a segment, in samples."""
        return int(self.refractory_seconds * self.sample_rate)

    @property
    def nyquist(self) -> float:
        """Highest frequency representable at this sample rate."""
        return self.sample_rate / 2

    @property
    def band_edges(self) -> tuple[float, ...]:
        """
        Edges of the energy bands, in Hz: n_bands + 1 of them.

        Derived from the sample rate rather than written down, so the top band
        always reaches the Nyquist frequency instead of stopping short of it or
        running past it when the rate changes.
        """
        edges = np.geomspace(self.band_low_hz, self.nyquist, self.n_bands + 1)
        return tuple(float(edge) for edge in edges)

    @property
    def envelope_frame_seconds(self) -> float:
        """Time between envelope frames, in seconds."""
        return self.envelope_hop_size / self.sample_rate
