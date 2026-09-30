"""
Stage 3: feature extraction.

A segment is always the same length, so every feature here returns a fixed
number of values and they can be concatenated into one vector. Features are
computed once per detected event rather than once per audio block, which is a
few times a second at most, so cost is not a design constraint: pick the
features that describe the sound best.

Adding a feature means adding one entry to FEATURES and, if it should be used
for classification, one name to DEFAULT_FEATURES. Any model trained before that
change keeps working, because a model records the feature set it was trained on
(see classifier.EventClassifier).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import librosa
import numpy as np
from matplotlib.axes import Axes

from acousticsensing.config import Config

# Magnitude floor so log10 and division never see zero
EPSILON = 1e-10


@dataclass(frozen=True)
class Feature:
    """
    One named feature: how to compute it, and how to draw it.

    :param name: Key in the registry and prefix of the vector element labels.
    :param title: Human-readable name, used for plot titles.
    :param compute: Takes segment audio and the config, returns a 1D array whose
        length depends only on the config.
    :param plot: Draws one computed feature onto a matplotlib axis.
    """

    name: str
    title: str
    compute: Callable[[np.ndarray, Config], np.ndarray]
    plot: Callable[[Axes, np.ndarray, Config], None]


# -------------------------------
# Feature computation
# -------------------------------


def _frame_args(config: Config) -> dict:
    """Frame size and hop shared by every framed feature."""
    return {"n_fft": config.fft_size, "hop_length": config.hop_size}


def _mfcc(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    Mean of each MFCC across the segment: the timbre of the event.

    Averaging over frames drops when things happened and keeps what the event
    sounded like, which is what separates a clap from a snap.
    """
    mfccs = librosa.feature.mfcc(
        y=audio, sr=config.sample_rate, n_mfcc=config.n_mfcc, **_frame_args(config)
    )
    return mfccs.mean(axis=1)


def _log_mel(audio: np.ndarray, config: Config) -> np.ndarray:
    """Mean log-mel band energy, in dB relative to the segment's peak."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.sample_rate, n_mels=config.n_mels, **_frame_args(config)
    )
    return librosa.power_to_db(mel, ref=np.max).mean(axis=1)


def _spectral_shape(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    Where the energy sits in the spectrum: centroid, bandwidth, rolloff, flatness.

    Four numbers that summarise brightness and noisiness. A snap is brighter and
    flatter than a clap, and these say so directly rather than through 13 MFCCs.
    """
    framed = {"y": audio, "sr": config.sample_rate, **_frame_args(config)}
    return np.array(
        [
            librosa.feature.spectral_centroid(**framed).mean(),
            librosa.feature.spectral_bandwidth(**framed).mean(),
            librosa.feature.spectral_rolloff(**framed).mean(),
            librosa.feature.spectral_flatness(y=audio, **_frame_args(config)).mean(),
        ]
    )


def _band_energy(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    Share of the segment's energy in each of config.n_bands log-spaced bands.

    Normalising by the total makes the feature independent of how loud the event
    was and how far away it happened, so what is left is the balance between low
    and high frequencies: a clap is bottom-heavy where a snap is not.
    """
    spectrum = np.abs(
        librosa.stft(audio, n_fft=config.fft_size, hop_length=config.hop_size)
    )
    power = np.square(spectrum, dtype=np.float64).mean(axis=1)
    frequencies = librosa.fft_frequencies(sr=config.sample_rate, n_fft=config.fft_size)

    edges = config.band_edges
    bands = np.array(
        [
            # The top edge is Nyquist, so the last band takes it rather than
            # dropping the final bin
            power[(frequencies >= low) & (frequencies < high)].sum()
            if index < len(edges) - 2
            else power[(frequencies >= low) & (frequencies <= high)].sum()
            for index, (low, high) in enumerate(pairwise(edges))
        ]
    )
    return bands / (bands.sum() + EPSILON)


def _envelope(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    How long the event takes to reach its peak, and how long it lasts.

    Both are read off a fine-grained RMS envelope and measured from the first
    frame above config.envelope_floor of the peak: attack time runs from there
    to the peak, duration to the last frame still above it. A snap reaches its
    peak almost immediately and dies; a clap does neither.
    """
    envelope = librosa.feature.rms(
        y=audio,
        frame_length=config.envelope_frame_size,
        hop_length=config.envelope_hop_size,
    )[0]

    peak = float(envelope.max())
    if peak < EPSILON:
        # Digital silence has no attack and no duration to measure
        return np.zeros(2)

    active = np.flatnonzero(envelope >= config.envelope_floor * peak)
    start = int(active[0])

    attack = (int(np.argmax(envelope)) - start) * config.envelope_frame_seconds
    duration = (int(active[-1]) - start + 1) * config.envelope_frame_seconds
    return np.array([attack, duration])


def _temporal(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    The shape of the event in time: level, zero crossings, crest, peak position.

    The crest factor (peak over RMS) is high for a sharp transient and low for a
    sustained sound, and the peak position says whether the segment holds an
    attack or only a decay.
    """
    level = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))

    zero_crossings = librosa.feature.zero_crossing_rate(
        audio, frame_length=config.fft_size, hop_length=config.hop_size
    )
    energy = librosa.feature.rms(
        y=audio, frame_length=config.fft_size, hop_length=config.hop_size
    )[0]
    peak_position = float(np.argmax(energy)) / max(1, len(energy) - 1)

    crest = peak / (level + EPSILON)
    return np.array([level, zero_crossings.mean(), crest, peak_position])


# -------------------------------
# Feature visualization
# -------------------------------


def _plot_mfcc(ax: Axes, values: np.ndarray, config: Config) -> None:
    ax.bar(range(len(values)), values)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Mean value")


def _plot_log_mel(ax: Axes, values: np.ndarray, config: Config) -> None:
    frequencies = librosa.mel_frequencies(n_mels=len(values), fmax=config.nyquist)
    ax.plot(frequencies, values, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")


def _plot_spectral_shape(ax: Axes, values: np.ndarray, config: Config) -> None:
    names = ["centroid", "bandwidth", "rolloff", "flatness"]
    # Flatness is a ratio and the rest are in Hz, so a shared axis would hide it
    ax.bar(names[:3], values[:3])
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectral shape (flatness {values[3]:.4f})")


def _plot_band_energy(ax: Axes, values: np.ndarray, config: Config) -> None:
    edges = config.band_edges
    names = [f"{low:.0f}-{high:.0f}" for low, high in pairwise(edges)]
    ax.bar(names, values)
    ax.set_xlabel("Band (Hz)")
    ax.set_ylabel("Share of energy")
    ax.tick_params(axis="x", labelrotation=45, labelsize="small")


def _plot_envelope(ax: Axes, values: np.ndarray, config: Config) -> None:
    names = ["attack", "duration"]
    # Milliseconds, because both are a fraction of a 0.15 s segment
    ax.barh(names, values * 1000)
    ax.set_xlabel("Time (ms)")


def _plot_temporal(ax: Axes, values: np.ndarray, config: Config) -> None:
    names = ["RMS", "zero crossings", "crest", "peak position"]
    ax.barh(names, values)
    ax.set_xscale("log")
    ax.set_xlabel("Value (log scale)")


# Every feature the package knows how to compute, keyed by name. Insertion order
# fixes the order of the feature vector; a model stores the names it was trained
# on, so reordering or removing an entry retires old models rather than
# corrupting them.
FEATURES: dict[str, Feature] = {
    feature.name: feature
    for feature in (
        Feature("mfcc", "MFCCs", _mfcc, _plot_mfcc),
        Feature(
            "spectral_shape", "Spectral shape", _spectral_shape, _plot_spectral_shape
        ),
        Feature("temporal", "Temporal shape", _temporal, _plot_temporal),
        Feature("band_energy", "Band energies", _band_energy, _plot_band_energy),
        Feature("envelope", "Attack and duration", _envelope, _plot_envelope),
        Feature("log_mel", "Log-mel spectrum", _log_mel, _plot_log_mel),
    )
}

# The features the classifier uses by default. log_mel is left out because MFCCs
# are derived from the same mel spectrum: including both adds 40 correlated
# dimensions without adding information. It stays registered so it can be
# switched on for comparison, and so the feature window can draw it.
DEFAULT_FEATURES = (
    "mfcc",
    "spectral_shape",
    "temporal",
    "band_energy",
    "envelope",
)


class FeatureExtractor:
    """
    Turns one segment into one feature vector.

    Holds the config and the chosen feature names so that training and live
    classification cannot accidentally use different ones.
    """

    def __init__(self, config: Config, names: Sequence[str] = DEFAULT_FEATURES):
        """
        :param config: Settings the features are computed under.
        :param names: Names of the features to extract, in vector order.
        :raises ValueError: If a name is not registered in FEATURES.
        """
        unknown = [name for name in names if name not in FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown feature(s): {', '.join(unknown)}. "
                f"Registered features: {', '.join(FEATURES)}"
            )

        self.config = config
        self.names = tuple(names)

    def extract(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute each feature of one segment, keyed by name.

        :param audio: Segment audio as a 1D numpy array.
        :return: One 1D array per selected feature, in vector order.
        """
        audio = np.asarray(audio, dtype=np.float32)
        return {name: FEATURES[name].compute(audio, self.config) for name in self.names}

    def vector(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute the feature vector of one segment.

        :param audio: Segment audio as a 1D numpy array.
        :return: The features concatenated into one 1D float32 array.
        """
        return self.flatten(self.extract(audio))

    @staticmethod
    def flatten(groups: dict[str, np.ndarray]) -> np.ndarray:
        """
        Join already-extracted features into the vector.

        Separate from extract so a caller that wants the features for display
        does not pay to compute them twice.

        :param groups: Features from extract.
        :return: The features concatenated into one 1D float32 array.
        """
        return np.concatenate(list(groups.values())).astype(np.float32)

    def element_names(self) -> list[str]:
        """
        Name every element of the vector, e.g. "mfcc[3]".

        Reading a linear model's coefficients needs this: it says which feature
        each weight belongs to.

        :return: One label per element, in vector order.
        """
        silence = np.zeros(self.config.segment_samples, dtype=np.float32)
        return [
            f"{name}[{index}]"
            for name, values in self.extract(silence).items()
            for index in range(len(values))
        ]


def spectrum_db(audio: np.ndarray, config: Config) -> np.ndarray:
    """
    Magnitude spectrum of the newest fft_size samples, in dB.

    Used by the live display rather than by the classifier: it is computed on
    every audio block to draw the spectrogram, whereas the features above are
    computed only on detected events.

    :param audio: Audio as a 1D numpy array. Shorter input than fft_size is
        zero-padded at the front; longer input keeps the newest samples.
    :param config: FFT size and sample rate.
    :return: One magnitude in dB per frequency bin of frequency_bins(config).
    """
    block = np.zeros(config.fft_size, dtype=np.float32)
    newest = np.ravel(audio)[-config.fft_size :]
    block[config.fft_size - len(newest) :] = newest

    # Remove the DC offset so bin 0 does not dominate, and window the frame so a
    # tone between two bins does not smear across the whole spectrum
    window = np.hanning(config.fft_size)
    spectrum = np.fft.rfft((block - block.mean()) * window)

    magnitude = np.abs(spectrum) / window.sum()
    return 20 * np.log10(np.maximum(magnitude, EPSILON))


def frequency_bins(config: Config) -> np.ndarray:
    """
    Centre frequency of each bin returned by spectrum_db.

    :param config: FFT size and sample rate.
    :return: The bin frequencies in Hz.
    """
    return np.fft.rfftfreq(config.fft_size, d=1 / config.sample_rate)
