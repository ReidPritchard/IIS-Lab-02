"""
Training data: recorded clips on disk, and the feature matrix built from them.

A clip is a raw recording made by pressing 'r' in the live display, stored as
training_data/<class>/<timestamp>.npz. Clips hold audio rather than features, so
changing the feature set only means retraining, not recording again.

Clips become training rows by running them through the same segmenter the live
pipeline uses, so a training row and a live event are cut the same way.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from acousticsensing.config import CLASS_LABELS, Config
from acousticsensing.features import FeatureExtractor
from acousticsensing.segmentation import loudest_segment, segment_audio

DATA_DIR = Path("training_data")

# Copies made of each training segment. Each one is a plausible version of the
# same event: quieter or louder, a little noisier, and starting slightly earlier
# or later than the detector happened to trigger. The shift matters most --
# onsets are found to within one block, so the model must not depend on the event
# sitting at exactly one offset.
AUGMENTATION_GAINS = (0.7, 1.4)
AUGMENTATION_NOISE = 0.004
AUGMENTATION_SHIFTS = (-0.25, 0.25)


@dataclass(frozen=True)
class Clip:
    """
    One recording as it sits on disk.

    :param audio: The raw recording as a 1D numpy array.
    :param label: Class label it was recorded under.
    :param sample_rate: Sample rate it was recorded at.
    :param path: Where it came from, for error messages.
    """

    audio: np.ndarray
    label: int
    sample_rate: int
    path: Path


def save_clip(
    audio: np.ndarray, label: int, config: Config, data_dir: Path = DATA_DIR
) -> Path:
    """
    Write one recording under its class directory.

    :param audio: The recording as a 1D numpy array.
    :param label: Class label to record it under.
    :param config: Used for the sample rate stored with the clip.
    :param data_dir: Root of the training data.
    :return: The path written.
    """
    name = CLASS_LABELS[label].lower().replace(" ", "_")
    directory = data_dir / name
    directory.mkdir(parents=True, exist_ok=True)

    # Local time, so the name matches when the recording was made. astimezone()
    # only makes that explicit; it does not shift the clock.
    path = directory / f"{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}.npz"
    np.savez(path, audio=audio, label=label, sample_rate=config.sample_rate)
    return path


def load_clips(data_dir: Path = DATA_DIR) -> list[Clip]:
    """
    Load every recorded clip.

    :param data_dir: Root of the training data.
    :return: The clips, in a stable order.
    """
    clips = []
    for path in sorted(Path(data_dir).glob("*/*.npz")):
        with np.load(path) as stored:
            clips.append(
                Clip(
                    audio=stored["audio"],
                    label=int(stored["label"]),
                    sample_rate=int(stored["sample_rate"]),
                    path=path,
                )
            )
    return clips


def augment(audio: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """
    Make plausible variants of one training segment.

    :param audio: Segment audio as a 1D numpy array.
    :param rng: Random source for the noise. Seed it once per run so two runs
        over the same clips produce the same model.
    :return: The variants, not including the original.
    """
    variants = [audio * gain for gain in AUGMENTATION_GAINS]
    variants.append(audio + rng.normal(0, AUGMENTATION_NOISE, audio.shape))
    variants += [shift_in_segment(audio, shift) for shift in AUGMENTATION_SHIFTS]
    return [variant.astype(np.float32) for variant in variants]


def shift_in_segment(audio: np.ndarray, fraction: float) -> np.ndarray:
    """
    Move the event within its segment, filling the gap with silence.

    Shifting rather than rotating matters: rotating would wrap the decay of the
    event around to the front, which is a sound no microphone ever produces.

    :param audio: Segment audio as a 1D numpy array.
    :param fraction: How far to move it, as a fraction of the segment length.
        Positive moves it later.
    :return: A segment of the same length.
    """
    offset = int(fraction * len(audio))
    shifted = np.zeros_like(audio)
    if offset >= 0:
        shifted[offset:] = audio[: len(audio) - offset]
    else:
        shifted[:offset] = audio[-offset:]
    return shifted


@dataclass(frozen=True)
class Dataset:
    """
    The segment-level training data.

    :param X: Feature matrix of shape (n_segments, n_features).
    :param y: Class label of each row.
    :param groups: Index of the clip each row came from. Splitting on groups
        keeps a clip and its own augmented copies out of opposite sides of a
        train/test split, where they would flatter the score badly.
    :param augmented: True where the row came from an augmented copy. Those rows
        are trained on but never scored: a noisy copy of a clip is nearly the
        clip, so scoring it measures nothing.
    :param feature_names: Features that built X, in vector order.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    augmented: np.ndarray
    feature_names: tuple[str, ...]


def build_dataset(
    clips: list[Clip],
    config: Config,
    extractor: FeatureExtractor,
    with_augmentation: bool = True,
) -> Dataset:
    """
    Turn recorded clips into a feature matrix.

    Every clip yields at least one segment. A clip the detector finds nothing in
    falls back to its loudest window, which is what makes quiet "No Event" clips
    usable: they never cross the onset threshold, yet they are exactly the
    negative examples the classifier needs.

    :param clips: Clips from load_clips.
    :param config: Settings the live pipeline uses.
    :param extractor: Feature extractor, already configured.
    :param with_augmentation: Whether to add augmented copies of each segment.
    :return: The dataset.
    :raises ValueError: If there are no clips, if a clip was recorded at another
        sample rate, or if no clip belongs to a class that still exists.
    """
    if not clips:
        raise ValueError(
            "No clips found. Record some first: run `uv run acousticsensing`, "
            "press a class key, then 'r' to start and stop recording."
        )

    mismatched = {
        clip.sample_rate for clip in clips if clip.sample_rate != config.sample_rate
    }
    if mismatched:
        raise ValueError(
            f"Clips recorded at {sorted(mismatched)} Hz but the pipeline runs at "
            f"{config.sample_rate} Hz; their features are not comparable."
        )

    rng = np.random.default_rng(0)
    rows, labels, groups, augmented = [], [], [], []

    for group, clip in enumerate(clips):
        # Clips keep the integer they were recorded with, so a class retired from
        # CLASS_LABELS leaves its clips behind. Training on them would bring the
        # class back as a bare number.
        if clip.label not in CLASS_LABELS:
            print(f"Skipping {clip.path}: class {clip.label} no longer exists.")
            continue

        segments = segment_audio(clip.audio, config) or [
            loudest_segment(clip.audio, config)
        ]

        for segment in segments:
            variants = [(segment.audio, False)]
            if with_augmentation:
                variants += [(variant, True) for variant in augment(segment.audio, rng)]

            for audio, is_augmented in variants:
                rows.append(extractor.vector(audio))
                labels.append(clip.label)
                groups.append(group)
                augmented.append(is_augmented)

    if not rows:
        raise ValueError(
            "Nothing to train on: every clip is labelled with a class that is no "
            "longer in CLASS_LABELS."
        )

    return Dataset(
        X=np.array(rows),
        y=np.array(labels),
        groups=np.array(groups),
        augmented=np.array(augmented),
        feature_names=extractor.names,
    )
