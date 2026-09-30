"""
Train and evaluate the classifier on the recorded clips.

Run it with `uv run acousticsensing-train`. Clips live in training_data/, one
directory per class, written by pressing 'r' in the live display.

Evaluation holds out whole clips, never single segments. Segments from one clap
are near-identical, so scoring a segment with a model that saw its clip reports
an accuracy the live system will not reach. Augmented copies stay on the
training side for the same reason.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from acousticsensing.classifier import EventClassifier
from acousticsensing.config import CLASS_LABELS, MODEL_PATH, Config
from acousticsensing.dataset import DATA_DIR, Dataset, build_dataset, load_clips
from acousticsensing.features import DEFAULT_FEATURES, FeatureExtractor


def describe(data: Dataset) -> None:
    """
    Print what the classifier is about to be fed.

    :param data: The dataset from build_dataset.
    """
    real = ~data.augmented
    print(
        f"\n{real.sum()} segments from {len(set(data.groups))} clips, plus "
        f"{data.augmented.sum()} augmented copies, {data.X.shape[1]} features each"
    )
    for label in sorted(set(data.y)):
        of_class = real & (data.y == label)
        clips = len(set(data.groups[of_class]))
        name = CLASS_LABELS.get(label, str(label))
        print(f"  {name:>10}: {of_class.sum():>4} segments from {clips} clips")

    print(f"\nFeatures: {', '.join(data.feature_names)}")


def cross_validate(
    data: Dataset, kernel: str = "linear", C: float = 1.0, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score the model on clips it was not trained on.

    :param data: The dataset from build_dataset.
    :param kernel: SVM kernel.
    :param C: SVM regularization parameter.
    :param n_splits: Folds to use, capped at the smallest number of clips in any
        class.
    :return: The true labels and the predicted labels of every held-out segment.
    :raises ValueError: If a class has fewer than two clips, leaving no way to
        hold one out.
    """
    clips_per_class = Counter(data.y[np.unique(data.groups, return_index=True)[1]])
    usable_splits = min(n_splits, min(clips_per_class.values()))
    if usable_splits < 2:
        thin = [
            CLASS_LABELS.get(label, label)
            for label, count in clips_per_class.items()
            if count < 2
        ]
        raise ValueError(
            f"Need at least 2 clips per class to hold one out; only 1 for: {thin}. "
            "Record more clips of that class."
        )

    splitter = StratifiedGroupKFold(
        n_splits=usable_splits, shuffle=True, random_state=0
    )
    true_labels, predictions = [], []

    for train_rows, test_rows in splitter.split(data.X, data.y, data.groups):
        # Score only real recordings; augmented copies belong to training
        test_rows = test_rows[~data.augmented[test_rows]]
        if not len(train_rows) or not len(test_rows):
            continue

        model = EventClassifier.train(
            data.X[train_rows],
            data.y[train_rows],
            feature_names=data.feature_names,
            kernel=kernel,
            C=C,
        )
        true_labels.append(data.y[test_rows])
        predictions.append([model.predict(row).label for row in data.X[test_rows]])

    return np.concatenate(true_labels), np.concatenate(predictions)


def report(true_labels: np.ndarray, predictions: np.ndarray) -> None:
    """
    Print accuracy, a per-class report and a confusion matrix.

    :param true_labels: True labels of the held-out segments.
    :param predictions: Predicted labels of the same segments.
    """
    present = sorted(set(true_labels) | set(predictions))
    names = [CLASS_LABELS.get(label, str(label)) for label in present]

    print(f"\nHeld-out accuracy: {(true_labels == predictions).mean():.3f}\n")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=present,
            target_names=names,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(true_labels, predictions, labels=present)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(" " * 12 + "  ".join(f"{name:>10}" for name in names))
    for name, row in zip(names, matrix):
        print(f"{name:>10}  " + "  ".join(f"{count:>10d}" for count in row))


def main() -> None:
    """Build the dataset, evaluate it, then save a model trained on all of it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--sample-rate", type=int, default=Config.sample_rate)
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Feature names to train on; see features.FEATURES.",
    )
    parser.add_argument("--kernel", default="linear")
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--no-augment", action="store_true", help="Train on the recorded segments only."
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="Skip cross-validation and only fit."
    )
    args = parser.parse_args()

    config = Config(sample_rate=args.sample_rate)
    extractor = FeatureExtractor(config, args.features)

    data = build_dataset(
        load_clips(args.data_dir),
        config,
        extractor,
        with_augmentation=not args.no_augment,
    )
    describe(data)

    if not args.no_eval:
        report(*cross_validate(data, args.kernel, args.C, args.folds))

    model = EventClassifier.train(
        data.X, data.y, feature_names=data.feature_names, kernel=args.kernel, C=args.C
    )
    model.save(args.model_path)
    print(f"\nSaved a model trained on {len(data.X)} segments to {args.model_path}")


if __name__ == "__main__":
    main()
