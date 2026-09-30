"""
Stage 4: classification, producing a class and a confidence.

The classifier is a scikit-learn pipeline of two steps:

1. StandardScaler, which normalises every feature to zero mean and unit
   variance. This is not optional: the vector mixes MFCCs (tens), a spectral
   centroid in Hz (thousands) and a crest factor (single digits), and without
   scaling the centroid alone would decide every prediction.
2. A calibrated SVC, the support vector machine itself wrapped in the
   calibration that turns its decision scores into probabilities.

A saved model carries the feature names and the class names it was trained
with, so a mismatch is caught with a clear message instead of producing
confident nonsense.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from acousticsensing.config import CLASS_LABELS


@dataclass(frozen=True)
class Prediction:
    """
    What the classifier says about one segment.

    :param label: The predicted class integer.
    :param name: The class name the model was trained with.
    :param confidence: Probability of the predicted class, 0 to 1. The
        displays use it to ignore uncertain events.
    :param probabilities: Probability of every class, keyed by label.
    """

    label: int
    name: str
    confidence: float
    probabilities: dict[int, float]

    def __str__(self) -> str:
        return f"{self.name} ({self.confidence:.0%})"


class EventClassifier:
    """
    A trained model plus everything needed to use it safely.

    Wraps the sklearn pipeline rather than exposing it, because a bare pipeline
    predicts integers with no way to tell what they mean or what features it
    expects.
    """

    def __init__(
        self,
        model: Pipeline,
        class_labels: dict[int, str],
        feature_names: tuple[str, ...],
    ):
        """
        :param model: Fitted scaler-and-SVM pipeline.
        :param class_labels: Class names as they were at training time.
        :param feature_names: Names of the features the model was trained on,
            in vector order.
        """
        self.model = model
        self.class_labels = dict(class_labels)
        self.feature_names = tuple(feature_names)

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...],
        class_labels: dict[int, str] = CLASS_LABELS,
        kernel: str = "linear",
        C: float = 1.0,
    ) -> EventClassifier:
        """
        Fit a normalising SVM on the given feature vectors.

        :param X: Feature matrix of shape (n_segments, n_features).
        :param y: Class label of each row.
        :param feature_names: Names of the features that built X, in vector order.
        :param class_labels: Class names to record with the model.
        :param kernel: SVM kernel.
        :param C: SVM regularization parameter. Lower values tolerate more
            training errors in exchange for a simpler boundary.
        :return: The trained classifier.
        """
        # An SVM scores classes but does not give probabilities, and the
        # displays need a confidence to threshold on. Calibration fits
        # that mapping on held-out folds of the training data. ensemble=False
        # keeps one model rather than one per fold, so prediction stays as fast
        # as a bare SVM.
        model = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(kernel=kernel, C=C), ensemble=False),
        )
        model.fit(X, y)
        return cls(model, class_labels, feature_names)

    def predict(self, vector: np.ndarray) -> Prediction:
        """
        Classify one feature vector.

        :param vector: One segment's features, as built by FeatureExtractor.
        :return: The most probable class and how probable it is.
        :raises ValueError: If the vector is not the length the model expects.
        """
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        expected = self.model.n_features_in_
        if vector.shape[1] != expected:
            raise ValueError(
                f"Model expects {expected} features but got {vector.shape[1]}. "
                f"It was trained on {list(self.feature_names)}; retrain it with "
                "`uv run acousticsensing-train`."
            )

        probabilities = {
            int(label): float(probability)
            for label, probability in zip(
                self.model.classes_, self.model.predict_proba(vector)[0]
            )
        }
        # Taken from the probabilities rather than from predict(), so the class
        # shown and the confidence shown always agree
        label = max(probabilities, key=probabilities.get)

        return Prediction(
            label=label,
            name=self.class_labels.get(label, str(label)),
            confidence=probabilities[label],
            probabilities=probabilities,
        )

    def save(self, path: str | Path) -> None:
        """
        Write the classifier to disk.

        :param path: File to write. Overwritten if it exists.
        """
        joblib.dump(self, Path(path), compress=3)

    @classmethod
    def load(cls, path: str | Path) -> EventClassifier:
        """
        Load a classifier saved by save().

        :param path: File to read.
        :return: The loaded classifier.
        :raises TypeError: If the file holds something else, which usually means
            a model from an older version of this package.
        """
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(
                f"{path} does not hold an EventClassifier but a "
                f"{type(loaded).__name__}. Retrain it with "
                "`uv run acousticsensing-train`."
            )
        return loaded
