"""
Acoustic sensing: recognise short sounds and show them.

The package is one module per stage of the pipeline:

    microphone           audio
      |
    event detection      segmentation
      |
    feature extraction   features
      |
    classifier           classifier
      |
    class + confidence   classifier.Prediction
      |
    display              visuals, sketch

pipeline.AcousticPipeline wires them together, config.Config holds the settings
they share, and app.main builds the running application. Training data and the
trainer live in dataset and train.
"""

from acousticsensing.app import main
from acousticsensing.classifier import EventClassifier, Prediction
from acousticsensing.config import CLASS_LABELS, Config
from acousticsensing.pipeline import AcousticPipeline, Event

__all__ = [
    "CLASS_LABELS",
    "AcousticPipeline",
    "Config",
    "Event",
    "EventClassifier",
    "Prediction",
    "main",
]
