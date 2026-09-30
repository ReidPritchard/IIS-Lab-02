"""
The pipeline: every stage, wired together in one place.

    microphone           audio.open_microphone
      |
    event detection      segmentation.StreamSegmenter
      |
    feature extraction   features.FeatureExtractor
      |
    classifier           classifier.EventClassifier
      |
    class + confidence   classifier.Prediction
      |
    display              visuals.Visualizer or sketch.Sketch

Each stage is an object this class holds, so a stage can be swapped or tested on
its own. Audio goes in one block at a time; what comes out is a list of Events,
one per detected acoustic event, carrying everything the later stages decided
about it. What to do with an event is up to the display that receives it.

A stage left out simply stops the chain there: with no classifier, events are
still detected and their features extracted, which is what recording training
data needs.
"""

from dataclasses import dataclass

import numpy as np

from acousticsensing.classifier import EventClassifier, Prediction
from acousticsensing.config import Config
from acousticsensing.features import DEFAULT_FEATURES, FeatureExtractor
from acousticsensing.segmentation import Segment, StreamSegmenter


@dataclass(frozen=True)
class Event:
    """
    One acoustic event and everything the pipeline decided about it.

    The later fields are None when their stage did not run or chose not to act,
    so an Event also records where the chain stopped.

    :param segment: The audio that was cut out, and where it came from.
    :param features: Extracted features, keyed by feature name.
    :param prediction: Class and confidence, or None with no classifier loaded.
    """

    segment: Segment
    features: dict[str, np.ndarray]
    prediction: Prediction | None = None


class AcousticPipeline:
    """
    Runs continuous audio through every stage and reports the events.

    :ivar errors: Stage failures already reported, so a broken stage is
        mentioned once instead of on every event.
    """

    def __init__(
        self,
        config: Config,
        classifier: EventClassifier | None = None,
    ):
        """
        :param config: Settings every stage shares.
        :param classifier: Trained model, or None to detect events without
            classifying them.
        """
        self.config = config
        self.segmenter = StreamSegmenter(config)
        # Extract what the model was trained on, so the two cannot disagree
        self.extractor = FeatureExtractor(
            config, classifier.feature_names if classifier else DEFAULT_FEATURES
        )
        self.classifier = classifier
        self.errors: list[str] = []

    def push(self, block: np.ndarray) -> list[Event]:
        """
        Feed one block of audio through the pipeline.

        :param block: One block from the microphone.
        :return: One Event per acoustic event finished by this block, usually
            none.
        """
        return [self._handle(segment) for segment in self.segmenter.push(block)]

    def process(self, audio: np.ndarray) -> list[Event]:
        """
        Run a whole recording through the pipeline, block by block.

        Uses the same stages in the same order as the live path, so an offline
        result matches what the microphone would have produced.

        :param audio: The recording as a 1D numpy array.
        :return: One Event per detected acoustic event.
        """
        self.segmenter.reset()
        events = []
        for start in range(0, len(audio), self.config.block_size):
            events.extend(self.push(audio[start : start + self.config.block_size]))
        events.extend(self._handle(segment) for segment in self.segmenter.flush())
        return events

    def _handle(self, segment: Segment) -> Event:
        """
        Carry one segment through feature extraction and classification.

        :param segment: A detected event from the segmenter.
        :return: The finished Event.
        """
        features = self.extractor.extract(segment.audio)

        return Event(segment, features, self._classify(features))

    def _classify(self, features: dict[str, np.ndarray]) -> Prediction | None:
        """
        Classify one event's features, if a working model is loaded.

        :param features: Features from the extractor.
        :return: The prediction, or None with no model or after a failure.
        """
        if self.classifier is None or self.errors:
            return None

        try:
            return self.classifier.predict(FeatureExtractor.flatten(features))
        except ValueError as error:
            # Nearly always a model trained on a different feature set; stop
            # trying rather than raising on every event from here on
            self.errors.append(str(error))
            print(f"Classification disabled: {error}")
            return None
