"""
Command line entry point: builds the pipeline and runs the live display.

Run it with `uv run acousticsensing`. Everything the application does is wiring:
each stage is constructed here and handed to the next, so this file is the place
to see what the running system is made of.
"""

import argparse
import queue
import sys
from pathlib import Path

from acousticsensing.audio import describe_input_devices, open_microphone
from acousticsensing.classifier import EventClassifier
from acousticsensing.config import MODEL_PATH, Config
from acousticsensing.dataset import DATA_DIR
from acousticsensing.pipeline import AcousticPipeline

WELCOME_MESSAGE = r"""
     __
 .--()°'.'
'|, . ,'
 !_-(_\

Art by: H P Barmario (Morfina)

Welcome to the Acoustic Sensing App!
"""


def load_classifier(path: Path) -> EventClassifier | None:
    """
    Load the trained model, if there is a usable one.

    The app is useful without a model -- that is how training clips get
    recorded -- so a missing or stale model is reported, not fatal.

    :param path: File written by `uv run acousticsensing-train`.
    :return: The classifier, or None.
    """
    if not path.exists():
        print(
            f"No model at {path}; running without classification.\n"
            "Record clips with 'r', then train with: uv run acousticsensing-train"
        )
        return None

    try:
        classifier = EventClassifier.load(path)
    except (TypeError, ValueError) as error:
        print(f"Ignoring {path}: {error}")
        return None

    print(f"Loaded classifier from {path}")
    return classifier


def parse_args() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        help="Input device index or name. Defaults to the system default input.",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="Print the audio devices and exit."
    )
    parser.add_argument(
        "--mode",
        choices=("analysis", "sketch"),
        default="analysis",
        help="'analysis' shows the spectrogram and records training clips; "
        "'sketch' opens a p5 window the pipeline drives.",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sample-rate", type=int, default=Config.sample_rate)
    parser.add_argument(
        "--display-seconds", type=int, default=10, help="Spectrogram history shown."
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=22_050,
        help="Highest frequency drawn, in Hz.",
    )
    return parser.parse_args()


def main() -> None:
    """Wire every stage together and run until the window closes."""
    args = parse_args()

    if args.list_devices:
        print(describe_input_devices())
        return

    print(WELCOME_MESSAGE)
    config = Config(sample_rate=args.sample_rate)

    # Stage 1: microphone
    blocks: queue.Queue = queue.Queue(maxsize=50)
    stream = None

    try:
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        stream = open_microphone(config, blocks, device)

        # Stages 2 to 4, in order
        pipeline = AcousticPipeline(config, classifier=load_classifier(args.model_path))

        # Imported here so each mode only loads its own graphics library
        if args.mode == "sketch":
            from acousticsensing.sketch import Sketch

            Sketch(pipeline, blocks).start()
        else:
            from acousticsensing.visuals import Visualizer

            Visualizer(
                pipeline,
                blocks,
                display_seconds=args.display_seconds,
                max_frequency=args.max_frequency,
                data_dir=args.data_dir,
            ).start()

    except RuntimeError as error:
        print(error)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting the Acoustic Sensing App. Goodbye!")
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
