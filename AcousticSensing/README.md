# Acoustic Sensing

Recognises short sounds (a clap, a snap) from a live microphone and reacts to
them. Built for the IIS acoustic sensing lab.

## The pipeline

One module per stage. Each stage takes the output of the one above it and knows
nothing about the rest, so a stage can be read, changed or tested on its own.

| Stage | Module | Output |
| --- | --- | --- |
| Microphone | `audio.py` | Blocks of audio on a queue |
| Event detection / segmentation | `segmentation.py` | A fixed-length `Segment` per event |
| Feature extraction | `features.py` | One array per named feature |
| Classifier | `classifier.py` | — |
| Class + confidence | `classifier.py` | A `Prediction` |
| Display | `visuals.py`, `sketch.py` | The analysis window, or the p5 looper |

`pipeline.py` wires them together; `config.py` holds the settings they share.

The pivot of the design is segmentation. Because it emits one fixed-length
segment per event, everything after it runs a few times a second instead of
fifty, on audio that is known to contain something. Features therefore have a
fixed length and can be as expensive as they need to be, and the classifier is
never asked to name silence.

The same segmenter runs live and over recorded clips, so a training example and
a live event are cut the same way.

## Running it

```sh
uv run acousticsensing                     # live display
uv run acousticsensing --mode sketch       # p5 looper driven by the pipeline
uv run acousticsensing --list-devices      # find a microphone
uv run acousticsensing --device 8          # use a specific one
```

Keys in the display window:

| Key | Action |
| --- | --- |
| `0`–`2` | Choose the class to record |
| `r` | Start or stop recording a clip |
| `u` | Delete the clip just saved |
| `f` | Show the features of the last event |
| `q` | Quit |

## Training

Record a few clips of each class with `r`, then:

```sh
uv run acousticsensing-train
```

It cuts every clip into segments, extracts features, scores the model on clips it
was not trained on, and writes `svm_clap_snap_model.pkl`. The app picks that file
up on the next start.

Accuracy is reported over held-out **clips**, never held-out segments: segments
from one clap are nearly identical, so scoring one with a model that saw its clip
reports a number the live system will never reach.

## Extending it

- **A new sound class**: add it to `CLASS_LABELS` in `config.py`, record clips,
  retrain. Its key, its colour and its row in the report appear on their own.
- **A new feature**: add a `Feature` to `FEATURES` in `features.py`, and its name
  to `DEFAULT_FEATURES` to let the classifier use it. It gets a panel in the
  feature window for free. Models trained earlier keep working, because a model
  records the features it was trained on.
- **A new reaction**: handle it in `Sketch.on_event` in `sketch.py`, which
  receives every classified event.
