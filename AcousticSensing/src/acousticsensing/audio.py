"""
Stage 1: microphone.

Turns the sound card into a queue of audio blocks. Nothing else in the package
talks to sounddevice, so the rest of the pipeline can be fed from a file or a
test just as easily as from a microphone.
"""

import queue

import numpy as np
import sounddevice as sd

from acousticsensing.config import Config


def describe_input_devices() -> str:
    """
    List the audio devices, for choosing one with --device.

    :return: The device table as printable text.
    """
    return str(sd.query_devices())


def open_microphone(
    config: Config, blocks: queue.Queue, device: int | str | None = None
) -> sd.InputStream:
    """
    Open the microphone and start pushing blocks onto a queue.

    Blocks are dropped rather than queued without limit: if the display falls
    behind, the newest audio matters and the backlog does not.

    :param config: Sample rate and block size to open the stream with.
    :param blocks: Queue the audio callback writes 1D float32 blocks to.
    :param device: Device index or name. None uses the system default input.
    :return: The started stream. The caller closes it.
    :raises RuntimeError: If the device cannot be opened at this sample rate,
        with the device list included in the message.
    """

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"Audio status: {status}")

        block = indata[:, 0].copy()
        try:
            blocks.put_nowait(block)
        except queue.Full:
            # Drop the oldest block so the newest audio still gets through
            try:
                blocks.get_nowait()
                blocks.put_nowait(block)
            except queue.Empty:
                pass

    try:
        stream = sd.InputStream(
            samplerate=config.sample_rate,
            blocksize=config.block_size,
            dtype="float32",
            channels=1,
            device=device,
            callback=callback,
            latency="low",
        )
        stream.start()
    except Exception as error:
        raise RuntimeError(
            f"Could not open input device {device!r} at {config.sample_rate} Hz: "
            f"{error}\n\nAvailable devices:\n{describe_input_devices()}\n"
            "Pick one with --device <index>."
        ) from error

    return stream


def play(audio: np.ndarray, config: Config) -> None:
    """
    Play audio on the default output device without waiting for it to finish.

    A new call cuts off whatever the previous one is still playing, so mix
    sounds that should overlap into one array first.

    :param audio: 1D float audio, -1 to 1.
    :param config: Sample rate to play at.
    """
    sd.play(audio, config.sample_rate)


def drain(blocks: queue.Queue) -> list[np.ndarray]:
    """
    Take every block waiting in the queue.

    :param blocks: Queue written by the audio callback.
    :return: The queued blocks, oldest first. Empty if none are waiting.
    """
    out = []
    while True:
        try:
            out.append(blocks.get_nowait())
        except queue.Empty:
            return out
