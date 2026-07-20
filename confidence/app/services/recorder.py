import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os

RECORDINGS_DIR = "recordings"

os.makedirs(
    RECORDINGS_DIR,
    exist_ok=True
)


def record_audio(
        filename,
        sample_rate=16000
):

    input(
        "\nPress ENTER to start recording..."
    )

    recording = []

    def callback(
            indata,
            frames,
            time,
            status
    ):
        recording.append(
            indata.copy()
        )

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        callback=callback
    ):

        input(
            "Recording... Press ENTER again to stop.\n"
        )

    if len(recording) == 0:
        audio = np.zeros((sample_rate, 1), dtype=np.int16)
    else:
        audio = np.concatenate(
            recording,
            axis=0
        )

    path = os.path.join(
        RECORDINGS_DIR,
        filename
    )

    write(
        path,
        sample_rate,
        audio
    )

    return path
