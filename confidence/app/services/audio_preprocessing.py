from pydub import AudioSegment
import os

PROCESSED_DIR = "processed_audio"

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


def preprocess_audio(input_path):

    audio = AudioSegment.from_file(
        input_path
    )

    audio = audio.set_channels(1)

    audio = audio.set_frame_rate(
        16000
    )

    audio = audio.set_sample_width(2)

    filename = (
        os.path.splitext(
            os.path.basename(input_path)
        )[0]
        + ".wav"
    )

    output_path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    audio.export(
        output_path,
        format="wav"
    )

    return output_path
