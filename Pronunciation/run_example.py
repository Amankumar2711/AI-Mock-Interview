"""
Example usage.

    py -3.12 run_example.py path/to/recording.wav "reference text here"

Requires:
    pip install whisperx g2p-en transformers torch torchaudio soundfile

MFA (optional, for true forced alignment):
    conda install -c conda-forge montreal-forced-aligner
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa

If MFA isn't installed, the pipeline still runs — phoneme durations
fall back to proportional estimates instead of true alignment timestamps.
"""

import sys
import json
import soundfile as sf
from pipeline import PronunciationPipeline
from Pronunciation.config_pron.config import get_cpu_config
import logging
from logger_setup import initialize_logger
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting pronunciation pipeline")
    if len(sys.argv) < 3:
        print('Usage: python run_example.py <audio.wav> "<reference text>"')
        sys.exit(1)
    initialize_logger()
    audio_path = sys.argv[1]
    reference_text = sys.argv[2]

    samples, sr = sf.read(audio_path)

    pipeline = PronunciationPipeline(get_cpu_config())
    pipeline.load()

    report = pipeline.evaluate(
        audio=samples,
        sample_rate=sr,
        reference_text=reference_text,
        session_id="demo_session",
    )
    logger.info("Done pronunciation pipeline")
    print("\n=== PRONUNCIATION REPORT ===")
    print(f"Overall score : {report.overall_score}/100")
    print(f"Transcribed   : {report.transcribed_text}")
    print(f"Language      : {report.language_detected}")
    print(f"Warnings      : {report.warnings}")
    print(f"Time          : {report.processing_time_seconds:.2f}s")
    print()
    # print("Word scores:")
    # for ws in report.word_scores:
    #     print(f"  {ws.word:15s} {ws.score:5.1f}  errors={ws.error_count}")
    # print()
    # print("Error summary:")
    # print(json.dumps(report.error_summary, indent=2))
    logger.info("Pronunciation report submitted")

if __name__ == "__main__":
    main()
