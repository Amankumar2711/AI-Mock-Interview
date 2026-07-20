import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.question_generator import ( QUESTIONS)

from services.question_speaker import ( speak_question)

from services.recorder import ( record_audio)

from services.interview_pipeline import ( run_pipeline)

from services.question_generator import generate_question
from utils.logger import get_logger

logger = get_logger()


def run_interview():
    logger.info("Initializing New Interview Session")
    results = []

    for index, question in enumerate(
            QUESTIONS
    ):

        speak_question(
            question
        )

        filename = (
            f"q{index+1}.wav"
        )

        logger.info(f"Asking Question {index+1}: {question}")
        audio_path = record_audio(
            filename
        )

        logger.info(f"Audio recorded: {audio_path}")
        result = run_pipeline(
            audio_path
        )

        logger.info(f"Pipeline finished for Question {index+1} with score: {result['confidence_data']['final_score']}")
        results.append(
            result
        )

        print("\n--- Detailed Score Breakdown ---")
        print(f"Transcript: \"{result['transcript'].strip()}\"")
        print(f"Total Confidence Score: {result['confidence_data']['final_score']}")
        print(f"Overall Status: {result['recommendation']['status']} - {result['recommendation']['remarks']}")
        
        cd = result['confidence_data']
        ap = cd['audio_penalties']
        tf = result['text_features']

        print("\n[Fluency & Pacing (40% Weight)]")
        print(f"- Words Per Minute (WPM): {cd['wpm']:.2f} (Penalty: -{cd['wpm_penalty']})")
        filler_ratio = cd.get('filler_ratio', 0)
        print(f"- Filler Word Ratio: {filler_ratio:.2%} (Penalty: -{cd['filler_penalty']})")
        print(f"- Pause Length (Unvoiced): {ap['pause_raw']:.2f}s (Penalty: -{ap['pause_penalty']})")

        print("\n[Vocal Stability (40% Weight)]")
        print(f"- Voice Tremor (Jitter): {ap['jitter_raw']:.4f} (Penalty: -{ap['jitter_penalty']})")
        print(f"- Volume Instability (Shimmer): {ap['shimmer_raw']:.4f} (Penalty: -{ap['shimmer_penalty']})")

        print("\n[Linguistic Certainty (20% Weight)]")
        hedge_ratio = cd.get('hedge_ratio', 0)
        print(f"- Hedge Word Ratio: {hedge_ratio:.2%} (Penalty: -{cd['hedge_penalty']})")
        print("--------------------------------\n")

    return results


if __name__ == "__main__":

    run_interview()
