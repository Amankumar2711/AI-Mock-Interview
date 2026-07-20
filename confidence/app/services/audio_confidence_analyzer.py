
def analyze_audio_confidence(audio_features):


    jitter = audio_features.get("jitterLocal_sma3nz_amean", 0)   # Jitter ratio lies between 0.0 to 0.06   
    jitter_penalty = min(20.0, (jitter - 0.02) * 1000) if jitter > 0.02 else 0

    shimmer = audio_features.get("shimmerLocaldB_sma3nz_amean", 0)  # Shimmer lies between 0.0 to 1.5 
    shimmer_penalty = min(20.0, (shimmer - 0.8) * 30) if shimmer > 0.8 else 0

    mean_silence = audio_features.get("MeanUnvoicedSegmentLength", 0)

    pause_penalty = min(13.34, (mean_silence - 0.4) * 30) if mean_silence > 0.4 else 0

    return {
        "jitter_penalty": round(jitter_penalty, 2),
        "shimmer_penalty": round(shimmer_penalty, 2),
        "pause_penalty": round(pause_penalty, 2),
        "jitter_raw": jitter,
        "shimmer_raw": shimmer,
        "pause_raw": mean_silence,
    }

