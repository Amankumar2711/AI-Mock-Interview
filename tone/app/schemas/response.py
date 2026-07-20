from pydantic import BaseModel
from typing import List, Optional


class PitchData(BaseModel):
    avg_pitch: float
    min_pitch: float
    max_pitch: float
    variation: float
    pitch_score: int


class EnergyData(BaseModel):
    mean_energy: float
    energy_variation: float
    energy_score: int


class PauseData(BaseModel):
    total_pauses: int
    long_pauses: int
    total_pause_time: float
    avg_pause_duration: float
    pause_details: List[float]


class FillerData(BaseModel):
    filler_count: int
    filler_ratio: float
    filler_score: int


class ToneAnalysisResponse(BaseModel):
    duration: float
    pitch: PitchData
    energy: EnergyData
    tone_score: int
    monotone: bool
    pauses: Optional[PauseData] = None
    fillers: Optional[FillerData] = None
    speech_rate_wpm: Optional[float] = None
    communication_score: Optional[int] = None
    feedback: List[str] = []
