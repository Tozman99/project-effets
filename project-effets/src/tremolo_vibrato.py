import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def apply_effect(
		x: np.ndarray,
		sr: int,
		N: int,
		tremolo_freq: float,
		tremolo_depth: float,
		vibrato_freq: float,
		vibrato_depth_ms: float
) -> np.ndarray:
    
    # Tremolo
    duration = N / sr
    time = np.linspace(0, duration, N, endpoint=False)
    lfo_trem = np.sin(2 * np.pi * tremolo_freq * time)
    tremolo_gain = (1 - tremolo_depth) + tremolo_depth * (lfo_trem + 1) / 2.0
    audio_trem = x * tremolo_gain.astype(np.float32)
    
    #Vibrato
    vibrato_depth_sec = vibrato_depth_ms / 1000.0
    lfo_vib = np.sin(2 * np.pi * vibrato_freq * time)
    delay_seconds = vibrato_depth_sec * lfo_vib
   
    delay_samples = delay_seconds * sr

    indices = np.arange(N, dtype=np.float32) + delay_samples.astype(np.float32)
    audio_vibrato = np.empty(N, dtype=np.float32)

    idx_floor = np.floor(indices).astype(np.int32)
    idx_ceil = idx_floor + 1
    idx_floor[idx_floor < 0] = 0
    idx_ceil[idx_ceil < 0] = 0
    idx_floor[idx_floor > N - 1] = N - 1
    idx_ceil[idx_ceil > N - 1] = N - 1

    frac = indices - idx_floor.astype(np.float32)
    audio_vibrato = audio_trem[idx_floor] * (1 - frac) + audio_trem[idx_ceil] * frac

    return audio_vibrato