"""
Effet Chorus / Flanger

Effets basés sur des lignes de délai modulées par un LFO.
- Chorus: délais 15-30ms, plusieurs voix → épaisseur
- Flanger: délais 1-10ms, feedback → effet "jet" métallique
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def apply_effect(
    x: np.ndarray,
    sr: int,
    effect_type: str,
    num_voices: int,
    base_delay_ms: float,
    mod_depth_ms: float,
    mod_freq: float,
    feedback: float,
    mix: float
) -> np.ndarray:
    """
    Applique l'effet Chorus ou Flanger.
    
    Paramètres:
    -----------
    x : np.ndarray
        Signal d'entrée (float32, -1 à 1)
    sr : int
        Fréquence d'échantillonnage
    effect_type : str
        'chorus' ou 'flanger'
    num_voices : int
        Nombre de voix pour le chorus (2-5)
    base_delay_ms : float
        Délai de base en ms (chorus: 15-30ms, flanger: 1-10ms)
    mod_depth_ms : float
        Profondeur de modulation en ms
    mod_freq : float
        Fréquence du LFO en Hz
    feedback : float
        Feedback pour flanger (0-0.9)
    mix : float
        Ratio wet/dry (0=dry, 1=wet)
    
    Retourne:
    ---------
    y : np.ndarray
        Signal avec effet appliqué
    """
    N = len(x)
    
    if effect_type == 'chorus':
        return _apply_chorus(x, sr, N, num_voices, base_delay_ms, mod_depth_ms, mod_freq, mix)
    elif effect_type == 'flanger':
        return _apply_flanger(x, sr, N, base_delay_ms, mod_depth_ms, mod_freq, feedback, mix)
    else:
        raise ValueError(f"Type inconnu: {effect_type}. Utiliser 'chorus' ou 'flanger'.")


def _apply_chorus(
    x: np.ndarray,
    sr: int,
    N: int,
    num_voices: int,
    base_delay_ms: float,
    mod_depth_ms: float,
    mod_freq: float,
    mix: float
) -> np.ndarray:
    """
    Applique l'effet Chorus.
    
    Principe:
    ---------
    Plusieurs lignes de délai modulées avec phases différentes
    créent une illusion de plusieurs instruments jouant ensemble.
    
    Équation:
        y(t) = (1-mix)*x(t) + (mix/N) * Σ x(t - τᵢ(t))
        τᵢ(t) = delay_base + depth * sin(2πf*t + φᵢ)
    """
    base_delay_samples = base_delay_ms * sr / 1000.0
    mod_depth_samples = mod_depth_ms * sr / 1000.0
    
    wet = np.zeros(N, dtype=np.float32)
    t = np.arange(N) / sr
    
    for voice in range(num_voices):
        
        phase = voice * 2 * np.pi / num_voices
        freq_variation = 1.0 + 0.15 * (voice - num_voices / 2) / num_voices
        voice_freq = mod_freq * freq_variation
        
        lfo = np.sin(2 * np.pi * voice_freq * t + phase)        
        delay = base_delay_samples + mod_depth_samples * lfo        
        read_indices = np.arange(N, dtype=np.float64) - delay        
        idx_floor = np.floor(read_indices).astype(np.int32)
        idx_ceil = idx_floor + 1
        frac = (read_indices - idx_floor).astype(np.float32)        
        idx_floor = np.clip(idx_floor, 0, N - 1)
        idx_ceil = np.clip(idx_ceil, 0, N - 1)        
        voice_signal = x[idx_floor] * (1 - frac) + x[idx_ceil] * frac
        
        wet += voice_signal
    
    wet = wet / num_voices
    y = (1 - mix) * x + mix * wet    
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val * 0.95
    
    return y.astype(np.float32)


def _apply_flanger(
    x: np.ndarray,
    sr: int,
    N: int,
    base_delay_ms: float,
    mod_depth_ms: float,
    mod_freq: float,
    feedback: float,
    mix: float
) -> np.ndarray:
    """
    Applique l'effet Flanger.
    
    Principe:
    ---------
    Délai court modulé avec feedback crée un effet de filtre en peigne
    balayant le spectre → son "jet" caractéristique.
    
    Équation:
        y(t) = x(t) + feedback * y(t - τ(t))
        τ(t) = delay_base + depth * sin(2πf*t)
    """
    base_delay_samples = base_delay_ms * sr / 1000.0
    mod_depth_samples = mod_depth_ms * sr / 1000.0    
    y = np.zeros(N, dtype=np.float32)    
    max_delay = int(base_delay_samples + mod_depth_samples + 10)
    delay_buffer = np.zeros(max_delay, dtype=np.float32)
    write_idx = 0    
    feedback = min(feedback, 0.95)    
    
    for n in range(N):

        lfo = np.sin(2 * np.pi * mod_freq * n / sr)
        delay = base_delay_samples + mod_depth_samples * lfo
        read_pos = write_idx - delay
        if read_pos < 0:
            read_pos += max_delay
        
        idx_floor = int(np.floor(read_pos)) % max_delay
        idx_ceil = (idx_floor + 1) % max_delay
        frac = read_pos - np.floor(read_pos)
        
        delayed_sample = delay_buffer[idx_floor] * (1 - frac) + delay_buffer[idx_ceil] * frac
        output = x[n] + feedback * delayed_sample
        delay_buffer[write_idx] = output
        write_idx = (write_idx + 1) % max_delay
        
        y[n] = output

    y_mixed = (1 - mix) * x + mix * y
    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        y_mixed = y_mixed / max_val * 0.95
    
    return y_mixed.astype(np.float32)
