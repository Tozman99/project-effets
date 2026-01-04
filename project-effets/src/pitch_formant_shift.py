"""
Effet Pitch Shifting + Formant Shifting

Modification de la hauteur (pitch) d'un signal audio sans changer sa durée.
Option de préservation ou modification des formants.

Méthode: Phase Vocoder (STFT) avec resampling.
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def apply_effect(
    x: np.ndarray,
    sr: int,
    pitch_semitones: float,
    formant_semitones: float,
    preserve_formants: bool,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Applique le Pitch Shifting avec option de modification des formants.
    
    Paramètres:
    -----------
    x : np.ndarray
        Signal d'entrée (float32, -1 à 1)
    sr : int
        Fréquence d'échantillonnage
    pitch_semitones : float
        Décalage de pitch en demi-tons
        - +12 = une octave plus haut
        - -12 = une octave plus bas
    formant_semitones : float
        Décalage des formants en demi-tons (indépendant du pitch)
        - 0 = formants préservés
        - Positif = voix plus "petite"
        - Négatif = voix plus "grosse"
    preserve_formants : bool
        Si True: tente de préserver les formants (voix naturelle)
        Si False: pitch shift simple (effet "chipmunk")
    n_fft : int
        Taille de la FFT
    hop_length : int
        Pas entre trames
    
    Retourne:
    ---------
    y : np.ndarray
        Signal avec pitch/formant modifié
    """
    if abs(pitch_semitones) < 0.01 and abs(formant_semitones) < 0.01:
        return x  # Pas de modification
    
    if not preserve_formants:
        return _pitch_shift_simple(x, sr, pitch_semitones, n_fft, hop_length)
    else:
        return _pitch_shift_with_formants(x, sr, pitch_semitones, formant_semitones, n_fft, hop_length)


def _phase_vocoder_stretch(
    x: np.ndarray,
    stretch_factor: float,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Time stretching par Phase Vocoder.
    
    Principe:
    ---------
    1. STFT du signal
    2. Interpolation des trames avec correction de phase
    3. ISTFT
    
    Paramètres:
    -----------
    x : np.ndarray
        Signal d'entrée
    stretch_factor : float
        Facteur d'étirement (>1 = plus long, <1 = plus court)
    """
    window = np.hanning(n_fft)
    
    n_frames = (len(x) - n_fft) // hop_length + 1
    
    if n_frames < 2:
        return x
    
    hop_out = int(hop_length * stretch_factor)
    
    output_length = int(len(x) * stretch_factor) + n_fft
    y = np.zeros(output_length, dtype=np.float32)
    
    phase_acc = np.zeros(n_fft // 2 + 1)
    prev_phase = np.zeros(n_fft // 2 + 1)
    
    omega = 2 * np.pi * np.arange(n_fft // 2 + 1) * hop_length / n_fft
    
    for i in range(n_frames):
        start = i * hop_length
        frame = x[start:start + n_fft]
        
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        phase_diff = phase - prev_phase - omega
        phase_diff = phase_diff - 2 * np.pi * np.round(phase_diff / (2 * np.pi))
        
        inst_freq = omega + phase_diff
        
        phase_acc += inst_freq * (hop_out / hop_length)
        
        spectrum_out = magnitude * np.exp(1j * phase_acc)
        frame_out = np.fft.irfft(spectrum_out, n_fft).real * window
        
        out_start = i * hop_out
        if out_start + n_fft <= len(y):
            y[out_start:out_start + n_fft] += frame_out
        
        prev_phase = phase
    
    y = y * hop_out / (n_fft * 0.5)
    
    return y[:int(len(x) * stretch_factor)]


def _resample(x: np.ndarray, target_length: int) -> np.ndarray:
    """
    Rééchantillonnage par interpolation linéaire.
    Implémenté from scratch.
    """
    if len(x) == target_length:
        return x
    
    old_indices = np.linspace(0, len(x) - 1, len(x))
    new_indices = np.linspace(0, len(x) - 1, target_length)
    
    # Interpolation linéaire
    y = np.interp(new_indices, old_indices, x)
    
    return y.astype(np.float32)


def _pitch_shift_simple(
    x: np.ndarray,
    sr: int,
    semitones: float,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Pitch shifting simple sans préservation des formants.
    
    Résultat: effet "chipmunk" si pitch monté, "monstre" si baissé.
    
    Méthode:
    1. Time stretch (inverse du pitch ratio)
    2. Resample pour revenir à la durée originale
    """
    pitch_ratio = 2 ** (semitones / 12)
    
    stretched = _phase_vocoder_stretch(x, 1.0 / pitch_ratio, n_fft, hop_length)
    
    y = _resample(stretched, len(x))
    
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val * 0.95
    
    return y.astype(np.float32)


def _extract_spectral_envelope(spectrum: np.ndarray, n_coeffs: int = 30) -> np.ndarray:
    """
    Extrait l'enveloppe spectrale par cepstre (liftering).
    
    L'enveloppe représente les formants (résonances du conduit vocal).
    """
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    
    cepstrum = np.fft.irfft(log_spectrum)
    
    liftered = np.zeros_like(cepstrum)
    liftered[:n_coeffs] = cepstrum[:n_coeffs]
    if len(cepstrum) > n_coeffs:
        liftered[-n_coeffs+1:] = cepstrum[-n_coeffs+1:]
    
    log_envelope = np.fft.rfft(liftered, len(spectrum) * 2 - 2).real
    envelope = np.exp(log_envelope[:len(spectrum)])
    
    return envelope


def _pitch_shift_with_formants(
    x: np.ndarray,
    sr: int,
    pitch_semitones: float,
    formant_semitones: float,
    n_fft: int,
    hop_length: int
) -> np.ndarray:
    """
    Pitch shifting avec préservation/modification des formants.
    
    Principe:
    ---------
    1. Extraire l'enveloppe spectrale (formants)
    2. Décaler les formants pour compenser le pitch shift
    3. Appliquer le pitch shift
    """
    pitch_ratio = 2 ** (pitch_semitones / 12)
    formant_ratio = 2 ** (formant_semitones / 12)
    
    formant_shift_ratio = formant_ratio / pitch_ratio
    
    window = np.hanning(n_fft)
    
    n_frames = (len(x) - n_fft) // hop_length + 1
    
    if n_frames < 2:
        return _pitch_shift_simple(x, sr, pitch_semitones, n_fft, hop_length)
    
    stretched = _phase_vocoder_stretch(x, 1.0 / pitch_ratio, n_fft, hop_length)
    
    n_frames_s = (len(stretched) - n_fft) // hop_length + 1
    y = np.zeros(len(stretched), dtype=np.float32)
    window_sum = np.zeros(len(stretched), dtype=np.float32)
    
    for i in range(n_frames_s):
        start = i * hop_length
        frame = stretched[start:start + n_fft]
        
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        envelope = _extract_spectral_envelope(spectrum, 40)
        
        fine_structure = magnitude / (envelope + 1e-10)
        
        n_bins = len(envelope)
        old_bins = np.arange(n_bins)
        new_bins = old_bins * formant_shift_ratio
        
        shifted_envelope = np.interp(new_bins, old_bins, envelope,
                                     left=envelope[0], right=envelope[-1])
        
        new_magnitude = fine_structure * shifted_envelope
        
        spectrum_out = new_magnitude * np.exp(1j * phase)
        frame_out = np.fft.irfft(spectrum_out, n_fft).real * window
        
        end = min(start + n_fft, len(y))
        y[start:end] += frame_out[:end - start]
        window_sum[start:end] += window[:end - start] ** 2
    
    window_sum = np.maximum(window_sum, 1e-10)
    y = y / window_sum
    
    y = _resample(y, len(x))
    
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val * 0.95
    
    return y.astype(np.float32)
