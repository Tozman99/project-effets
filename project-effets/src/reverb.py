"""
Effet Réverbération par Convolution

Module implémentant la réverbération par convolution avec une réponse
impulsionnelle synthétique. 

TOUT est implémenté from scratch:
- FFT (Cooley-Tukey radix-2)
- IFFT
- Convolution via FFT

Aucune fonction built-in de convolution ou FFT n'est utilisée.
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def apply_effect(
    x: np.ndarray,
    sr: int,
    rt60: float,
    pre_delay_ms: float,
    wet: float,
    early_reflections: int,
    diffusion: float
) -> np.ndarray:
    """
    Applique la réverbération par convolution.
    
    Paramètres:
    -----------
    x : np.ndarray
        Signal d'entrée (float32, -1 à 1)
    sr : int
        Fréquence d'échantillonnage
    rt60 : float
        Temps de réverbération en secondes (0.3-5.0 typique)
        - 0.3-0.5s: Petite pièce
        - 1.0-2.0s: Salle de concert
        - 3.0-5.0s: Cathédrale
    pre_delay_ms : float
        Délai initial en ms (10-80 typique)
    wet : float
        Ratio wet/dry (0.0=dry, 1.0=100% reverb)
    early_reflections : int
        Nombre de réflexions précoces (4-12)
    diffusion : float
        Densité de la queue (0.0-1.0)
    
    Retourne:
    ---------
    y : np.ndarray
        Signal avec réverbération
    """

    ir = generate_impulse_response(sr, rt60, pre_delay_ms, early_reflections, diffusion)
    reverb_signal = fft_convolve(x, ir)[:len(x)]
    max_reverb = np.max(np.abs(reverb_signal))
    if max_reverb > 0:
        reverb_signal = reverb_signal / max_reverb

    y = (1 - wet) * x + wet * reverb_signal.astype(np.float32)
    
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val * 0.95
    
    return y.astype(np.float32)


# =============================================================================
# FFT COOLEY-TUKEY 
# =============================================================================

def _next_power_of_2(n: int) -> int:
    """
    Retourne la prochaine puissance de 2 >= n.
    """
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _fft_recursive(x: np.ndarray) -> np.ndarray:
    """
    FFT Cooley-Tukey récursive (radix-2 DIT).
    
    Implémentation from scratch de la Transformée de Fourier Rapide.
    
    Algorithme:
    -----------
    1. Diviser le signal en échantillons pairs et impairs
    2. Calculer récursivement la FFT de chaque moitié
    3. Combiner avec les "twiddle factors" (facteurs de rotation)
    
    Équation DFT (ce que la FFT calcule efficacement):
        X[k] = Σ(n=0 to N-1) x[n] * e^(-j*2π*k*n/N)
    
    Complexité: O(N log N) au lieu de O(N²) pour la DFT directe
    
    Paramètres:
    -----------
    x : np.ndarray
        Signal d'entrée (longueur doit être une puissance de 2)
    
    Retourne:
    ---------
    X : np.ndarray
        Spectre complexe
    """
    N = len(x)
    
    if N == 1:
        return x.astype(np.complex128)
    
    if N % 2 != 0:
        raise ValueError("La longueur doit être une puissance de 2")
    
    x_even = x[0::2]  # x[0], x[2], x[4], ...
    x_odd = x[1::2]   # x[1], x[3], x[5], ...
    
    X_even = _fft_recursive(x_even)
    X_odd = _fft_recursive(x_odd)
    
    # W_N^k = e^(-j*2π*k/N)
    X = np.zeros(N, dtype=np.complex128)
    
    for k in range(N // 2):
        angle = -2.0 * np.pi * k / N
        twiddle = np.cos(angle) + 1j * np.sin(angle)
        t = twiddle * X_odd[k]
        X[k] = X_even[k] + t
        X[k + N // 2] = X_even[k] - t
    
    return X


def _ifft_recursive(X: np.ndarray) -> np.ndarray:
    """
    IFFT (Inverse FFT) implémentée from scratch.
    
    Principe:
    ---------
    IFFT(X) = (1/N) * conjugate(FFT(conjugate(X)))
    
    Ou de manière équivalente: utiliser la FFT avec des twiddle factors conjugués
    puis diviser par N.
    
    Paramètres:
    -----------
    X : np.ndarray
        Spectre complexe
    
    Retourne:
    ---------
    x : np.ndarray
        Signal temporel
    """
    N = len(X)
    
    X_conj = np.conjugate(X)
    x_conj = _fft_recursive(X_conj)
    x = np.conjugate(x_conj) / N
    
    return x


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Convolution rapide utilisant notre FFT from scratch.
    
    Théorème de convolution:
        x * h = IFFT(FFT(x) · FFT(h))
    
    Paramètres:
    -----------
    x : np.ndarray
        Premier signal (signal d'entrée)
    h : np.ndarray
        Second signal (réponse impulsionnelle)
    
    Retourne:
    ---------
    y : np.ndarray
        Résultat de la convolution
    """
    len_x = len(x)
    len_h = len(h)
    len_full = len_x + len_h - 1
    
    n_fft = _next_power_of_2(len_full)
    
    x_padded = np.zeros(n_fft, dtype=np.float64)
    h_padded = np.zeros(n_fft, dtype=np.float64)
    
    x_padded[:len_x] = x
    h_padded[:len_h] = h
    
    X = _fft_recursive(x_padded)
    H = _fft_recursive(h_padded)
    
    Y = X * H
    
    y = _ifft_recursive(Y)
    
    return y[:len_full].real.astype(np.float32)


# =============================================================================
# GÉNÉRATION DE RÉPONSE IMPULSIONNELLE
# =============================================================================

def generate_impulse_response(
    sr: int,
    rt60: float = 1.5,
    pre_delay_ms: float = 20.0,
    early_reflections: int = 8,
    diffusion: float = 0.7
) -> np.ndarray:
    """
    Génère une réponse impulsionnelle synthétique pour la convolution.
    
    Paramètres:
    -----------
    sr : int
        Fréquence d'échantillonnage
    rt60 : float
        Temps de réverbération (secondes) - temps pour décroissance de 60dB
    pre_delay_ms : float
        Délai initial avant les premières réflexions (ms)
    early_reflections : int
        Nombre de réflexions précoces distinctes
    diffusion : float
        Densité de la queue de réverbération (0-1)
    
    Retourne:
    ---------
    ir : np.ndarray
        Réponse impulsionnelle
    """
    ir_length = int(sr * rt60 * 1.2)
    ir = np.zeros(ir_length, dtype=np.float32)
    
    ir[0] = 1.0
    
    pre_delay_samples = int(pre_delay_ms * sr / 1000)
    
    for i in range(early_reflections):
        delay = pre_delay_samples + int((i + 1) * (50 + 30 * np.sin(i * 1.3)) * sr / 1000)
        if delay < ir_length:
            amplitude = 0.6 * (0.8 ** i) * (1 if i % 2 == 0 else -1)
            ir[delay] = amplitude
    
    tail_start = pre_delay_samples + int(100 * sr / 1000)
    tail_length = ir_length - tail_start
    
    if tail_length > 0:
        decay_rate = -np.log(1e-3) / (rt60 * sr)
        
        t = np.arange(tail_length)
        envelope = np.exp(-decay_rate * t)
        
        np.random.seed(42)
        noise = np.random.randn(tail_length) * diffusion
        
        tail = noise * envelope * 0.3
        ir[tail_start:] += tail.astype(np.float32)
    
    ir = ir / np.max(np.abs(ir))
    
    return ir
