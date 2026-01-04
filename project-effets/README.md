# Projet : Effets Audio par Traitement du Signal

## Description

Ce projet implémente plusieurs effets audio classiques en utilisant uniquement des techniques de traitement du signal numérique (DSP), sans recourir à des fonctions toutes faites de bibliothèques externes.

**Effets implémentés :**
1. **Réverbération par convolution** - Simulation d'acoustique de salle
2. **Chorus / Flanger** - Effets de modulation temporelle
3. **Pitch Shifting + Formant Shifting** - Modification de hauteur tonale

## Structure du projet

```
projet-effets/
├── src/
│   ├── reverb.py              # Réverbération par convolution (FFT from scratch)
│   ├── chorus_flanger.py      # Chorus et Flanger
│   ├── tremolo_vibrato.py     # Tremolo et Vibrato
│   └── pitch_formant_shift.py # Pitch shifting avec formants
├── data/
│   ├── voices/                # Échantillons vocaux
│   │   └── sample_voice_01.wav
│   ├── instruments/           # Échantillons instrumentaux
│   │   └── sample_instrument_01.wav
│   └── mixtures/              # Mélanges voix + instruments
│       └── voice_plus_instrument.wav
├── notebooks/
│   └── demo.ipynb             # Démonstration interactive
├── rapport.pdf                # Article scientifique (format IEEE)
└── README.md                  # Ce fichier
```

## Installation

### Prérequis

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Jupyter (pour les notebooks)

### Installation des dépendances

```bash
pip install numpy scipy matplotlib jupyter
```

## Utilisation

### 1. Utilisation des modules Python

Chaque effet est implémenté dans un fichier `.py` séparé avec une fonction `apply_effect()` :

```python
import numpy as np
from scipy.io import wavfile

# Importer un effet
from src.reverb import apply_effect

# Charger un fichier audio
sr, audio = wavfile.read('data/voices/sample_voice_01.wav')
audio = audio.astype(np.float32) / 32768.0  # Normaliser en [-1, 1]

# Appliquer l'effet
audio_processed = apply_effect(audio, sr, rt60=2.0, wet=0.5)

# Sauvegarder
wavfile.write('output.wav', sr, (audio_processed * 32767).astype(np.int16))
```

### 2. Signatures des fonctions

#### Réverbération (`src/reverb.py`)
```python
apply_effect(
    x: np.ndarray,           
    sr: int,                 
    rt60: float = 1.5,       
    pre_delay_ms: float = 20.0,  
    wet: float = 0.5,        
    early_reflections: int = 8,
    diffusion: float = 0.7
) -> np.ndarray
```

#### Tremolo et Vibrato (`src/tremolo_vibrato.py`)
```python
apply_effect(
		x: np.ndarray,
		sr: int,
		N: int = len(x),
		tremolo_freq: float = 5.0,
		tremolo_depth: float = 0.8,
		vibrato_freq: float = 6.0,
		vibrato_depth_ms: float = 5.0
) -> np.ndarray
```


#### Chorus / Flanger (`src/chorus_flanger.py`)
```python
apply_effect(
    x: np.ndarray,
    sr: int,
    effect_type: str = 'chorus',  
    num_voices: int = 3,
    base_delay_ms: float = 20.0,
    mod_depth_ms: float = 7.0,
    mod_freq: float = 1.2,
    feedback: float = 0.5,
    mix: float = 0.5
) -> np.ndarray
```

#### Pitch Shifting (`src/pitch_shift.py`)
```python
apply_effect(
    x: np.ndarray,
    sr: int,
    pitch_semitones: float = 0.0,      
    formant_semitones: float = 0.0,    
    preserve_formants: bool = True,    
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray
```

## Auteurs

- Karim Sadiki

## Références

[1] U. Zölzer, DAFX : Digital Audio Effects, 2nd ed. Wiley, 2011.
[2] M. R. Schroeder, “Natural sounding artificial reverberation,” J. Audio
Eng. Soc., vol. 10, no. 3, pp. 219–223, 1962.
[3] J. A. Moorer, “About this reverberation business,” Computer Music
J., vol. 3, no. 2, pp. 13–28, 1979.
[4] J.W.CooleyandJ.W.Tukey,“Analgorithmforthemachinecalcula-
tion of complex Fourier series,” Math. Comput., vol. 19, pp. 297–301,
1965.
[5] J.Dattorro,“Effectdesign,part2:Delay-linemodulationandchorus,”
J. Audio Eng. Soc., vol. 45, no. 10, pp. 764–788, 1997.
[6] M. Dolson, “The phase vocoder : A tutorial,” Computer Music J.,
vol. 10, no. 4, pp. 14–27, 1986.
[7] J. Laroche and M. Dolson, “New phase-vocoder techniques for pitch-
shifting,” Proc. IEEE WASPAA, 1999.
[8] R. Bristow-Johnson, “A detailed analysis of a time-domain formant-
corrected pitch-shifting algorithm,” Proc. 99th AES Conv., 1995.
[9] A. Röbel, “A shape-invariant phase vocoder for speech transforma-
tion,” Proc. DAFx-10, 2010.
[10] S.Bernsee,“Timestretchingandpitchshifting–Anoverview,”1999.
[Online].
[11] N. Bernardini, “Traditional implementation of a phase vocoder,” Univ.
Trieste, 2000.
[12] J. O. Smith, Spectral Audio Signal Processing. CCRMA, Stanford. [Online].

## Licence

Projet académique - Helha - 2025
