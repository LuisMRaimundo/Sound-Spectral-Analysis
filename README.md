# Sound Spectral Analysis
Copyright (c) 2024, Luís Miguel da Luz Raimundo

---

## Overview
**Spectral Sound Analysis** is a Python package for performing spectral and harmonic analysis of audio signals. It provides tools for analyzing brief audio recordings, calculating spectral power, evaluating harmonic densities, and visualizing results. This package is designed for researchers, musicians, and engineers working in fields such as acoustics, musicology, and psychoacoustics.

---

## Important Remarks
This package is designed to work exclusively with short audio files, each containing only a single musical note. 

---

### For optimal results:
- **Remove sound initial transients**: Ensure the starting portion of the sound is removed, allowing to focus on the steady-state of the sound.
- **Apply sound attenuation**: Fade out the ending of the sound to avoid abrupt cuts. The cutfade.py module allows this task to be performed automatically. 

Each audio file must include the note name and octave in its filename, enclosed in single quotes. Examples:
`Violin_'A4'.wav`, `Oboe_'Bb4'.mp3`, `Cello_'G#4'.aif`.

---

## Additional Tools
In addition to the main package containing the essential Python modules, the following tools are provided to assist users in preparing audio files for analysis:

- **`cutfade.py`**: This script allows authomatic processing of several audio files at a time, by:

  - Removing silence;
  - Applying fade-in and fade-out effects;
  - Defining the intended sound length;
  - Converting audio modes (stereo to mono or vice versa).
  - **Supported formats**: `.WAV`, `.AIF`, `.AIFF`, `.MP3`;
  - **Supported bit depths**: 16-bit, 24-bit, 32-bit, 64-bit.

- **`comcreate.py`**: This script automatically adds single quotes (`'`) around note names in audio file names for compatibility with the package requirements.

---

## Features
- **FFT Analysis**: Extract spectral features from audio signals.
- **Spectral Power Computation**: Visualize the power of spectral components.
- **Harmonic Analysis**: Detect fundamental frequencies and harmonics.
- **Density Metrics**: Evaluate harmonic densities with customizable weighting functions.
- **Batch Processing**: Compile results from multiple audio files into Excel reports.
- **Visualization**: Generate 2D and 3D spectrograms and plots.
- **Supported Audio Formats**: `.wav`, `.mp3`, `.flac`, `.aif`, `.aiff`.

---


## Installation

**1.** Clone the Repository
```bash
git clone https://github.com/LuisMRaimundo/Sound-Spectral-Analysis.git
cd Sound-Spectral-Analysis
```

**2.** (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv
```


*On macOS/Linux:*
```bash
source venv/bin/activate
```

*On Windows:*
```bash
venv\Scripts\activate
```


**3.** Install Dependencies
```bash


pip install -r requirements.txt

```

**4.** run the project
```bash 
python init.py
```
---

## Modules Overview

**Main Script:**
- init.py – Starts the code and displays the interface.
- Integrated Modules Under init.py:
- compile_metrics.py
- density.py
- interface.py
- proc_audio.py
- spectral_power.py

**Autonomous/Complementary Modules:**
- cutfade.py (requires pydub)
- comcreate.py

---

## Requirements
- Python: 3.8 or higher


**Libraries:**
- numpy
- scipy
- matplotlib
- pandas
- librosa (requires FFmpeg for audio file support)
- PyQt5
- plotly
- soundfile
- xlsxwriter 
(Dependencies are automatically installed when you run the pip install ... command above.)


---


## How to use

**1.** Place your audio files in a folder (preferably they should be between 1 and 3 seconds long and of good audio quality, containing ONLY a single musical note). The file name must contain the name of the note, enclosed in single inverted commas (i.e. ‘Ab4’).

**2.** Run the init.py module

**3.** Load the audio files

**4.** Choose the folder where the results should be saved.

**5.** Toggle on 'spectral power'

**6.** Press the ‘spectral power’ button

**7.** Press the ‘multiple spectral powers’ button

**8.** Open the ‘Filters’ tab, and choose the analysis options:


- Minimum and maximum frequency [in Hertz] (*e.g.*: 440 - 20000)
- Minimum and maximum magnitudes [in dB] (*e.g.*: min: -90, max: 0) 
- Tolerance [in Hertz] - Useful when sounds do not comply with the diapason (A = 440). 
- Weight function (linear, square root, exponential, logarithmic, inverse logarithmic, sum)
- FFT (Fast Fourier Transform) window size (512, 1024, 2048, 4096, etc.)  
- Size of the FFT jump (512, 1024, 2048, 4096, etc.) Should be 1/2 or, even better, 1/4 the size of the window.
- Type of window (‘hann’, ‘hamming’, ‘blackman’, ‘bartlett’)
- Press ‘apply filters’
- Once the analysis has been carried out, go back to the ‘controls’ tab and press ‘Compile Density Metrics’, and select the folder where the results of the analysis have been saved.

---


## Acknowledgments

This software was developed by Luís Raimundo, as part of a broader study on Music Analysis
**DOI 10.54499/2020.08817.BD 8D** (https://doi.org/10.54499/2020.08817.BD) 

and was funded by:

**Foundation for Science and Technology (FCT)** - Portugal


And also supported by:

**Universidade NOVA de Lisboa**
**Centre for the Study of Sociology and Musical Aesthetics** (CESEM)
**Contemporary Music Group Investigation** (GIMC)
**In2Past**

---


- As to the 'spectral_power.py module, we follow the methods described by Maddage et al. (2002), & Plazak et al. (2010) respectively, the **spectral power formula**, and the **decibel to linear conversion formula**. 


REFERENCES:

**Maddage**, N. C., **Xu**, C., **Lee**, C., **Kankanhalli**, M., & **Tian**, Q. (2002). "Statistical analysis of musical instruments". In Advances in Multimedia Information Processing – PCM 2002: Third IEEE Pacific Rim Conference on Multimedia, Hsinchu, Taiwan, December 16–18, 2002 (Lecture Notes in *Computer Science*, Vol. 2532). pp. 581--588. Springer. DOI:10.1007/3-540-36228-2_72


Plazak, J., Huron, D., & Williams, B. (2010). "Fixed average spectra of orchestral instrument tones". *Empirical Musicology Review*, 5(1). DOI:10.18061/1811/45171

---

## Contributing
At the moment, contributions are not accepted as the project is under active development. Future versions may include guidelines for contributors.

---

## Contact
Please feel free to open an issue on this repository if you encounter problems or want to request more/new features to the code

**Luís Miguel da Luz Raimundo**
ORCID Profile: https://orcid.org/0000-0003-1712-6358

## Email Addresses:
lmr.2020@gmail.com
luisraimundo@fcsh.unl.pt

---

## License

This package is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**This license allows others to download and share your work with attribution, but not to modify it in any way or use it commercially**










