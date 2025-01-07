# Spectral Sound Analysis

## License
This package is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.  

![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-blue.svg)

https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Overview
**Spectral Sound Analysis** is a Python package for performing spectral and harmonic analysis of audio signals. It provides tools for analyzing audio recordings, calculating spectral power, evaluating harmonic densities, and visualizing results. This package is designed for researchers, musicians, and engineers working in fields such as acoustics, musicology, and psychoacoustics.

---

## Important Remarks
This package is designed to work exclusively with short audio files, each containing only a single musical note. 

### For optimal results:
- **Remove initial transients**: Ensure the starting portion of the sound is removed to focus on the steady-state of the note.
- **Apply sound attenuation**: Fade out the ending of the sound to avoid abrupt cuts.

Each audio file must include the note name and octave in its filename, enclosed in single quotes. Examples:
`Violin_'A4'.wav`, `Oboe_'Bb4'.mp3`, `Cello_'G#4'.aif`.
---


## Additional Tools
In addition to the main package containing the essential Python modules, the following tools are provided to assist users in preparing audio files for analysis:

- **`cut_fade.py`**: This script allows automatic processing of audio files by:
  - Removing silence;
  - Applying fade-in and fade-out effects;
  - Defining the intended sound length;
  - Converting audio modes (stereo to mono or vice versa).
  - **Supported formats**: `.WAV`, `.AIF`, `.AIFF`, `.MP3`;
  - **Supported bit depths**: 16-bit, 24-bit, 32-bit, 64-bit.

- **`comma_create.py`**: This script automatically adds single quotes (`'`) around note names in audio file names for compatibility with the package requirements.

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
For developers, you can install the package in editable mode using:

```bash
pip install -e .

---
##Requirements
This package requires the following Python libraries:

numpy
scipy
matplotlib
pandas
librosa
PyQt5
Dependencies will be automatically installed when you install the package.
Requires Python 3.8 or higher.

---

##Acknowledgments

This work was funded by:
Foundation for Science and Technology (FCT) - Portugal
Under the DOI 10.54499/2020.08817.BD 8D 
(https://doi.org/10.54499/2020.08817.BD) 


And also supported by:

Universidade NOVA de Lisboa
Centre for the Study of Sociology and Musical Aesthetics (CESEM)
Contemporary Music Group Investigation (GIMC)
In2Past

---

Contributing
At the moment, contributions are not accepted as the project is under active development. Future versions may include guidelines for contributors.

---

Contact
If you have questions or feedback, feel free to reach out to the author:

---

Author: Luis Miguel da Luz Raimundo
ORCID Profile: https://orcid.org/0000-0003-1712-6358


Email Addresses:
lmr.2020@gmail.com
luisraimundo@fcsh.unl.pt
 
