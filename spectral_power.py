# spectral_power.py

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Tuple, Optional, List


def load_sound(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Loads an audio file and returns the signal (as mono) and sampling rate.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]:
            A tuple (signal, fs) where 'signal' is a 1D NumPy array containing 
            the audio samples, and 'fs' is the sampling rate. If loading fails, 
            returns (None, None).
    """
    try:
        signal, fs = sf.read(file_path)
        if signal.ndim > 1:
            # Convert to mono by averaging channels, adjusted by sqrt(2) to preserve energy
            signal = np.mean(signal, axis=1) / np.sqrt(2)
        return signal, fs
    except Exception as e:
        print(f"Error loading sound from {file_path}: {e}")
        return None, None


def spectral_power(
    signal: np.ndarray, 
    n_fft: int = 8192, 
    hop_length: Optional[int] = None, 
    window_type: str = 'hann', 
    order: int = 30
) -> np.ndarray:
    """
    Computes the spectral power of a signal using FFT with a quadratic norm 
    and returns the power of the first 'order' components.

    Args:
        signal (np.ndarray): The audio signal (1D NumPy array).
        n_fft (int): Number of FFT points (default=8192).
        hop_length (Optional[int]): Hop length (default=None => n_fft//4).
        window_type (str): Type of window to apply (e.g., 'hann', 'hamming', 'blackman', 'bartlett').
        order (int): Number of spectral components (bins) to return.

    Returns:
        np.ndarray: A 1D NumPy array of size 'order' representing the spectral power in dB.
    """
    if signal is None or len(signal) == 0:
        print("Warning: Input signal is empty or None. Returning an empty array.")
        return np.array([])

    # Set default hop length if not provided
    if hop_length is None:
        hop_length = n_fft // 4

    # Zero-pad the signal if it's shorter than n_fft
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')

    # Create a dictionary for the supported window functions
    window_functions = {
        'hann': np.hanning,
        'hamming': np.hamming,
        'blackman': np.blackman,
        'bartlett': np.bartlett
    }

    # Apply the selected window (or raise an error if invalid)
    window_func = window_functions.get(window_type.lower())
    if window_func is None:
        raise ValueError(f"Invalid window type '{window_type}'. "
                         f"Supported types: {list(window_functions.keys())}")

    window = window_func(n_fft)
    windowed_signal = signal[:n_fft] * window

    # FFT with quadratic norm
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_magnitude_quadratic = np.abs(fft_result[:n_fft // 2]) ** 2  # Quadratic norm

    # Convert to spectral power in dB
    epsilon = 1e-12  # Small value to prevent log(0)
    spectral_power_db = 10 * np.log10((1 / n_fft) * (fft_magnitude_quadratic + epsilon))

    # Replace invalid values (-inf, inf) with a large negative number or a safe default
    spectral_power_db = np.where(np.isfinite(spectral_power_db), spectral_power_db, -np.inf)

    # Return only the first 'order' components
    return spectral_power_db[:order]


def plot_spectral_power(
    spectral_power_values: np.ndarray, 
    label: str, 
    save_path: Optional[str] = None, 
    show_plot: bool = True
) -> None:
    """
    Plots the spectral power of a single signal.

    Args:
        spectral_power_values (np.ndarray): A 1D array with the spectral power in dB.
        label (str): Label for the plot legend.
        save_path (Optional[str]): File path to save the plot (PNG, JPG, etc.).
        show_plot (bool): If True, displays the plot on screen.
    """
    if spectral_power_values is None or len(spectral_power_values) == 0:
        print(f"Error: Spectral power data is empty or invalid for label '{label}'.")
        return

    plt.figure()
    plt.plot(spectral_power_values, label=label)
    plt.title('Spectral Power')
    plt.xlabel('Harmonic Order')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    plt.legend()

    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")

    if show_plot:
        plt.show()
    plt.close()


def plot_multiple_spectral_powers(
    spectral_powers: List[np.ndarray], 
    labels: List[str], 
    save_path: Optional[str] = None, 
    show_plot: bool = True
) -> None:
    """
    Plots multiple spectral power curves on the same graph.

    Args:
        spectral_powers (List[np.ndarray]): A list of 1D NumPy arrays representing spectral power in dB.
        labels (List[str]): A list of corresponding labels for each spectral power array.
        save_path (Optional[str]): File path to save the combined plot.
        show_plot (bool): If True, displays the plot on screen.
    """
    if not spectral_powers or not labels or len(spectral_powers) != len(labels):
        print("Error: Spectral power data or labels are missing or mismatched in length.")
        return

    plt.figure()

    for sp_values, lbl in zip(spectral_powers, labels):
        if sp_values is None or len(sp_values) == 0:
            print(f"Warning: Spectral power data for '{lbl}' is empty or invalid. Skipping.")
            continue
        plt.plot(sp_values, label=lbl)

    plt.title('Spectral Powers')
    plt.xlabel('Harmonic Order')
    plt.ylabel('Power (dB)')
    plt.grid(True)
    plt.legend()

    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")

    if show_plot:
        plt.show()
    plt.close()

