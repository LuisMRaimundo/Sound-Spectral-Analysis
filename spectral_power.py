import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

def load_sound(file_path):
    """
    Loads an audio file and returns the signal and sampling rate.

    :param file_path: Path to the audio file.
    :return: Tuple (signal, fs) where 'signal' is the audio array and 'fs' is the sampling rate.
    """
    try:
        signal, fs = sf.read(file_path)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1) / np.sqrt(2)
        return signal, fs
    except Exception as e:
        print(f"Error loading sound: {e}")
        return None, None

def spectral_power(signal, n_fft=8192, hop_length=1024, window_type='hann', order=30):
    """
    Computes the spectral power of a signal using FFT with quadratic norm.

    :param signal: Audio signal.
    :param n_fft: Number of FFT points.
    :param hop_length: Hop length (default: n_fft // 4).
    :param window_type: Type of window to apply (e.g., 'hann', 'hamming').
    :param order: Number of spectral components to return.
    :return: Array with the spectral power of the first 'order' components.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    if len(signal) < n_fft:
        # Pad the signal with zeros to match n_fft
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')

    # Apply the selected window
    window = np.zeros(n_fft)
    if window_type == 'hann':
        window = np.hanning(n_fft)
    elif window_type == 'hamming':
        window = np.hamming(n_fft)
    elif window_type == 'blackman':
        window = np.blackman(n_fft)
    elif window_type == 'bartlett':
        window = np.bartlett(n_fft)
    else:
        raise ValueError(f"Window type '{window_type}' not recognized.")

    windowed_signal = signal[:n_fft] * window

    # FFT with quadratic norm
    fft_result = np.fft.fft(windowed_signal, n=n_fft)
    fft_magnitude_quadratic = np.abs(fft_result[:n_fft // 2]) ** 2  # Quadratic norm

    # Spectral power in dB
    epsilon = 1e-12  # Small value to prevent log of zero
    spectral_power = 10 * np.log10((1 / n_fft) * (fft_magnitude_quadratic + epsilon))

    # Remove infinite or negative infinite values
    spectral_power = np.where(np.isfinite(spectral_power), spectral_power, -np.inf)

    # Return the first 'order' spectral components
    return spectral_power[:order]

def plot_spectral_power(spectral_power_values, label, save_path=None, show_plot=True):
    """
    Plots the spectral power.

    :param spectral_power_values: Array with the spectral power.
    :param label: Label for the plot.
    :param save_path: Path to save the plot.
    :param show_plot: If True, displays the plot on screen.
    """
    if spectral_power_values is None or len(spectral_power_values) == 0:
        print("Error: Spectral power data is empty or invalid.")
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

def plot_multiple_spectral_powers(spectral_powers, labels, save_path=None, show_plot=True):
    """
    Plots multiple spectral powers on the same graph.

    :param spectral_powers: List of spectral power arrays.
    :param labels: List of corresponding labels (note names).
    :param save_path: Path to save the plot, if provided.
    :param show_plot: If True, displays the plot on screen.
    """
    if not spectral_powers or not labels or len(spectral_powers) != len(labels):
        print("Error: Spectral power data or labels are incorrect.")
        return

    plt.figure()

    for spectral_power_values, label in zip(spectral_powers, labels):
        if spectral_power_values is None or len(spectral_power_values) == 0:
            print(f"Error: Spectral power data for {label} is empty or invalid.")
            continue
        plt.plot(spectral_power_values, label=label)

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
