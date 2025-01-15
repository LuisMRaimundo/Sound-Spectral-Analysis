# proc_audio.py

import os
import re
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Optional, Tuple

from density import apply_density_metric, apply_density_metric_df


def frequency_to_note_name(frequency: float) -> str:
    """
    Converts a frequency in Hz to the nearest musical note name (including octave and cent offset).

    Args:
        frequency (float): The frequency to convert.

    Returns:
        str: A string representing the closest note name and cents deviation.
    """
    if frequency <= 0:
        return "Invalid Frequency"

    freq_A4 = 440.0
    freq_C0 = freq_A4 * 2 ** (-4.75)  # Frequency of C0
    h = int(round(12 * np.log2(frequency / freq_C0)))  # Semitone offset from C0
    octave = h // 12
    n = h % 12

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                       'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    note_name_sharp = note_names[n] + str(octave)
    note_name_flat = flat_note_names[n] + str(octave)

    closest_note_frequency = freq_C0 * 2 ** (h / 12)
    cents_deviation = 1200 * np.log2(frequency / closest_note_frequency)

    # If the note name is enharmonically sharp, return its flat equivalent
    if note_name_sharp in ['C#' + str(octave), 'D#' + str(octave),
                           'F#' + str(octave), 'G#' + str(octave), 'A#' + str(octave)]:
        return f"{note_name_flat} ({cents_deviation:+.2f} cents)"
    else:
        return f"{note_name_sharp} ({cents_deviation:+.2f} cents)"


class AudioProcessor:
    """
    A class for audio processing, FFT analysis, and spectral data generation.

    This class provides methods for loading audio files, performing FFT analysis,
    applying filters, generating harmonic data, and saving results.

    Attributes:
        audio_data (List[Tuple[np.ndarray, int, str, str]]):
            List of tuples containing:
                - (audio signal, sampling rate, note name, file path).
        y (Optional[np.ndarray]): Current audio signal.
        sr (Optional[int]): Sampling rate of the current audio signal.
        S (Optional[np.ndarray]): Magnitude spectrogram from FFT analysis.
        db_S (Optional[np.ndarray]): Log-amplitude spectrogram (in decibels).
        freqs (Optional[np.ndarray]): Array of FFT frequencies.
        times (Optional[np.ndarray]): Array of time values for FFT frames.
        complete_list_df (Optional[pd.DataFrame]): Frequencies, magnitudes, and notes.
        filtered_list_df (Optional[pd.DataFrame]): Filtered spectral data.
        harmonic_list_df (Optional[pd.DataFrame]): Harmonic spectral data.
        density_metric_value (Optional[float]): Computed density metric value.
        scaled_density_metric_value (Optional[float]): Scaled density metric value.
        n_fft (int): Number of FFT points for analysis.
        hop_length (Optional[int]): Hop length for FFT (defaults to n_fft if None).
        window (str): Window type for FFT.
    """

    def __init__(self):
        self.audio_data: List[Tuple[np.ndarray, int, str, str]] = []
        self.y: Optional[np.ndarray] = None
        self.sr: Optional[int] = None

        self.S: Optional[np.ndarray] = None
        self.db_S: Optional[np.ndarray] = None
        self.freqs: Optional[np.ndarray] = None
        self.times: Optional[np.ndarray] = None

        self.complete_list_df: Optional[pd.DataFrame] = None
        self.filtered_list_df: Optional[pd.DataFrame] = None
        self.harmonic_list_df: Optional[pd.DataFrame] = None

        self.density_metric_value: Optional[float] = None
        self.scaled_density_metric_value: Optional[float] = None

        self.n_fft: int = 8192
        self.hop_length: Optional[int] = None  # Will default to n_fft if not provided
        self.window: str = 'hann'

    # -------------------------------------------------------------------------
    #                              LOADING FILES
    # -------------------------------------------------------------------------

    def load_audio_files(self, file_paths: List[str]) -> None:
        """
        Loads and stores audio data (y, sr, note, file_path) from the provided file paths.

        Args:
            file_paths (List[str]): A list of file paths for the audio files.

        Raises:
            Exception: If an audio file fails to load.

        Example:
            processor = AudioProcessor()
            processor.load_audio_files(["file1.wav", "file2.wav"])
        """
        for file_path in file_paths:
            try:
                y, sr = librosa.load(file_path, sr=None)
                note = self.extract_note_name(file_path)
                if y is not None and note is not None:
                    self.audio_data.append((y, sr, note, file_path))
            except Exception as exc:
                print(f"Error loading {file_path}: {exc}")

        print("Audio successfully loaded. Configure filters and proceed.")

    def extract_note_name(self, file_path: str) -> Optional[str]:
        """
        Extracts a note name (including octave) from the file name using a regex pattern.

        Args:
            file_path (str): The file path string.

        Returns:
            Optional[str]: The extracted note name if found, otherwise None.
        """
        file_name = os.path.basename(file_path)
        # Adjust the regex to capture note and octave, e.g., A#3 or Bb4
        match = re.search(r"([A-G][#b]?)(\d)", file_name)
        if match:
            return match.group(1) + match.group(2)
        else:
            print(f"Unable to extract the note from the file name: {file_name}")
            return None

    # -------------------------------------------------------------------------
    #                              FFT ANALYSIS
    # -------------------------------------------------------------------------

    def fft_analysis(self) -> None:
        """
        Performs FFT analysis on the current audio signal, storing the
        magnitude spectrogram, log-amplitude spectrogram, FFT frequencies,
        and time frames.

        Raises:
            ValueError: If no audio data (y, sr) is loaded.
        """
        if self.y is None or self.sr is None:
            raise ValueError("Audio data not loaded.")

        # Compute STFT and log-amplitude spectrogram
        self.S = np.abs(librosa.stft(
            self.y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window
        ))
        self.db_S = librosa.amplitude_to_db(self.S, ref=np.max)

        # Compute frequencies and time frames
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        frame_indices = np.arange(self.S.shape[1])
        self.times = librosa.frames_to_time(
            frame_indices, sr=self.sr, hop_length=self.hop_length
        )

    def generate_complete_list(self) -> None:
        """
        Generates a complete list of (frequency, magnitude (dB), note) for the current audio data.

        Stores the data in `self.complete_list_df`.
        """
        if self.S is None or self.db_S is None or self.freqs is None:
            return

        complete_list = []
        for i, freq in enumerate(self.freqs):
            if freq > 0:
                # Convert magnitudes (in dB) to linear scale before averaging
                magnitude_linear = np.mean(10 ** (self.db_S[i] / 20))
                magnitude_linear = np.maximum(magnitude_linear, 1e-12)  # Avoid log(0)
                magnitude_db = 20 * np.log10(magnitude_linear)
                note_str = frequency_to_note_name(freq)
                complete_list.append((freq, magnitude_db, note_str))

        self.complete_list_df = pd.DataFrame(
            complete_list, columns=['Frequency (Hz)', 'Magnitude (dB)', 'Note']
        )

    # -------------------------------------------------------------------------
    #                              FILTERS & DATA
    # -------------------------------------------------------------------------

    def apply_filters_and_generate_data(
        self,
        freq_min: float = 200,
        freq_max: float = 8000,
        db_min: float = -80,
        db_max: float = 0,
        n_fft: int = 8192,
        hop_length: Optional[int] = None,
        window: str = 'hann',
        s: float = 1,
        e: float = 1,
        alpha: float = 0,
        results_directory: str = './results',
        **kwargs
    ) -> None:
        """
        Applies filters to the loaded audio data and generates results (spectra, metrics, etc.).

        Args:
            freq_min (float): Minimum frequency for filtering (Hz).
            freq_max (float): Maximum frequency for filtering (Hz).
            db_min (float): Minimum magnitude for filtering (dB).
            db_max (float): Maximum magnitude for filtering (dB).
            n_fft (int): Number of FFT points.
            hop_length (Optional[int]): Hop length for FFT (defaults to n_fft if None).
            window (str): Window type for FFT.
            s (float): Harmonic scaling parameter (not fully used in this snippet).
            e (float): Harmonic energy scaling parameter (not fully used here).
            alpha (float): Harmonic smoothing parameter (not fully used here).
            results_directory (str): Directory for saving results.
            **kwargs: Additional parameters for filtering or weighting.

        Raises:
            PermissionError: If the results directory is not writable.
            ValueError: If invalid parameters are provided.
            Exception: For unexpected errors.

        Example:
            processor = AudioProcessor()
            processor.apply_filters_and_generate_data(
                freq_min=300, freq_max=5000, db_min=-60, db_max=-10,
                results_directory='/path/to/results'
            )
        """
        # Default hop_length to n_fft if None
        hop_length = hop_length or n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.weight_function = kwargs.get('weight_function', 'linear')
        tolerance = float(kwargs.get('tolerance', 10.0))  # Filter tolerance in Hz

        if not os.path.exists(results_directory):
            os.makedirs(results_directory, exist_ok=True)

        for y, sr, note, file_path in self.audio_data:
            self.y = y
            self.sr = sr

            # Run core analysis and list generation
            self.fft_analysis()
            self.generate_complete_list()

            if self.complete_list_df is None or self.complete_list_df.empty:
                self.filtered_list_df = pd.DataFrame()
                self.harmonic_list_df = pd.DataFrame()
                self.density_metric_value = None
                self.scaled_density_metric_value = None
                continue

            # Apply filtering criteria
            filtered_list = self.complete_list_df[
                (self.complete_list_df['Frequency (Hz)'] >= freq_min) &
                (self.complete_list_df['Frequency (Hz)'] <= freq_max) &
                (self.complete_list_df['Magnitude (dB)'] >= db_min) &
                (self.complete_list_df['Magnitude (dB)'] <= db_max)
            ].copy()

            if not filtered_list.empty:
                filtered_list['Amplitude'] = 10 ** (filtered_list['Magnitude (dB)'] / 20)
                # Select the peak amplitude for each unique frequency
                self.filtered_list_df = filtered_list.groupby('Frequency (Hz)', as_index=False).apply(
                    lambda x: x.loc[x['Amplitude'].idxmax()]
                ).reset_index(drop=True)
            else:
                self.filtered_list_df = pd.DataFrame()

            fundamental_frequency = self.calculate_fundamental_frequency(note)
            if fundamental_frequency > 0:
                # Compute expected harmonics up to freq_max
                expected_harmonics = [
                    fundamental_frequency * n
                    for n in range(1, int(freq_max // fundamental_frequency) + 1)
                ]

                harmonic_list = []
                for harmonic in expected_harmonics:
                    candidates = self.filtered_list_df[
                        (self.filtered_list_df['Frequency (Hz)'] >= harmonic - tolerance) &
                        (self.filtered_list_df['Frequency (Hz)'] <= harmonic + tolerance)
                    ]
                    if not candidates.empty:
                        highest_amp_entry = candidates.loc[candidates['Amplitude'].idxmax()]
                        harmonic_list.append(highest_amp_entry)

                self.harmonic_list_df = pd.DataFrame(harmonic_list).drop_duplicates().reset_index(drop=True)

                if not self.harmonic_list_df.empty:
                    self.density_metric_value = apply_density_metric_df(
                        self.harmonic_list_df,
                        weight_function=self.weight_function
                    )
                    self.scaled_density_metric_value = self.density_metric_value * 10
                else:
                    self.harmonic_list_df = pd.DataFrame()
                    self.density_metric_value = None
                    self.scaled_density_metric_value = None
            else:
                self.harmonic_list_df = pd.DataFrame()
                self.density_metric_value = None
                self.scaled_density_metric_value = None

            # Save results for this audio file/note
            output_folder = os.path.join(results_directory, note)
            os.makedirs(output_folder, exist_ok=True)
            self.save_results(output_folder, note)
            print(f"Data for note '{note}' saved in {output_folder}.")

    def calculate_fundamental_frequency(self, note: str) -> float:
        """
        Calculates a fundamental frequency from a note name (e.g., 'A4', 'C#3').

        Args:
            note (str): The note name.

        Returns:
            float: The fundamental frequency in Hz, or 0 if invalid note format.
        """
        match = re.match(r'^([A-G][#b]?)(\d)$', note)
        if not match:
            print(f"Invalid note format: {note}")
            return 0.0

        note_name, octave_str = match.groups()
        octave = int(octave_str)

        note_frequencies = {
            'C': 16.35, 'C#': 17.32, 'Db': 17.32, 'D': 18.35, 'D#': 19.45, 'Eb': 19.45,
            'E': 20.60, 'F': 21.83, 'F#': 23.12, 'Gb': 23.12, 'G': 24.50, 'G#': 25.96,
            'Ab': 25.96, 'A': 27.50, 'A#': 29.14, 'Bb': 29.14, 'B': 30.87
        }

        base_freq = note_frequencies.get(note_name)
        if base_freq is None:
            print(f"Note not recognized in dictionary: {note_name}")
            return 0.0

        return base_freq * (2 ** octave)

    # -------------------------------------------------------------------------
    #                              VISUALIZATIONS
    # -------------------------------------------------------------------------

    def plot_spectrograms(self, path: Optional[str] = None, note: str = "") -> None:
        """
        Plots combined spectrograms (Log 2D, Frequency Spectrum, Mel) and saves or shows them.

        Args:
            path (Optional[str]): File path to save the resulting figure (PNG).
            note (str): The note name for labeling the plots.
        """
        if self.db_S is None or self.freqs is None or self.times is None:
            print("Insufficient data to plot the spectrogram.")
            return

        plt.figure(figsize=(12, 10))

        # Log 2D Spectrogram
        plt.subplot(3, 1, 1)
        librosa.display.specshow(self.db_S, sr=self.sr, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - Note: {note}')

        # Frequency Spectrum (mean across time)
        plt.subplot(3, 1, 2)
        mean_spectrum = self.S.mean(axis=1) if self.S is not None else None
        if mean_spectrum is not None:
            plt.plot(self.freqs[:len(mean_spectrum)], mean_spectrum)
        plt.title(f'Frequency Spectrum - Note: {note}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xscale('log')

        # Mel Spectrogram
        plt.subplot(3, 1, 3)
        S_mel = librosa.feature.melspectrogram(S=self.S, sr=self.sr, n_mels=128)
        S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
        librosa.display.specshow(S_db_mel, sr=self.sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Note: {note}')

        if path:
            plt.savefig(path)
            plt.close()
            print(f"Combined spectrogram saved at: {path}")
            # After saving 2D spectrograms, automatically plot 3D
            self.plot_3d_spectrogram(path=f"{os.path.splitext(path)[0]}_3d.html", note=note)
        else:
            plt.show()
            plt.close()

    def plot_3d_spectrogram(self, path: Optional[str] = None, note: str = "") -> None:
        """
        Plots the 3D spectrogram using Plotly.

        Args:
            path (Optional[str]): Path to save the 3D spectrogram as an HTML file.
            note (str): Note name for labeling the plot.
        """
        if self.db_S is None or self.freqs is None or self.times is None:
            print("Insufficient data to plot the 3D spectrogram.")
            return

        trace = go.Surface(z=self.db_S, x=self.times, y=self.freqs)
        layout = go.Layout(
            title=f'3D Spectrogram - Note: {note}',
            scene=dict(
                xaxis=dict(title='Time (s)'),
                yaxis=dict(title='Frequency (Hz)'),
                zaxis=dict(title='Magnitude (dB)')
            )
        )
        fig = go.Figure(data=[trace], layout=layout)

        if path:
            fig.write_html(path)
            print(f"3D spectrogram saved at: {path}")
        else:
            fig.show()

    # -------------------------------------------------------------------------
    #                              SAVE RESULTS
    # -------------------------------------------------------------------------

    def save_results(self, output_folder: str, note: str) -> None:
        """
        Saves results (spectrogram plots, dataframes, metrics) to the specified output folder.

        Args:
            output_folder (str): The directory path to save output files.
            note (str): The note name, used for labeling and file naming.
        """
        # Save combined spectrogram (2D + 3D)
        spectrogram_png_path = os.path.join(output_folder, "spectrogram.png")
        self.plot_spectrograms(path=spectrogram_png_path, note=note)

        excel_path = os.path.join(output_folder, 'spectral_analysis.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Complete Spectrum
                if self.complete_list_df is not None and not self.complete_list_df.empty:
                    self.complete_list_df.to_excel(writer, sheet_name='Complete Spectrum', index=False)

                # Filtered Spectrum
                if self.filtered_list_df is not None and not self.filtered_list_df.empty:
                    self.filtered_list_df.to_excel(writer, sheet_name='Filtered Spectrum', index=False)

                # Harmonic Spectrum
                if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                    self.harmonic_list_df.to_excel(writer, sheet_name='Harmonic Spectrum', index=False)

                # Density Metric
                if self.density_metric_value is not None:
                    pd.DataFrame({'Density Metric': [self.scaled_density_metric_value]}) \
                      .to_excel(writer, sheet_name='Density Metric', index=False)

                # Spectral Power
                if self.db_S is not None:
                    spectral_power_db = self.db_S.mean(axis=1)  # Mean across time
                    relative_amplitude_squared = 10 ** (spectral_power_db / 10)
                    frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

                    df_spectral_power = pd.DataFrame({
                        'Frequency (Hz)': frequencies,
                        'Spectral Power (dB)': spectral_power_db,
                        'Relative Amplitude Squared': relative_amplitude_squared
                    })

                    fundamental_freq = self.calculate_fundamental_frequency(note)
                    if fundamental_freq > 0:
                        harmonics = np.array([
                            fundamental_freq * n
                            for n in range(1, len(frequencies) + 1)
                        ])
                        tolerance_hz = 10
                        # Filter partials near any harmonic
                        filtered_partials = df_spectral_power[
                            df_spectral_power.apply(
                                lambda row: np.any(np.abs(row['Frequency (Hz)'] - harmonics) <= tolerance_hz),
                                axis=1
                            )
                        ].copy()

                        # Only fill the 'Filtered Partials' column where applicable
                        df_spectral_power['Filtered Partials'] = None
                        df_spectral_power.loc[filtered_partials.index, 'Filtered Partials'] = \
                            filtered_partials['Relative Amplitude Squared']

                        weight_function = getattr(self, 'weight_function', 'linear')
                        partial_values = filtered_partials['Relative Amplitude Squared'].values
                        density_metric_partials = apply_density_metric(
                            partial_values, weight_function=weight_function
                        )

                        df_spectral_power['Filtered Density Metric'] = [density_metric_partials] + \
                            [None] * (len(df_spectral_power) - 1)

                        total_column_c = df_spectral_power['Relative Amplitude Squared'].sum()
                        df_spectral_power.insert(5, 'Total Metric', '')
                        df_spectral_power.at[0, 'Total Metric'] = total_column_c

                    df_spectral_power.to_excel(writer, sheet_name='Spectral Power', index=False)

            print(f"Spectral analysis saved at: {excel_path}")
        except PermissionError as pe:
            print(f"Permission Error: {pe}. The file might be open in another program.")
        except Exception as exc:
            print(f"An error occurred while saving the results: {exc}")
