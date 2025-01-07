import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from density import apply_density_metric, apply_density_metric_df
import re

def frequency_to_note_name(frequency):
    if frequency <= 0:
        return "Invalid Frequency"
    
    freq_A4 = 440.0
    freq_C0 = freq_A4 * 2 ** (-4.75)
    h = int(round(12 * np.log2(frequency / freq_C0)))  # Use round to get the nearest note
    octave = h // 12
    n = h % 12

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    note_name = note_names[n] + str(octave)
    flat_note_name = flat_note_names[n] + str(octave)

    closest_note_frequency = freq_C0 * 2 ** (h / 12)
    cents_deviation = 1200 * np.log2(frequency / closest_note_frequency)

    # Preferentially return the note name in flat form, if applicable, including the cents deviation
    if note_name in ['C#', 'D#', 'F#', 'G#', 'A#']:
        return f"{flat_note_name} ({cents_deviation:+.2f} cents)"
    else:
        return f"{note_name} ({cents_deviation:+.2f} cents)"


class AudioProcessor:

    """
    A class for audio processing, FFT analysis, and spectral data generation.

    This class provides methods for loading audio files, performing FFT analysis,
    applying filters, generating harmonic data, and saving results.

    Attributes:
        audio_data (list): List of tuples containing audio signals, sampling rates, 
            note names, and file paths.
        y (np.ndarray): Current audio signal.
        sr (int): Sampling rate of the current audio signal.
        S (np.ndarray): Magnitude spectrogram from FFT analysis.
        db_S (np.ndarray): Log-amplitude spectrogram (in decibels).
        freqs (np.ndarray): Array of FFT frequencies.
        times (np.ndarray): Array of time values for FFT frames.
        complete_list_df (pd.DataFrame): DataFrame containing frequencies, magnitudes, and notes.
        filtered_list_df (pd.DataFrame): DataFrame containing filtered spectral data.
        harmonic_list_df (pd.DataFrame): DataFrame containing harmonic spectral data.
        density_metric_value (float): Computed density metric value.
        scaled_density_metric_value (float): Scaled density metric value.
        n_fft (int): Number of FFT points for analysis.
        hop_length (int): Hop length for FFT.
        window (str): Window type for FFT.
    """

    def __init__(self):
        self.audio_data = []
        self.y = None
        self.sr = None
        self.S = None
        self.db_S = None
        self.freqs = None
        self.times = None
        self.complete_list_df = None
        self.filtered_list_df = None
        self.harmonic_list_df = None
        self.density_metric_value = None
        self.scaled_density_metric_value = None
        self.n_fft = 8192
        self.hop_length = None  # Will be set to n_fft if not provided
        self.window = 'hann'

    
    def load_audio_files(self, file_paths):

        """
        Loads and processes audio files.

        Parameters:
            file_paths (list of str): List of file paths for the audio files.

        Raises:
            Exception: If an audio file cannot be loaded.

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
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        print("Audio successfully loaded. Configure filters and proceed.")

    
    def extract_note_name(self, file_path):
        file_name = os.path.basename(file_path)
        # Adjust the regex to capture note and octave
        match = re.search(r"([A-G][#b]?)(\d)", file_name)
        if match:
            note = match.group(1) + match.group(2)
            return note
        else:
            print(f"Unable to extract the note from the file: {file_name}")
            return None

    
    def fft_analysis(self):

        """
        Performs FFT analysis on the current audio signal.

        This method computes the magnitude spectrogram (`S`), log-amplitude
        spectrogram (`db_S`), FFT frequencies (`freqs`), and time frames (`times`).

        Raises:
            ValueError: If no audio data is loaded.
        """

        if self.y is None or self.sr is None:
            raise ValueError("Audio data not loaded.")
        self.S = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window))
        self.db_S = librosa.amplitude_to_db(self.S, ref=np.max)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.times = librosa.frames_to_time(np.arange(self.S.shape[1]), sr=self.sr, hop_length=self.hop_length)

    
    def generate_complete_list(self):
        """
        Generates a complete list of frequencies, magnitudes, and notes.
        """
        complete_list = []
        for i, freq in enumerate(self.freqs):
            if freq > 0:
                # Convert magnitudes to linear scale before averaging
                magnitude_linear = np.mean(10 ** (self.db_S[i] / 20))
                magnitude_linear = np.maximum(magnitude_linear, 1e-12)  # Avoid log of zero
                magnitude_db = 20 * np.log10(magnitude_linear)
                note = frequency_to_note_name(freq)
                complete_list.append((freq, magnitude_db, note))

        self.complete_list_df = pd.DataFrame(complete_list, columns=['Frequency (Hz)', 'Magnitude (dB)', 'Note'])

    
    def apply_filters_and_generate_data(
    self, 
    freq_min: float = 200, 
    freq_max: float = 8000, 
    db_min: float = -80, 
    db_max: float = 0, 
    n_fft: int = 8192, 
    hop_length: int = None, 
    window: str = 'hann', 
    s: float = 1, 
    e: float = 1, 
    alpha: float = 0, 
    results_directory: str = './results', 
    **kwargs
) -> None:

        """
        Applies filters to the audio data and generates results.

        Parameters:
            freq_min (float): Minimum frequency for filtering (Hz).
            freq_max (float): Maximum frequency for filtering (Hz).
            db_min (float): Minimum magnitude for filtering (dB).
            db_max (float): Maximum magnitude for filtering (dB).
            n_fft (int): Number of FFT points.
            hop_length (int, optional): Hop length for FFT.
            window (str): Window type for FFT.
            s (float): Harmonic scaling parameter.
            e (float): Harmonic energy scaling parameter.
            alpha (float): Harmonic smoothing parameter.
            results_directory (str): Directory for saving results.
            **kwargs: Additional parameters for filtering.

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

    
        # Set hop_length to n_fft if not provided
        if hop_length is None:
            hop_length = n_fft

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

        # Default to 'linear' weighting function if not specified
        self.weight_function = kwargs.get('weight_function', 'linear')

        # Ensure the results directory exists
        if not os.path.exists(results_directory):
            os.makedirs(results_directory, exist_ok=True)

        for y, sr, note, file_path in self.audio_data:
            self.y = y
            self.sr = sr
            self.fft_analysis()
            self.generate_complete_list()

            # Apply filtering criteria
            filtered_list = self.complete_list_df[
                (self.complete_list_df['Frequency (Hz)'] >= freq_min) &
                (self.complete_list_df['Frequency (Hz)'] <= freq_max) &
                (self.complete_list_df['Magnitude (dB)'] >= db_min) &
                (self.complete_list_df['Magnitude (dB)'] <= db_max)
            ].copy()

            if not filtered_list.empty:
                filtered_list['Amplitude'] = 10 ** (filtered_list['Magnitude (dB)'] / 20)
                # Select the peak amplitude for each frequency
                self.filtered_list_df = filtered_list.groupby('Frequency (Hz)').apply(
                    lambda x: x.loc[x['Amplitude'].idxmax()]
                ).reset_index(drop=True)
            else:
                self.filtered_list_df = pd.DataFrame()

            fundamental_frequency = self.calculate_fundamental_frequency(note)

            if fundamental_frequency > 0:
                expected_harmonics = [
                    fundamental_frequency * n
                    for n in range(1, int(freq_max // fundamental_frequency) + 1)
                ]

                harmonic_list = []
                tolerance = float(kwargs.get('tolerance', 10))  # Increased tolerance to 10 Hz
                for harmonic in expected_harmonics:
                    candidates = self.filtered_list_df[
                        (self.filtered_list_df['Frequency (Hz)'] >= harmonic - tolerance) &
                        (self.filtered_list_df['Frequency (Hz)'] <= harmonic + tolerance)
                    ]
                    if not candidates.empty:
                        highest_amplitude = candidates.loc[candidates['Amplitude'].idxmax()]
                        harmonic_list.append(highest_amplitude)

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

            # Save results to the specified results directory
            output_folder = os.path.join(results_directory, note)
            os.makedirs(output_folder, exist_ok=True)
            self.save_results(output_folder, note)
            print(f"Data for {note} saved in {output_folder}.")


    def calculate_fundamental_frequency(self, note):
        # Use regular expressions to extract note name and octave
        match = re.match(r'^([A-G][#b]?)(\d)$', note)
        if match:
            note_name, octave = match.groups()
            octave = int(octave)
        else:
            print(f"Invalid note format: {note}")
            return 0

        note_frequencies = {
            'C': 16.35, 'C#': 17.32, 'Db': 17.32, 'D': 18.35, 'D#': 19.45, 'Eb': 19.45,
            'E': 20.60, 'F': 21.83, 'F#': 23.12, 'Gb': 23.12, 'G': 24.50, 'G#': 25.96,
            'Ab': 25.96, 'A': 27.50, 'A#': 29.14, 'Bb': 29.14, 'B': 30.87
        }

        base_freq = note_frequencies.get(note_name)
        if base_freq:
            fundamental_frequency = base_freq * (2 ** octave)
            return fundamental_frequency
        else:
            print(f"Note not found in dictionary: {note_name}")
            return 0

    def plot_spectrograms(self, path=None, note=""):
        """
        Plots combined spectrograms (Log 2D, Frequency Spectrum, and Mel).
        """
        if self.db_S is None or self.freqs is None or self.times is None:
            print("Insufficient data to plot the spectrogram.")
            return

        # Create a figure with 3 subplots (Log Spectrogram, Frequency Spectrum, and Mel Spectrogram)
        plt.figure(figsize=(12, 10))

        # Log 2D Spectrogram
        plt.subplot(3, 1, 1)
        librosa.display.specshow(self.db_S, sr=self.sr, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram - Note: {note}')

        # Frequency Spectrum
        plt.subplot(3, 1, 2)
        plt.plot(self.freqs[:self.S.shape[0]], self.S.mean(axis=1))
        plt.title(f'Frequency Spectrum - Note: {note}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xscale('log')

        # Mel Spectrogram
        S_mel = librosa.feature.melspectrogram(S=self.S, sr=self.sr, n_mels=128)
        S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(S_db_mel, sr=self.sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Note: {note}')

        # Save combined plot
        if path:
            plt.savefig(path)
            plt.close()
            print(f"Combined spectrogram saved at: {path}")
        else:
            plt.show()
            plt.close()

        # 3D Spectrogram
        self.plot_3d_spectrogram(
            path=f"{os.path.splitext(path)[0]}_3d.html" if path else None,
            note=note
        )

    def plot_3d_spectrogram(self, path=None, note=""):
        """
        Plots the 3D spectrogram using Plotly.

        :param path: Path to save the 3D spectrogram.
        :param note: Note name.
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

    def save_results(self, output_folder, note):
        """Saves the results to Excel files and images."""
        self.plot_spectrograms(path=f"{output_folder}/spectrogram.png", note=note)

        try:
            with pd.ExcelWriter(f'{output_folder}/spectral_analysis.xlsx', engine='xlsxwriter') as writer:
                if self.complete_list_df is not None and not self.complete_list_df.empty:
                    self.complete_list_df.to_excel(writer, sheet_name='Complete Spectrum', index=False)

                if self.filtered_list_df is not None and not self.filtered_list_df.empty:
                    self.filtered_list_df.to_excel(writer, sheet_name='Filtered Spectrum', index=False)

                if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                    self.harmonic_list_df.to_excel(writer, sheet_name='Harmonic Spectrum', index=False)

                if self.density_metric_value is not None:
                    pd.DataFrame({'Density Metric': [self.scaled_density_metric_value]}).to_excel(
                        writer, sheet_name='Density Metric', index=False
                    )

                if self.db_S is not None:
                    spectral_power_db = self.db_S.mean(axis=1)
                    relative_amplitude_squared = 10 ** (spectral_power_db / 10)
                    frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

                    df_spectral_power = pd.DataFrame({
                        'Frequency (Hz)': frequencies,
                        'Spectral Power (dB)': spectral_power_db,
                        'Relative Amplitude Squared': relative_amplitude_squared
                    })

                    fundamental_frequency = self.calculate_fundamental_frequency(note)
                    harmonics = np.array([
                        fundamental_frequency * n for n in range(1, len(frequencies) + 1)
                    ])

                    tolerance_hz = 10
                    filtered_partials = df_spectral_power[
                        df_spectral_power.apply(
                            lambda row: np.any(np.abs(row['Frequency (Hz)'] - harmonics) <= tolerance_hz),
                            axis=1
                        )
                    ].copy()

                    df_spectral_power['Filtered Partials'] = filtered_partials['Relative Amplitude Squared']

                    weight_function = getattr(self, 'weight_function', 'linear')
                    density_metric_partials = apply_density_metric(
                        filtered_partials['Relative Amplitude Squared'].values,
                        weight_function=weight_function
                    )

                    df_spectral_power['Filtered Density Metric'] = [density_metric_partials] + [None] * (len(df_spectral_power) - 1)

                    total_column_c = df_spectral_power['Relative Amplitude Squared'].sum()
                    
                    # Insert the 'Total Metric' column and value
                    df_spectral_power.insert(5, 'Total Metric', '')  # Insert empty column at position 5 (F)
                    df_spectral_power.at[0, 'Total Metric'] = total_column_c  # Add the value to the first row

                    df_spectral_power.to_excel(writer, sheet_name='Spectral Power', index=False)

        except PermissionError as e:
            print(f"Error: {e}. Please ensure the file is not open in another program.")
        except Exception as e:
            print(f"An error occurred while saving the results: {e}")
