import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pydub import AudioSegment
from pydub.silence import detect_silence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def trim_silence(audio, silence_thresh=-50.0, min_silence_len=300):

    """
    Remove silence from the beginning and end of the audio.

    Parameters:
    audio (AudioSegment): The input audio to be processed.
    silence_thresh (float): Threshold in dBFS below which the audio is considered silence.
    min_silence_len (int): Minimum duration (in milliseconds) of silence to be detected.

    Returns:
    AudioSegment: Trimmed audio with silence removed from start and end.
    """

    audio_len = len(audio)
    if audio_len == 0:
        return audio

    # Detect silence regions
    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    if not silences:
        logging.info("No silence detected; skipping trimming.")
        return audio

    start_trim = silences[0][1] if silences[0][0] <= 100 else 0
    end_trim = silences[-1][0] if silences[-1][1] >= (audio_len - 100) else audio_len

    return audio[start_trim:end_trim]



def apply_fade(audio, fade_duration_ms=200):

    """
    Apply fade-in and fade-out effects to the audio.

    Parameters:
    audio (AudioSegment): The audio to which fades will be applied.
    fade_duration_ms (int): Duration (in milliseconds) of the fade-in and fade-out.

    Returns:
    AudioSegment: The audio with fade-in and fade-out applied.
    """

    if fade_duration_ms > len(audio) // 2:
        logging.warning("Fade duration exceeds half the audio length; reducing fade duration.")
        fade_duration_ms = len(audio) // 2

    if fade_duration_ms > 0:
        return audio.fade_in(fade_duration_ms).fade_out(fade_duration_ms)
    return audio



def process_audio_file(
    input_path,
    output_path,
    silence_threshold=-50.0,
    min_silence_duration=300,
    output_format="wav",
    fade_duration_ms=200,
    target_duration_ms=None,
    output_channels="mono",
    bit_depth="32"
):

    """
    Process an audio file by trimming silence, applying fades, and adjusting its properties.

    Parameters:
    input_path (str): Path to the input audio file.
    output_path (str): Path to save the processed audio file.
    silence_threshold (float): Threshold in dBFS for detecting silence.
    min_silence_duration (int): Minimum silence duration (in milliseconds) for detection.
    output_format (str): Format of the output audio file (e.g., 'wav', 'mp3').
    fade_duration_ms (int): Duration of fade-in and fade-out effects (in milliseconds).
    target_duration_ms (int, optional): Target duration for the audio (in milliseconds).
    output_channels (str): Desired number of audio channels ('mono' or 'stereo').
    bit_depth (str): Bit depth for the output audio ('16', '24', '32', '64').

    Returns:
    bool: True if processing is successful, False otherwise.
    """

    try:
        # Load audio
        audio = AudioSegment.from_file(input_path)

        # Convert channels if needed
        if output_channels == "mono" and audio.channels > 1:
            audio = audio.set_channels(1)
        elif output_channels == "stereo" and audio.channels == 1:
            audio = audio.set_channels(2)

        # Trim silence
        audio = trim_silence(
            audio,
            silence_thresh=silence_threshold,
            min_silence_len=min_silence_duration
        )

        # Pad or trim to target duration
        if target_duration_ms is not None:
            if len(audio) > target_duration_ms:
                audio = audio[:target_duration_ms]
            else:
                silence_needed = target_duration_ms - len(audio)
                audio += AudioSegment.silent(duration=silence_needed)

        # Apply fade after all other operations
        audio = apply_fade(audio, fade_duration_ms=fade_duration_ms)

        # Adjust bit depth if not MP3
        audio = set_bit_depth(audio, bit_depth, output_format)

        # Decide export parameters
        export_params = []
        if bit_depth == "64" and output_format.lower() in ("wav", "aiff", "aif"):
            export_params = ["-c:a", "pcm_f64le"]

        audio.export(output_path, format=output_format, parameters=export_params)

        logging.info(f"Processed file: {output_path}")
        return True
    except Exception as e:
        logging.exception(f"Error processing {input_path}: {e}")
        return False



def set_bit_depth(audio, bit_depth, output_format):

    """
    Adjust the bit depth of the audio if applicable.

    Parameters:
    audio (AudioSegment): The audio to be adjusted.
    bit_depth (str): Desired bit depth ('16', '24', '32', '64').
    output_format (str): Format of the audio (e.g., 'wav', 'mp3').

    Returns:
    AudioSegment: The audio with adjusted bit depth if applicable.
    """

    if output_format.lower() == "mp3":
        logging.info("MP3 format detected; skipping bit depth adjustments.")
        return audio

    bit_depth_map = {
        "16": 2,
        "24": 3,
        "32": 4
    }

    if bit_depth in bit_depth_map:
        return audio.set_sample_width(bit_depth_map[bit_depth])
    elif bit_depth == "64":
        logging.warning(
            "64-bit float in-memory not supported by pydub/audioop. "
            "Will force 64-bit float at export if format is WAV/AIFF."
        )
        return audio
    else:
        logging.warning(f"Unsupported bit depth '{bit_depth}'. Defaulting to 16-bit.")
        return audio.set_sample_width(2)



class AudioProcessorGUI:

    """
    Graphical interface for trimming silence, applying fades, and adjusting audio properties.

    Attributes:
    master (Tk): The main Tkinter window instance.
    file_paths (list): List of selected audio file paths for processing.
    output_dir (str): Directory to save processed files.
    silence_threshold (float): Threshold in dBFS for detecting silence.
    min_silence_duration (int): Minimum silence duration (in milliseconds) for detection.
    fade_duration_ms (int): Duration of fade-in and fade-out effects (in milliseconds).
    target_duration_ms (int or None): Target duration for the audio (in milliseconds).
    output_format (str): Format of the output audio file (e.g., 'wav', 'mp3').
    bit_depth (str): Bit depth for the output audio ('16', '24', '32', '64').
    output_channels (str): Desired number of audio channels ('mono' or 'stereo').
    """

    def __init__(self, master):
        self.master = master
        master.title("Audio Processor (Silence Removal + Fade)")

        self.file_paths = []
        self.output_dir = ""

        self.silence_threshold = -50.0
        self.min_silence_duration = 300
        self.fade_duration_ms = 200
        self.target_duration_ms = None
        self.output_format = "wav"
        self.bit_depth = "32"
        self.output_channels = "mono"

        self.create_widgets()


    def create_widgets(self):
        
        """
        Create and layout the GUI widgets for user interaction.
        """

        self.select_files_button = ttk.Button(
            self.master, 
            text="Select Files", 
            command=self.select_files
        )
        self.select_files_button.grid(row=0, column=0, padx=5, pady=5, sticky="we")

        self.select_output_button = ttk.Button(
            self.master, 
            text="Select Output Directory", 
            command=self.select_output
        )
        self.select_output_button.grid(row=1, column=0, padx=5, pady=5, sticky="we")

        ttk.Label(self.master, text="Silence Threshold (dBFS):").grid(row=2, column=0, sticky="w")
        self.threshold_entry = ttk.Entry(self.master)
        self.threshold_entry.insert(0, str(self.silence_threshold))
        self.threshold_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Minimum Silence Duration (ms):").grid(row=3, column=0, sticky="w")
        self.min_silence_entry = ttk.Entry(self.master)
        self.min_silence_entry.insert(0, str(self.min_silence_duration))
        self.min_silence_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Fade In/Out Duration (ms):").grid(row=4, column=0, sticky="w")
        self.fade_entry = ttk.Entry(self.master)
        self.fade_entry.insert(0, "200")
        self.fade_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Target Duration (ms):").grid(row=5, column=0, sticky="w")
        self.target_entry = ttk.Entry(self.master)
        self.target_entry.insert(0, "")
        self.target_entry.grid(row=5, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Output Format:").grid(row=6, column=0, sticky="w")
        self.format_combo = ttk.Combobox(
            self.master,
            values=["wav", "mp3", "aiff", "aif"],
            state="readonly"
        )
        self.format_combo.set("wav")
        self.format_combo.grid(row=6, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Bit Depth:").grid(row=7, column=0, sticky="w")
        self.bit_depth_combo = ttk.Combobox(
            self.master, 
            values=["16", "24", "32", "64"], 
            state="readonly"
        )
        self.bit_depth_combo.set("32")
        self.bit_depth_combo.grid(row=7, column=1, padx=5, pady=5)

        ttk.Label(self.master, text="Channel Mode:").grid(row=8, column=0, sticky="w")
        self.channel_mode_combo = ttk.Combobox(
            self.master,
            values=["mono", "stereo"],
            state="readonly"
        )
        self.channel_mode_combo.set("mono")
        self.channel_mode_combo.grid(row=8, column=1, padx=5, pady=5)

        self.process_button = ttk.Button(
            self.master,
            text="Process Files",
            command=self.process_files
        )
        self.process_button.grid(row=9, column=0, columnspan=2, padx=5, pady=10, sticky="we")


    def select_files(self):

        """
        Open a file dialog to select audio files for processing.
        """    

        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.aiff *.aif"),
            ("All Files", "*.*")
        ]
        self.file_paths = filedialog.askopenfilenames(
            title="Select Audio Files", 
            filetypes=filetypes
        )
        logging.info(f"Selected files: {self.file_paths}")



    def select_output(self):
        """
        Open a directory dialog to select an output directory.
        """
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        logging.info(f"Output directory: {self.output_dir}")


    def process_files(self):

        """
        Process all selected files with the specified parameters.
        """

        if not self.file_paths:
            messagebox.showerror("Error", "No files selected!")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "No output directory selected!")
            return

        try:
            self.silence_threshold = float(self.threshold_entry.get())
            self.min_silence_duration = int(self.min_silence_entry.get())
            self.fade_duration_ms = int(self.fade_entry.get())
            self.target_duration_ms = (
                int(self.target_entry.get()) 
                if self.target_entry.get().strip() 
                else None
            )
            self.output_format = self.format_combo.get().lower()
            self.bit_depth = self.bit_depth_combo.get()
            self.output_channels = self.channel_mode_combo.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters!")
            return

        for file_path in self.file_paths:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(self.output_dir, f"{base_name}.{self.output_format}")

            success = process_audio_file(
                input_path=file_path,
                output_path=out_file,
                silence_threshold=self.silence_threshold,
                min_silence_duration=self.min_silence_duration,
                output_format=self.output_format,
                fade_duration_ms=self.fade_duration_ms,
                target_duration_ms=self.target_duration_ms,
                output_channels=self.output_channels,
                bit_depth=self.bit_depth
            )
            if not success:
                logging.error(f"Failed to process: {file_path}")

        messagebox.showinfo("Done", "Processing completed!")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Audio Processor (Silence Removal + Fade)")
    root.geometry("480x400")

    app = AudioProcessorGUI(root)
    root.mainloop()
