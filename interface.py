import os
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel,
    QLineEdit, QComboBox, QTabWidget, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from proc_audio import AudioProcessor
from spectral_power import spectral_power, plot_spectral_power, plot_multiple_spectral_powers
from compile_metrics import extract_density_metric

#extract_density_metric

class SpectrumAnalyzer(QMainWindow):
    """
    A PyQt5-based graphical user interface for spectral analysis.

    This class provides an interactive GUI for tasks like loading audio files,
    applying spectral analysis, configuring filters, and compiling results.
    It integrates functionalities like density metrics computation, spectral 
    power analysis, and visualizations.

    Attributes:
        audio_processor (AudioProcessor): Handles core audio processing tasks.
        spectral_power_enabled (bool): Flag to enable or disable spectral power analysis.
        results_directory (str): Directory where results are saved.
    """

    def __init__(self):
        """
    Initializes the graphical user interface (GUI).

    This method sets up the main layout and tabs for the application, including 
    controls for loading files, applying filters, and analyzing spectral data.
    """
        super().__init__()
        self.setWindowTitle('Spectrum Analyzer')
        self.setGeometry(100, 100, 800, 600)
        self.audio_processor = AudioProcessor()
        self.spectral_power_enabled = False  # Initialization of the variable
        self.initUI()

    def initUI(self):
        """
        Initializes the user interface.
        """
        print("Initializing user interface...")
        self.Main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.setup_controls_tab()
        self.setup_filters_tab()

        self.Main_layout.addWidget(self.tabs)
        container = QWidget()
        container.setLayout(self.Main_layout)
        self.setCentralWidget(container)

    def setup_controls_tab(self):
        """
        Configures the controls tab of the GUI.

        The controls tab contains buttons for actions like loading audio files,
        toggling spectral power, analyzing spectral power, and compiling results.
        """
        controls_tab = QWidget()
        controls_layout = QVBoxLayout()

        # Button to load audio files
        self.load_button = QPushButton('Load Audio Files')
        self.load_button.clicked.connect(self.load_audio_files)
        controls_layout.addWidget(self.load_button)

        # Button to toggle spectral power
        self.toggle_spectral_power_button = QPushButton('Toggle Spectral Power')
        self.toggle_spectral_power_button.setCheckable(True)
        self.toggle_spectral_power_button.clicked.connect(self.toggle_spectral_power)
        controls_layout.addWidget(self.toggle_spectral_power_button)

        # Button to analyze spectral power
        self.analyze_spectral_power_button = QPushButton('Analyze Spectral Power')
        self.analyze_spectral_power_button.clicked.connect(self.analyze_spectral_power)
        controls_layout.addWidget(self.analyze_spectral_power_button)

        # Button to analyze multiple spectral powers
        self.analyze_multiple_spectral_powers_button = QPushButton('Analyze Multiple Spectral Powers')
        self.analyze_multiple_spectral_powers_button.clicked.connect(self.analyze_multiple_spectral_powers)
        controls_layout.addWidget(self.analyze_multiple_spectral_powers_button)

        # Button to compile density metrics
        self.compile_density_metrics_button = QPushButton('Compile Density Metrics')
        self.compile_density_metrics_button.clicked.connect(self.compile_density_metrics)
        controls_layout.addWidget(self.compile_density_metrics_button)

        controls_tab.setLayout(controls_layout)
        self.tabs.addTab(controls_tab, "Controls")

    def setup_filters_tab(self):
        """
        Configures the filters tab of the GUI.

        The filters tab allows users to specify filter parameters such as minimum 
        and maximum frequency, decibel range, FFT size, and window type. Users can 
        apply these filters to the loaded audio data.
        """
        filters_tab = QWidget()
        filters_layout = QVBoxLayout()
        grid_filters = QHBoxLayout()

        # Add filter controls to the filters tab
        self.label_min_freq = QLabel('Minimum Frequency (Hz):')
        self.input_min_freq = QLineEdit()
        grid_filters.addWidget(self.label_min_freq)
        grid_filters.addWidget(self.input_min_freq)

        self.label_max_freq = QLabel('Maximum Frequency (Hz):')
        self.input_max_freq = QLineEdit()
        grid_filters.addWidget(self.label_max_freq)
        grid_filters.addWidget(self.input_max_freq)

        self.label_min_db = QLabel('Minimum Magnitude (dB):')
        self.input_min_db = QLineEdit()
        grid_filters.addWidget(self.label_min_db)
        grid_filters.addWidget(self.input_min_db)

        self.label_max_db = QLabel('Maximum Magnitude (dB):')
        self.input_max_db = QLineEdit()
        grid_filters.addWidget(self.label_max_db)
        grid_filters.addWidget(self.input_max_db)

        self.label_tolerance = QLabel('Tolerance (Hz):')
        self.input_tolerance = QLineEdit()
        grid_filters.addWidget(self.label_tolerance)
        grid_filters.addWidget(self.input_tolerance)

        filters_layout.addLayout(grid_filters)

        self.label_weight_function = QLabel('Weight Function:')
        self.combo_weight_function = QComboBox()
        self.combo_weight_function.addItems(['linear', 'sqrt', 'exp', 'log', 'inverse log', 'sum'])
        filters_layout.addWidget(self.label_weight_function)
        filters_layout.addWidget(self.combo_weight_function)

        self.label_n_fft = QLabel('FFT Window Size (n_fft):')
        self.input_n_fft = QLineEdit()
        filters_layout.addWidget(self.label_n_fft)
        filters_layout.addWidget(self.input_n_fft)

        self.label_hop_length = QLabel('Hop Length:')
        self.input_hop_length = QLineEdit()
        filters_layout.addWidget(self.label_hop_length)
        filters_layout.addWidget(self.input_hop_length)

        self.label_window_type = QLabel('Window Type:')
        self.combo_window_type = QComboBox()
        self.combo_window_type.addItems(['hann', 'hamming', 'blackman', 'bartlett'])
        filters_layout.addWidget(self.label_window_type)
        filters_layout.addWidget(self.combo_window_type)

        self.apply_filters_button = QPushButton('Apply Filters')
        self.apply_filters_button.clicked.connect(self.apply_filters)
        self.apply_filters_button.setFont(QFont("Arial", 10, QFont.Bold))
        filters_layout.addWidget(self.apply_filters_button)

        filters_tab.setLayout(filters_layout)
        self.tabs.addTab(filters_tab, "Filters")

    def load_audio_files(self):
        """
        Opens a dialog for selecting and loading audio files.

        Uses the `AudioProcessor` class to load and preprocess selected audio files. 
        Displays a success or warning message based on the outcome.

        Exceptions:
            Exception: Displays an error message if file loading fails.
        """
        try:
            options = QFileDialog.Options()
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Audio Files",
                "",
                "Audio Files (*.wav *.mp3 *.flac *.aif *.aiff);;All Files (*)",
                options=options
            )
            if files:
                self.audio_processor.load_audio_files(files)  # Call method from AudioProcessor
                QMessageBox.information(self, "Success", f"{len(files)} files successfully loaded.")
            else:
                QMessageBox.warning(self, "Warning", "No files selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while loading the files: {str(e)}")

    def toggle_spectral_power(self):
        """
        Toggles the state of spectral power analysis.

        Updates the `spectral_power_enabled` attribute based on the button's state 
        and informs the user of the change.
        """
        self.spectral_power_enabled = self.toggle_spectral_power_button.isChecked()
        status = "enabled" if self.spectral_power_enabled else "disabled"
        QMessageBox.information(self, "Spectral Power", f"Spectral power is {status}.")

    def analyze_spectral_power(self):
        """
        Performs spectral power analysis on loaded audio files.

        Generates spectral power plots for each audio file and saves them in the 
        selected results directory.

        Raises:
            Exception: Displays an error message if the analysis fails.
        """
        if not self.audio_processor.audio_data:
            QMessageBox.warning(self, "Warning", "No audio files loaded.")
            return

        if not hasattr(self, 'results_directory') or not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a directory to save results.")
            return

        if self.spectral_power_enabled:
            try:
                for y, sr, note, file_path in self.audio_processor.audio_data:
                    save_path = os.path.join(self.results_directory, f"{note}/spectral_power.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sp = spectral_power(y, min(len(y), 256))
                    plot_spectral_power(sp, label=note, save_path=save_path)

                QMessageBox.information(self, "Analysis", "Spectral power analysis completed successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error in spectral power analysis: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Spectral power is not enabled.")


    def analyze_multiple_spectral_powers(self):
        """
        Performs combined spectral power analysis for multiple audio files.

        Plots and saves a combined graph of spectral powers for all loaded audio 
        files. Requires spectral power to be enabled.

        Raises:
            Exception: Displays an error message if the analysis fails.
        """
        if not self.audio_processor.audio_data:
            QMessageBox.warning(self, "Warning", "No audio files loaded.")
            return

        if not hasattr(self, 'results_directory') or not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a directory to save results.")
            return

        if self.spectral_power_enabled:
            try:
                spectral_powers = []
                labels = []

                # Collect spectral power data and labels
                for y, sr, note, file_path in self.audio_processor.audio_data:
                    sp = spectral_power(y, min(len(y), 256))
                    spectral_powers.append(sp)
                    labels.append(note)

                # Define save path
                save_path = os.path.join(self.results_directory, "multiple_spectral_powers.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Plot and save spectral powers
                plot_multiple_spectral_powers(spectral_powers, labels, save_path=save_path)
                QMessageBox.information(self, "Analysis", f"Analysis of multiple spectral powers completed successfully.\nSaved at: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error in multiple spectral powers analysis: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Spectral power is not enabled.")


    def apply_filters(self):
        """
        Applies user-defined filters to the loaded audio data.

        Retrieves filter parameters from the GUI, applies them using the 
        `AudioProcessor` class, and saves the filtered results.

        Raises:
            ValueError: If filter parameters are invalid.
            PermissionError: If results cannot be saved due to directory permissions.
            Exception: For unexpected errors during filter application.
        """
        try:
            # Retrieve filter parameters
            freq_min_text = self.input_min_freq.text().strip()
            freq_max_text = self.input_max_freq.text().strip()
            db_min_text = self.input_min_db.text().strip()
            db_max_text = self.input_max_db.text().strip()
            tolerance_text = self.input_tolerance.text().strip()
            n_fft_text = self.input_n_fft.text().strip()
            hop_length_text = self.input_hop_length.text().strip()

            # Parse filter parameters
            freq_min = float(freq_min_text) if freq_min_text else 200  # Default value
            freq_max = float(freq_max_text) if freq_max_text else 8000  # Default value
            db_min = float(db_min_text) if db_min_text else -80  # Default value
            db_max = float(db_max_text) if db_max_text else 0  # Default value
            tolerance = float(tolerance_text) if tolerance_text else 10.0  # Default tolerance
            n_fft = int(n_fft_text) if n_fft_text else 8192
            hop_length = int(hop_length_text) if hop_length_text else None
            window = self.combo_window_type.currentText()
            weight_function = self.combo_weight_function.currentText()

            # Validate results directory
            if not hasattr(self, 'results_directory') or not self.results_directory:
                QMessageBox.warning(self, "Warning", "Please select a directory to save results.")
                return

            # Apply filters through the audio processor
            self.audio_processor.apply_filters_and_generate_data(
                freq_min=freq_min, freq_max=freq_max, db_min=db_min, db_max=db_max,
                tolerance=tolerance, n_fft=n_fft, hop_length=hop_length,
                window=window, weight_function=weight_function,
                results_directory=self.results_directory
            )
            QMessageBox.information(self, "Filters Applied", "Filters applied and results saved successfully.")
        except ValueError as ve:
            QMessageBox.critical(self, "Value Error", f"Error applying filters: {str(ve)}")
        except PermissionError as pe:
            QMessageBox.critical(self, "Permission Error", f"Permission denied: {str(pe)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filters: {str(e)}")


    def compile_density_metrics(self):
        """
        Compiles density metrics from processed results.

        Opens a dialog for selecting a folder containing result files and generates 
        a summary of density metrics. Saves the summary as an Excel file.

        Raises:
            Exception: Displays an error message if the compilation fails.
        """
        try:
            selected_folder = QFileDialog.getExistingDirectory(
                self, "Select the Folder with Results",
                os.getcwd()
            )
            if not selected_folder:
                QMessageBox.warning(self, "Warning", "No folder selected.")
                return

            output_path = os.path.join(selected_folder, 'compiled_density_metrics.xlsx')
            extract_density_metric(selected_folder, output_path)
            QMessageBox.information(
                self, "Success", f"Density metrics compiled successfully in: {output_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error compiling metrics: {str(e)}")


    def setup_controls_tab(self):
        """
        Configures the controls tab.
        """
        controls_tab = QWidget()
        controls_layout = QVBoxLayout()

        # Button to load audio files
        self.load_button = QPushButton('Load Audio Files')
        self.load_button.clicked.connect(self.load_audio_files)
        controls_layout.addWidget(self.load_button)

        # Button to choose save directory
        self.choose_save_dir_button = QPushButton('Choose Save Directory')
        self.choose_save_dir_button.clicked.connect(self.choose_save_directory)
        controls_layout.addWidget(self.choose_save_dir_button)

        # Button to toggle spectral power
        self.toggle_spectral_power_button = QPushButton('Toggle Spectral Power')
        self.toggle_spectral_power_button.setCheckable(True)
        self.toggle_spectral_power_button.clicked.connect(self.toggle_spectral_power)
        controls_layout.addWidget(self.toggle_spectral_power_button)

        # Button to analyze spectral power
        self.analyze_spectral_power_button = QPushButton('Analyze Spectral Power')
        self.analyze_spectral_power_button.clicked.connect(self.analyze_spectral_power)
        controls_layout.addWidget(self.analyze_spectral_power_button)

        # Button to analyze multiple spectral powers
        self.analyze_multiple_spectral_powers_button = QPushButton('Analyze Multiple Spectral Powers')
        self.analyze_multiple_spectral_powers_button.clicked.connect(self.analyze_multiple_spectral_powers)
        controls_layout.addWidget(self.analyze_multiple_spectral_powers_button)

        # Button to compile density metrics
        self.compile_density_metrics_button = QPushButton('Compile Density Metrics')
        self.compile_density_metrics_button.clicked.connect(self.compile_density_metrics)
        controls_layout.addWidget(self.compile_density_metrics_button)

        controls_tab.setLayout(controls_layout)
        self.tabs.addTab(controls_tab, "Controls")

    def choose_save_directory(self):
        """
        Opens a dialog for selecting the directory to save results.

        Updates the `results_directory` attribute and informs the user of the 
        selected directory.
        """

        selected_directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results", os.getcwd()
        )
        if selected_directory:
            self.results_directory = selected_directory
            QMessageBox.information(self, "Directory Selected", f"Results will be saved in: {selected_directory}")
        else:
            QMessageBox.warning(self, "Warning", "No directory selected.")

        
