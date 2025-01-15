# interface.py

import os
import sys
from typing import Optional, List, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel,
    QLineEdit, QComboBox, QTabWidget, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from proc_audio import AudioProcessor
from spectral_power import spectral_power, plot_spectral_power, plot_multiple_spectral_powers
from compile_metrics import extract_density_metric


class SpectrumAnalyzer(QMainWindow):
    """
    A PyQt5-based graphical user interface for spectral analysis.

    This class provides an interactive GUI for tasks like loading audio files,
    applying spectral analysis, configuring filters, and compiling results.
    It integrates functionalities like density metrics computation, spectral 
    power analysis, and visualizations.
    """

    def __init__(self):
        """
        Initializes the graphical user interface (GUI).
        """
        super().__init__()
        self.setWindowTitle('Spectrum Analyzer')
        self.setGeometry(100, 100, 800, 600)

        # Core data/processing objects
        self.audio_processor = AudioProcessor()
        self.spectral_power_enabled = False
        self.results_directory: Optional[str] = None

        # Set up the UI
        self.init_ui()

    def init_ui(self) -> None:
        """
        Initializes the user interface layout and tabs.
        """
        # 1) Light blue background for the entire QMainWindow
        self.setStyleSheet("background-color:rgb(230, 218, 204);")

        self.main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.setup_controls_tab()
        self.setup_filters_tab()

        self.main_layout.addWidget(self.tabs)
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

    def setup_controls_tab(self) -> None:
        """
        Configures the 'Controls' tab of the GUI.
        """
        controls_tab = QWidget()
        controls_layout = QVBoxLayout()

        # Button: Load Audio Files
        self.load_button = QPushButton('Load Audio Files')
        # A pale olive green color
        self.load_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.load_button.clicked.connect(self.load_audio_files)
        controls_layout.addWidget(self.load_button)

        # Button: Choose Save Directory
        self.choose_save_dir_button = QPushButton('Choose Save Directory')
        # A pale olive green color
        self.choose_save_dir_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.choose_save_dir_button.clicked.connect(self.choose_save_directory)
        controls_layout.addWidget(self.choose_save_dir_button)

        # Button: switch_on Spectral Power
        self.switch_on_spectral_power_button = QPushButton('Switch on Spectral Power')
        # A pale olive green color
        self.switch_on_spectral_power_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.switch_on_spectral_power_button.setCheckable(True)
        self.switch_on_spectral_power_button.clicked.connect(self.switch_on_spectral_power)
        controls_layout.addWidget(self.switch_on_spectral_power_button)

        # Button: Analyze Spectral Power
        self.analyze_spectral_power_button = QPushButton('Analyze Spectral Power')
        # A pale olive green color
        self.analyze_spectral_power_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.analyze_spectral_power_button.clicked.connect(self.analyze_spectral_power)
        controls_layout.addWidget(self.analyze_spectral_power_button)

        # Button: Analyze Multiple Spectral Powers
        self.analyze_multiple_spectral_powers_button = QPushButton('Analyze Multiple Spectral Powers')
        # A pale olive green color
        self.analyze_multiple_spectral_powers_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.analyze_multiple_spectral_powers_button.clicked.connect(self.analyze_multiple_spectral_powers)
        controls_layout.addWidget(self.analyze_multiple_spectral_powers_button)

        # Button: Compile Density Metrics
        self.compile_density_metrics_button = QPushButton('Compile Density Metrics')
        # A pale olive green color
        self.compile_density_metrics_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.compile_density_metrics_button.clicked.connect(self.compile_density_metrics)
        controls_layout.addWidget(self.compile_density_metrics_button)

        controls_tab.setLayout(controls_layout)
        self.tabs.addTab(controls_tab, "Controls")
 

    def setup_filters_tab(self) -> None:
        """
        Configures the 'Filters' tab of the GUI.
        """
        filters_tab = QWidget()
        filters_layout = QVBoxLayout()
        grid_filters = QHBoxLayout()

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

    # -------------------------------------------------------------------------
    #                           CONTROLS TAB FUNCTIONS
    # -------------------------------------------------------------------------

    def choose_save_directory(self) -> None:
        """
        Opens a dialog for selecting the directory to save results.
        """
        selected_directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results", os.getcwd()
        )
        if selected_directory:
            self.results_directory = selected_directory
            QMessageBox.information(self, "Directory Selected", 
                                    f"Results will be saved in: {selected_directory}")
        else:
            QMessageBox.warning(self, "Warning", "No directory selected.")

    def load_audio_files(self) -> None:
        """
        Opens a dialog for selecting and loading audio files.
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
                self.audio_processor.load_audio_files(files)
                QMessageBox.information(self, "Success", 
                                        f"{len(files)} files successfully loaded.")
            else:
                QMessageBox.warning(self, "Warning", "No files selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                                 f"An error occurred while loading the files: {str(e)}")

    def switch_on_spectral_power(self) -> None:
        """
        switch_ons the state of spectral power analysis.
        """
        self.spectral_power_enabled = self.switch_on_spectral_power_button.isChecked()
        status = "enabled" if self.spectral_power_enabled else "disabled"
        QMessageBox.information(self, "Spectral Power", f"Spectral power is {status}.")

    def analyze_spectral_power(self) -> None:
        """
        Performs spectral power analysis on loaded audio files.
        """
        if not self.audio_processor.audio_data:
            QMessageBox.warning(self, "Warning", "No audio files loaded.")
            return

        if not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a directory to save results.")
            return

        if self.spectral_power_enabled:
            try:
                for y, sr, note, file_path in self.audio_processor.audio_data:
                    save_path = os.path.join(self.results_directory, f"{note}", "spectral_power.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    sp = spectral_power(y, min(len(y), 256))
                    plot_spectral_power(sp, label=note, save_path=save_path)

                QMessageBox.information(self, "Analysis", 
                                        "Spectral power analysis completed successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                     f"Error in spectral power analysis: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Spectral power is not enabled.")

    def analyze_multiple_spectral_powers(self) -> None:
        """
        Performs combined spectral power analysis for multiple audio files.
        """
        if not self.audio_processor.audio_data:
            QMessageBox.warning(self, "Warning", "No audio files loaded.")
            return

        if not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a directory to save results.")
            return

        if self.spectral_power_enabled:
            try:
                spectral_powers = []
                labels = []

                for y, sr, note, file_path in self.audio_processor.audio_data:
                    sp = spectral_power(y, min(len(y), 256))
                    spectral_powers.append(sp)
                    labels.append(note)

                save_path = os.path.join(self.results_directory, "multiple_spectral_powers.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                plot_multiple_spectral_powers(spectral_powers, labels, save_path=save_path)
                QMessageBox.information(self, "Analysis", 
                                        f"Analysis of multiple spectral powers completed.\nSaved at: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                     f"Error in multiple spectral powers analysis: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Spectral power is not enabled.")

    def compile_density_metrics(self) -> None:
        """
        Compiles density metrics from processed results.
        """
        try:
            selected_folder = QFileDialog.getExistingDirectory(
                self, "Select the Folder with Results", os.getcwd()
            )
            if not selected_folder:
                QMessageBox.warning(self, "Warning", "No folder selected.")
                return

            output_path = os.path.join(selected_folder, 'compiled_density_metrics.xlsx')
            extract_density_metric(selected_folder, output_path)
            QMessageBox.information(self, "Success", 
                                    f"Density metrics compiled successfully in: {output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                                 f"Error compiling metrics: {str(e)}")

    def apply_filters(self) -> None:
        """
        Applies user-defined filters to the loaded audio data.
        """
        try:
            freq_min = float(self.input_min_freq.text().strip() or 200)
            freq_max = float(self.input_max_freq.text().strip() or 8000)
            db_min = float(self.input_min_db.text().strip() or -80)
            db_max = float(self.input_max_db.text().strip() or 0)
            tolerance = float(self.input_tolerance.text().strip() or 10.0)
            n_fft = int(self.input_n_fft.text().strip() or 8192)
            hop_length_text = self.input_hop_length.text().strip()
            hop_length = int(hop_length_text) if hop_length_text else None
            window = self.combo_window_type.currentText()
            weight_function = self.combo_weight_function.currentText()

            if not self.results_directory:
                QMessageBox.warning(self, "Warning", "Please select a directory to save results.")
                return

            self.audio_processor.apply_filters_and_generate_data(
                freq_min=freq_min,
                freq_max=freq_max,
                db_min=db_min,
                db_max=db_max,
                tolerance=tolerance,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                weight_function=weight_function,
                results_directory=self.results_directory
            )
            QMessageBox.information(self, "Filters Applied", 
                                    "Filters applied and results saved successfully.")
        except ValueError as ve:
            QMessageBox.critical(self, "Value Error", f"Error applying filters: {str(ve)}")
        except PermissionError as pe:
            QMessageBox.critical(self, "Permission Error", f"Permission denied: {str(pe)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying filters: {str(e)}")


