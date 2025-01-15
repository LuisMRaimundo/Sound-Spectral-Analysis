# init.py

import logging
import traceback
import sys
from typing import NoReturn

from PyQt5.QtWidgets import QApplication, QMessageBox

from interface import SpectrumAnalyzer


def show_critical_error(error_msg: str, error_trace: str) -> None:
    """
    Display a critical error message dialog to the user.

    Args:
        error_msg (str): The main error message to display.
        error_trace (str): Detailed traceback for the error.
    """
    app = QApplication(sys.argv)  # Only create a QApplication in this fallback scenario
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Critical Error")
    msg.setText(error_msg)
    msg.setInformativeText("Check the logs for more details.")
    msg.setDetailedText(error_trace)
    msg.exec_()


def run_application() -> int:
    """
    Initializes and runs the spectral analysis application.

    This function sets up the PyQt5 application environment, initializes the
    SpectrumAnalyzer graphical interface, and starts the event loop. If an error
    occurs during initialization, it logs the error and displays a critical
    error message to the user.

    Returns:
        int: The return code from the Qt event loop.
    """
    app = QApplication(sys.argv)
    analyzer = SpectrumAnalyzer()
    analyzer.show()
    return app.exec_()


def main() -> NoReturn:
    """
    Main entry point for the spectral analysis application.

    This function configures logging, executes the core application logic,
    and handles any top-level exceptions.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        # filename='application.log',  # Uncomment to log to a file
        # filemode='a'
    )

    logging.info("Initializing the application...")

    try:
        exit_code = run_application()
        logging.info("Interface successfully initialized.")
        sys.exit(exit_code)
    except Exception as exc:
        err_message = f"An error occurred while starting the application: {exc}"
        logging.error(err_message)
        logging.error(traceback.format_exc())

        # Display an error message to the user
        show_critical_error(
            "An error occurred while starting the application.",
            traceback.format_exc()
        )
        sys.exit(1)


if __name__ == '__main__':
    main()

