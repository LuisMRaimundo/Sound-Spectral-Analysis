import logging
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from interface import SpectrumAnalyzer
import sys
print(sys.path)



def main():
    """
    Initializes and runs the spectral analysis application.

    This function sets up the PyQt5 application environment, initializes the 
    SpectrumAnalyzer graphical interface, and starts the event loop. If an error 
    occurs during initialization, it logs the error and displays a critical error 
    message to the user.

    Exceptions:
        If an exception is raised during the application startup, it is logged,
        and a QMessageBox displays the error details.

    Usage:
        Run this script as the main entry point to start the application.

    Example:
        python init.py
    """
    app = QApplication(sys.argv)
    analyzer = SpectrumAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    try:
        logging.info("Initializing the application...")
        main()
        logging.info("Interface successfully initialized.")
    except Exception as e:
        logging.error(f"An error occurred while starting the application: {e}")
        logging.error(traceback.format_exc())
        # Display an error message to the user
        app = QApplication(sys.argv)  # Ensure QApplication is instantiated before creating QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Critical Error")
        msg.setText("An error occurred while starting the application.")
        msg.setInformativeText("Check the logs for more details.")
        msg.setDetailedText(traceback.format_exc())
        msg.exec_()
        sys.exit(1)
