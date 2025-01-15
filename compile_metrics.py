# compile_metrics.py

import os
import re
import logging
from typing import Tuple, Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_note_from_quotes(note: str) -> str:
    """
    Extracts the content between single or double quotes in a string.

    Args:
        note (str): The input string that may contain a quoted musical note.

    Returns:
        str: The extracted note if quotes are found; otherwise, returns the original string.

    Example:
        >>> extract_note_from_quotes("'A4'")
        'A4'
    """
    match = re.search(r"[\"']([^\"']+)[\"']", note)
    return match.group(1) if match else note


def note_sort_key(note: str) -> Tuple[int, int]:
    """
    Generates a sorting key for musical notes based on their pitch and octave.

    The note should be in the format: 'C4', 'D#5', 'Ab3', etc.
    Accidentals can be '#' or 'b'. The octave is assumed to be an integer.

    Args:
        note (str): The musical note string (e.g., 'C4', 'D#5').

    Returns:
        Tuple[int, int]: A tuple (octave, note_order) used for sorting.

    Example:
        >>> note_sort_key("'A#4'")
        (4, 11)
    """
    # Remove any quotes from the note first
    note_extracted = extract_note_from_quotes(note)

    # Attempt to parse the note: letter (A-G), accidental (#/b) optional, then an octave number
    match = re.match(r"([A-Ga-g])([#b]?)(\d+)", note_extracted)
    if not match:
        return 0, 0

    letter = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))

    # Mapping of note letters (with accidentals) to a numeric sequence 1..12
    note_order_map = {
        'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
        'E': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9,
        'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12
    }

    full_note_key = f"{letter}{accidental}"
    note_value = note_order_map.get(full_note_key, 0)  # default to 0 if not found
    return octave, note_value


def read_excel_metrics(file_path: str) -> Dict[str, Optional[float]]:
    """
    Reads density metrics and spectral power info from an Excel file.

    The file may contain the following sheets:
      - 'Density Metric'
      - 'Spectral Power'

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        Dict[str, Optional[float]]:
            A dictionary with keys: 
              'Density Metric', 'Spectral Density Metric', and 'Total Metric'.
            If a metric is not found, it remains None.

    Raises:
        Exception: If there is an error reading the Excel file.
    """
    # Initialize our return dictionary
    metrics = {
        'Density Metric': None,
        'Spectral Density Metric': None,
        'Total Metric': None
    }

    try:
        excel_data = pd.ExcelFile(file_path)

        # Extract Density Metric if present
        if 'Density Metric' in excel_data.sheet_names:
            df_density = excel_data.parse('Density Metric')
            if not df_density.empty:
                # By convention, the first cell in this sheet is the metric
                metrics['Density Metric'] = df_density.iloc[0, 0]

        # Extract spectral info if present
        if 'Spectral Power' in excel_data.sheet_names:
            df_spectral = excel_data.parse('Spectral Power')
            if not df_spectral.empty:
                # If 'Total Metric' is a column, capture the first non-NaN value
                if 'Total Metric' in df_spectral.columns:
                    valid_metric = df_spectral['Total Metric'].dropna()
                    if not valid_metric.empty:
                        metrics['Total Metric'] = valid_metric.iloc[0]

                # The 'Spectral Density Metric' is expected in a single cell (row 0, col 4)
                # You may adjust this logic depending on your data layout
                if df_spectral.shape[1] >= 5:  # Ensure at least 5 columns exist
                    metrics['Spectral Density Metric'] = df_spectral.iloc[0, 4]

    except Exception as e:
        logging.error(f"Error reading '{file_path}': {e}")

    return metrics


def compile_density_metrics(folder_path: str, output_path: str = 'compiled_density_metrics.xlsx') -> None:
    """
    Compiles density metrics from all Excel files in a directory into a single spreadsheet.

    The compiled file includes columns:
      - 'File Name': The name of the folder where the file was found (assumed to contain the note name).
      - 'Density Metric'
      - 'Spectral Density Metric'
      - 'Total Metric'

    Args:
        folder_path (str): The path to the directory containing Excel files.
        output_path (str): File path to save the compiled results (defaults to 'compiled_density_metrics.xlsx').

    Returns:
        None

    Example:
        >>> compile_density_metrics("path/to/folder", "output.xlsx")
    """
    results = []

    # Walk through all subfolders for .xlsx files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)  # We assume the folder name might hold the note info

                try:
                    metrics = read_excel_metrics(file_path)
                    # Construct the row data for our results table
                    row_data = {
                        'File Name': folder_name,
                        **metrics
                    }
                    results.append(row_data)
                except Exception as exc:
                    logging.error(f"Error processing '{file_path}': {exc}")

    if not results:
        logging.warning("No valid data found to compile.")
        return

    # Build a DataFrame for all results
    results_df = pd.DataFrame(results)

    # Sort using the custom note-based key on the 'File Name' column
    results_df = results_df.sort_values(
        by='File Name',
        key=lambda col: col.map(note_sort_key)
    )

    # Save to Excel
    try:
        results_df.to_excel(output_path, index=False)
        logging.info(f"Compiled results saved to '{output_path}'.")
    except Exception as exc:
        logging.error(f"Failed to save compiled results to '{output_path}': {exc}")


# Alias for backward compatibility
extract_density_metric = compile_density_metrics


if __name__ == "__main__":
    # Example usage
    compile_density_metrics(
        folder_path='path_to_results_folder',
        output_path='compiled_density_metrics.xlsx'
    )

