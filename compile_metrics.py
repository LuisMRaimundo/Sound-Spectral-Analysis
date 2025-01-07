import os
import pandas as pd
import re
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_note_from_quotes(note: str) -> str:
    """
    Extracts the content between single or double quotes in a string.

    Parameters:
        note (str): The input string containing a musical note.

    Returns:
        str: The extracted content if quotes are found; otherwise, returns the original string.

    Example:
        >>> extract_note_from_quotes("'A4'")
        'A4'
    """
    match = re.search(r"[\"']([^\"']+)[\"']", note)
    return match.group(1) if match else note


def note_sort_key(note: str) -> Tuple[int, int]:
    """
    Generates a sorting key for musical notes based on their pitch and octave.

    Parameters:
        note (str): The musical note string (e.g., 'C4', 'D#5').

    Returns:
        Tuple[int, int]: A tuple (octave, note order) for sorting purposes.

    Example:
        >>> note_sort_key("C4")
        (4, 1)
    """
    note = extract_note_from_quotes(note)
    match = re.match(r"([A-Ga-g])([#b]?)(\d+)", note)
    if match:
        letter = match.group(1).upper()
        accidental = match.group(2)
        octave = int(match.group(3))
        note_order = {
            'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
            'E': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9,
            'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12
        }
        full_note = f"{letter}{accidental}"
        return (octave, note_order.get(full_note, 0))
    return (0, 0)


def read_excel_metrics(file_path: str) -> dict:
    """
    Reads density metrics and spectral power from an Excel file.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        dict: A dictionary with the following keys:
            - 'Density Metric': Value from the 'Density Metric' sheet.
            - 'Spectral Density Metric': Value from the 'Spectral Power' sheet.
            - 'Total Metric': Total metric from the spectral data.

    Raises:
        Exception: If the file cannot be read or metrics cannot be extracted.
    """
    metrics = {'Density Metric': None, 'Spectral Density Metric': None, 'Total Metric': None}
    try:
        excel_data = pd.ExcelFile(file_path)

        if 'Density Metric' in excel_data.sheet_names:
            df_density = excel_data.parse('Density Metric')
            if not df_density.empty:
                metrics['Density Metric'] = df_density.iloc[0, 0]

        if 'Spectral Power' in excel_data.sheet_names:
            df_spectral = excel_data.parse('Spectral Power')
            if not df_spectral.empty:
                if 'Total Metric' in df_spectral.columns:
                    metrics['Total Metric'] = df_spectral['Total Metric'].dropna().iloc[0]
                metrics['Spectral Density Metric'] = df_spectral.iloc[0, 4]
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
    return metrics


def compile_density_metrics(folder_path: str, output_path: str = 'compiled_density_metrics.xlsx') -> None:
    """
    Compiles density metrics from Excel files in a directory.

    Parameters:
        folder_path (str): Directory containing the Excel files.
        output_path (str): Path to save the compiled results.

    Returns:
        None

    Raises:
        Exception: If no data is found or results cannot be saved.

    Example:
        >>> compile_density_metrics("path/to/folder", "output.xlsx")
    """
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                file_path = os.path.join(root, file)
                try:
                    metrics = read_excel_metrics(file_path)
                    folder_name = os.path.basename(root)
                    results.append({'File Name': folder_name, **metrics})
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")

    if not results:
        logging.warning("No valid data found to compile.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='File Name', key=lambda col: col.map(note_sort_key))
    results_df.to_excel(output_path, index=False)
    logging.info(f"Compiled results saved to {output_path}.")


# Alias for backward compatibility
extract_density_metric = compile_density_metrics

# Example usage
if __name__ == "__main__":
    compile_density_metrics(
        folder_path='path_to_results_folder',
        output_path='compiled_density_metrics.xlsx'
    )
