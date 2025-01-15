import os
import re
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the regex pattern to recognize musical notes, including uppercase and lowercase
# The pattern matches a letter (A-G), optionally followed by #, b, or 'is', and ends with a digit (e.g., C#4, Bb3, Ais4).
note_pattern = re.compile(r'([a-gA-G](#|b|is)?\d)')

# Supported audio file extensions
# These formats cover the most common uncompressed and compressed audio file types. These specific extensions were
# chosen based on their prevalence in music production and general audio usage. If additional formats become commonly used,
# such as newer compressed formats, they should be added here.
audio_extensions = ('.wav', '.aif', '.aiff', '.mp3', '.flac', '.ogg', '.m4a', '.aac')

def list_files(directory, max_depth=None):
    """
    List all files in the directory with supported audio extensions.

    Parameters:
    directory (str): Path to the directory to scan for files.
    max_depth (int, optional): Maximum depth to search within subdirectories. None for unlimited depth.

    Returns:
    list: A list of file paths with supported extensions.
    """
    files = []
    for root, dirs, file_names in os.walk(directory):
        # Check directory depth if max_depth is set
        if max_depth is not None:
            depth = root[len(directory):].count(os.sep)
            if depth >= max_depth:
                del dirs[:]
                continue
        
        for file_name in file_names:
            if file_name.lower().endswith(audio_extensions):
                files.append(os.path.join(root, file_name))
    return files

def extract_and_normalize_note(file_name):
    """
    Extract the musical note from the file name and convert it to uppercase.

    Parameters:
    file_name (str): The name of the file to process.

    Returns:
    str or None: The normalized musical note if found, otherwise None.
    """
    match = note_pattern.search(file_name)
    if match:
        note = match.group(1).upper()
        # Replace 'IS' with '#' for normalization
        note = note.replace('IS', '#')
        return note
    else:
        logging.warning(f"No valid musical note found in: {file_name}")
        return None

def rename_files(directory, progress_callback=None, dry_run=False):
    """
    Rename files by adding single quotes around the musical notes in their names.

    Parameters:
    directory (str): Path to the directory containing files to rename.
    progress_callback (function): A callback function to update progress.
    dry_run (bool): If True, simulate renaming without making actual changes.

    Returns:
    None

    Raises:
    Exception: If there is an error during file renaming.
    """
    files = list_files(directory)
    if not files:
        logging.info("No audio files found in the directory.")
        messagebox.showinfo("No Files Found", "No supported audio files were found in the selected directory.")
        return

    renamed_count = 0

    for i, full_path in enumerate(files):
        parent_dir, file_name = os.path.split(full_path)
        note = extract_and_normalize_note(file_name)

        if note:
            # Add single quotes to the note and rename the file
            new_name = re.sub(note_pattern, f"'{note}'", file_name)
            new_path = os.path.join(parent_dir, new_name)

            if dry_run:
                logging.info(f"[DRY RUN] Would rename: {file_name} -> {new_name}")
            else:
                try:
                    os.rename(full_path, new_path)
                    logging.info(f"Renamed: {file_name} -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    logging.error(f"Error renaming {file_name} at {full_path}: {e}")

        if progress_callback:
            progress_callback(i + 1, len(files))

    if dry_run:
        logging.info(f"Dry run completed. {renamed_count} file(s) would have been renamed.")
    else:
        messagebox.showinfo("Completed", f"Renaming completed! {renamed_count} file(s) renamed.")

def select_directory(progress_bar):
    """
    Open a dialog for the user to select a directory with audio files.

    Parameters:
    progress_bar (Progressbar): Progress bar widget to update.

    Returns:
    None

    Raises:
    Exception: If an error occurs during the renaming process.
    """
    directory = filedialog.askdirectory(title="Select a Directory")
    if directory:
        try:
            def update_progress(current, total):
                progress_bar['value'] = (current / total) * 100
                progress_bar.update_idletasks()

            rename_files(directory, progress_callback=update_progress, dry_run=False)
        except Exception as e:
            logging.error(f"Error processing directory {directory}: {e}")
            messagebox.showerror("Error", f"Error processing files: {e}")
    else:
        logging.info("Directory selection cancelled by user.")
        messagebox.showinfo("Cancelled", "No directory selected.")

# Graphical user interface using Tkinter
def create_interface():
    """
    Create the graphical user interface for the file renamer application.

    Returns:
    None
    """
    root = tk.Tk()
    root.title("Audio File Renamer")

    # Instruction text
    label = tk.Label(root, text="Click the button to select the directory containing audio files:", wraplength=400)
    label.pack(padx=10, pady=10)

    # Progress bar
    progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress_bar.pack(pady=10)

    # Button to select directory
    select_button = tk.Button(root, text="Select Directory", command=lambda: select_directory(progress_bar))
    select_button.pack(pady=10)

    # Button to exit
    exit_button = tk.Button(root, text="Exit", command=root.quit)
    exit_button.pack(pady=10)

    root.mainloop()

# Run the interface
if __name__ == "__main__":
    create_interface()


