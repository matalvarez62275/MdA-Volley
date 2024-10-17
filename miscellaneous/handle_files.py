import tkinter as tk
import tkinter.filedialog as filedialog

def get_video_name():
    
    """
    Opens a file dialog for the user to select a video file and returns the validity of the selection and the file path.
    The function creates a hidden root window to open a file dialog for selecting a file. It checks if the selected file
    has an extension of either 'mp4' or 'MOV'. If a valid file is selected, it prints the file path and returns True 
    along with the file path. If an invalid file is selected, it prints an error message and returns False along with 
    the file path.
    Returns:
        tuple: A tuple containing a boolean indicating the validity of the selected file and the file path as a string.
    """
    
    # Create a root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog
    file_path = filedialog.askopenfilename()

    # Check if a file was selected
    valid = False
    if file_path[-3:] == "mp4" or file_path[-3:] == "MOV":
        print(f"Selected file: {file_path}")
        valid = True
    else:
        print("Error: Incorrect File Type Selected")

    # Remember to destroy the root window
    root.destroy()
    return valid, file_path