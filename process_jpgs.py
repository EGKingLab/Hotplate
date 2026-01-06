import os
import subprocess
from tkinter import Tk, filedialog


def main():
    # Hide tkinter root window
    root = Tk()
    root.withdraw()

    # Select directory containing .jpg files
    print("Select directory containing .jpg files.")
    input_dir = filedialog.askdirectory(title="Select directory containing .jpg files")
    if not input_dir:
        print("No directory selected. Exiting.")
        return

    # Select output file location and name
    print("Select output file location.")
    video_filename = filedialog.asksaveasfilename(
        title="Save video as",
        defaultextension=".avi",
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
    )
    if not video_filename:
        print("No output file specified. Exiting.")
        return

    # Build ffmpeg command
    input_pattern = os.path.join(input_dir, "img%04d.jpg")

    print("Writing movie.")
    result = subprocess.run(
        ['ffmpeg', '-framerate', '1', '-i', input_pattern, video_filename, '-y'],
        capture_output=True,
        text=True
    )

    # Report result
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed with exit code {result.returncode}")
        print(f"Error output: {result.stderr}")
    elif os.path.exists(video_filename):
        print(f"Video created successfully: {video_filename}")
    else:
        print("Warning: ffmpeg command completed but video file not found")


if __name__ == "__main__":
    main()
