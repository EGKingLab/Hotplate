import glob
import os
import pickle
import shutil
import subprocess
import tempfile
from tkinter import Tk, filedialog

import cv2 as cv


def main():
    # Hide tkinter root window
    root = Tk()
    root.withdraw()

    # Select directory containing .jpg files
    print("Select directory containing .jpg files.")
    input_dir = filedialog.askdirectory(
        title="Select directory containing .jpg files")
    if not input_dir:
        print("No directory selected. Exiting.")
        return

    # Select calibration file
    print("Select calibration .pkl file.")
    cal_file = filedialog.askopenfilename(
        title="Select calibration .pkl file",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    if not cal_file:
        print("No calibration file selected. Exiting.")
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

    # Load calibration
    try:
        with open(cal_file, 'rb') as F:
            _ = pickle.load(F)  # Skip first value
            mtx = pickle.load(F)
            dist = pickle.load(F)
        print(f"Calibration loaded from {cal_file}")
    except (IOError, pickle.UnpicklingError, EOFError) as e:
        print(f"Error: Failed to load calibration file: {e}")
        return

    # Load image list
    images = sorted(glob.glob(os.path.join(input_dir, "img*.jpg")))
    if not images:
        print("No images found in selected directory. Exiting.")
        return
    print(f"Found {len(images)} images to process.")

    # Create temp directory for undistorted images
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    # Undistort images
    print("Undistorting images.")
    undistort_errors = 0
    for i, img_path in enumerate(images):
        try:
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")

            h, w = img.shape[:2]
            newcameramtx, _ = cv.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))

            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            out_path = os.path.join(temp_dir, f"img{i+1:04d}.jpg")
            if not cv.imwrite(out_path, dst):
                raise ValueError(
                    f"Failed to write undistorted image: {out_path}")

        except Exception as e:
            undistort_errors += 1
            print(f"Warning: Error undistorting {img_path}: {e}")

    if undistort_errors > 0:
        print(f"Warning: {undistort_errors} images failed to undistort")

    # Build ffmpeg command
    input_pattern = os.path.join(temp_dir, "img%04d.jpg")

    print("Writing movie.")
    result = subprocess.run(
        ['ffmpeg', '-framerate', '1', '-i', input_pattern,
         video_filename, '-y'],
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

    # Cleanup temporary directory
    try:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory removed: {temp_dir}")
    except OSError as e:
        print(f"Warning: Failed to remove temporary directory: {e}")


if __name__ == "__main__":
    main()
