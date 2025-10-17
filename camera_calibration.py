#!/usr/bin/env python3
"""
Camera Calibration Script

This script performs camera calibration using a chessboard pattern.
It captures images, detects chessboard corners, computes camera parameters,
and saves the calibration data.

Reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
"""

import glob
import pickle
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
from picamera2 import Picamera2


# Configuration constants
N_IMAGES = 15
WAIT_TIME = 4
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 768
CHESSBOARD_ROWS = 13
CHESSBOARD_COLS = 19
PREVIEW_WAIT_MS = 1000
MIN_VALID_IMAGES = 3


class CalibrationError(Exception):
    """Custom exception for calibration errors."""
    pass


@contextmanager
def camera_context(width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
    """
    Context manager for camera initialization and cleanup.

    Args:
        width: Camera frame width
        height: Camera frame height

    Yields:
        Picamera2: Configured and started camera instance

    Raises:
        CalibrationError: If camera initialization fails
    """
    camera = None
    try:
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (width, height)}
        )
        camera.configure(config)
        camera.start()
        yield camera
    except Exception as e:
        raise CalibrationError(f"Camera initialization failed: {e}")
    finally:
        if camera is not None:
            try:
                camera.stop()
                camera.close()
            except Exception as e:
                print(f"Warning: Error during camera cleanup: {e}")


def capture_calibration_images(
    n_images: int = N_IMAGES,
    wait_time: int = WAIT_TIME,
    output_prefix: str = "calibrate_image",
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT
) -> List[str]:
    """
    Capture calibration images using the camera.

    Args:
        n_images: Number of images to capture
        wait_time: Wait time in seconds between captures
        output_prefix: Prefix for output image files
        width: Camera frame width
        height: Camera frame height

    Returns:
        List of captured image filenames

    Raises:
        CalibrationError: If image capture fails
    """
    print(f"Taking {n_images} calibration images...")
    print(f"Please position the chessboard in different orientations.")
    print(f"Wait time between captures: {wait_time} seconds\n")

    captured_files = []

    try:
        with camera_context(width, height) as camera:
            for i in range(n_images):
                try:
                    # Show preview frame
                    frame = camera.capture_array()
                    cv.imshow('Camera Preview', frame)
                    cv.waitKey(wait_time * 1000)

                    # Capture and save image
                    filename = f'{output_prefix}{i}.jpg'
                    camera.capture_file(filename)
                    captured_files.append(filename)
                    print(f"Captured image {i + 1}/{n_images}: {filename}")

                except Exception as e:
                    print(f"Warning: Failed to capture image {i + 1}: {e}")

    finally:
        cv.destroyAllWindows()

    if not captured_files:
        raise CalibrationError("No images were successfully captured")

    print(f"\nSuccessfully captured {len(captured_files)}/{n_images} images\n")
    return captured_files


def find_chessboard_corners(
    images: List[str],
    nrows: int = CHESSBOARD_ROWS,
    ncols: int = CHESSBOARD_COLS,
    show_preview: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find chessboard corners in calibration images.

    Args:
        images: List of image file paths
        nrows: Number of inner corners in chessboard rows
        ncols: Number of inner corners in chessboard columns
        show_preview: Whether to display detected corners

    Returns:
        Tuple of (object points, image points)

    Raises:
        CalibrationError: If insufficient valid images are found
    """
    print(f"Processing {len(images)} images for chessboard corners...")
    print(f"Chessboard pattern: {ncols}x{nrows} inner corners\n")

    # Termination criteria for corner refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((nrows * ncols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    valid_count = 0

    for fname in images:
        try:
            img = cv.imread(fname)
            if img is None:
                print(f"Warning: Could not read image {fname}")
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv.findChessboardCorners(gray, (ncols, nrows), None)

            if ret:
                objpoints.append(objp)

                # Refine corner positions to sub-pixel accuracy
                corners_refined = cv.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                imgpoints.append(corners_refined)

                valid_count += 1
                print(f"✓ Found corners in {fname}")

                # Display the corners
                if show_preview:
                    cv.drawChessboardCorners(
                        img, (ncols, nrows), corners_refined, ret
                    )
                    cv.imshow('Detected Corners', img)
                    cv.waitKey(PREVIEW_WAIT_MS)
            else:
                print(f"✗ No corners found in {fname}")

        except Exception as e:
            print(f"Warning: Error processing {fname}: {e}")

    if show_preview:
        cv.destroyAllWindows()

    print(f"\nSuccessfully detected corners in {valid_count}/{len(images)} images")

    if valid_count < MIN_VALID_IMAGES:
        raise CalibrationError(
            f"Insufficient valid images. Need at least {MIN_VALID_IMAGES}, "
            f"but only found {valid_count}"
        )

    return objpoints, imgpoints


def calibrate_camera(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    image_size: Tuple[int, int]
) -> Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Perform camera calibration.

    Args:
        objpoints: 3D object points
        imgpoints: 2D image points
        image_size: Image size (width, height)

    Returns:
        Tuple of (ret, camera_matrix, dist_coeffs, rvecs, tvecs)

    Raises:
        CalibrationError: If calibration fails
    """
    print("\nComputing camera calibration parameters...")

    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )

        if not ret:
            raise CalibrationError("Camera calibration failed")

        return ret, mtx, dist, rvecs, tvecs

    except Exception as e:
        raise CalibrationError(f"Calibration computation failed: {e}")


def compute_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    mtx: np.ndarray,
    dist: np.ndarray
) -> float:
    """
    Compute mean reprojection error.

    Args:
        objpoints: 3D object points
        imgpoints: 2D image points
        rvecs: Rotation vectors
        tvecs: Translation vectors
        mtx: Camera matrix
        dist: Distortion coefficients

    Returns:
        Mean reprojection error
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    return mean_error / len(objpoints)


def save_calibration(
    ret: bool,
    mtx: np.ndarray,
    dist: np.ndarray,
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    output_dir: str = "."
) -> str:
    """
    Save calibration parameters to a pickle file.

    Args:
        ret: Calibration return value
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
        output_dir: Output directory for calibration file

    Returns:
        Path to saved calibration file

    Raises:
        CalibrationError: If saving fails
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cal_filename = output_path / f"calibration_{datetime.now():%Y-%m-%d_%H-%M-%S}.pkl"

        with open(cal_filename, 'wb') as f:
            pickle.dump(ret, f)
            pickle.dump(mtx, f)
            pickle.dump(dist, f)
            pickle.dump(rvecs, f)
            pickle.dump(tvecs, f)

        print(f"\nCalibration saved to: {cal_filename}")
        return str(cal_filename)

    except Exception as e:
        raise CalibrationError(f"Failed to save calibration: {e}")


def test_undistortion(
    test_image_path: str,
    mtx: np.ndarray,
    dist: np.ndarray,
    output_path: Optional[str] = None
) -> None:
    """
    Test calibration by undistorting an image.

    Args:
        test_image_path: Path to test image
        mtx: Camera matrix
        dist: Distortion coefficients
        output_path: Optional output path for undistorted image

    Raises:
        CalibrationError: If test fails
    """
    try:
        img = cv.imread(test_image_path)
        if img is None:
            raise CalibrationError(f"Could not read test image: {test_image_path}")

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        if output_path is None:
            output_path = f"{Path(test_image_path).stem}_undistorted.png"

        cv.imwrite(output_path, dst)
        print(f"Undistorted test image saved to: {output_path}")

    except Exception as e:
        raise CalibrationError(f"Undistortion test failed: {e}")


def main():
    """Main calibration workflow."""
    try:
        # Step 1: Capture calibration images
        print("Starting camera calibration workflow...\n")
        captured_files = capture_calibration_images(
            n_images=N_IMAGES,
            wait_time=WAIT_TIME
        )

        # Step 2: Find all calibration images
        images = sorted(glob.glob('calibrate_image*.jpg'))
        if not images:
            raise CalibrationError("No calibration images found")
        print(f"Found {len(images)} calibration images\n")

        # Step 3: Find chessboard corners
        objpoints, imgpoints = find_chessboard_corners(
            images,
            nrows=CHESSBOARD_ROWS,
            ncols=CHESSBOARD_COLS
        )

        # Step 4: Get image size from first valid image
        sample_img = cv.imread(images[0])
        if sample_img is None:
            raise CalibrationError(f"Could not read sample image: {images[0]}")
        gray_shape = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY).shape
        image_size = gray_shape[::-1]  # (width, height)

        # Step 5: Calibrate camera
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(
            objpoints, imgpoints, image_size
        )

        # Step 6: Compute reprojection error
        mean_error = compute_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, mtx, dist
        )
        print(f"\nReprojection error: {mean_error:.3f} pixels")

        # Provide feedback on calibration quality
        if mean_error < 0.5:
            quality = "Excellent"
        elif mean_error < 1.0:
            quality = "Good"
        elif mean_error < 2.0:
            quality = "Acceptable"
        else:
            quality = "Poor - consider recalibrating"
        print(f"Calibration quality: {quality}")

        # Step 7: Save calibration
        cal_file = save_calibration(ret, mtx, dist, rvecs, tvecs)

        # Step 8: Test undistortion with last captured image
        if images:
            test_img = images[-1]
            print(f"\nTesting calibration with image: {test_img}")
            test_undistortion(test_img, mtx, dist)

        print("\n✓ Calibration completed successfully!")
        return 0

    except CalibrationError as e:
        print(f"\nCalibration Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
