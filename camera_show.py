#!/usr/bin/env python3
"""
Camera Live Feed Display

Display live video feed from Raspberry Pi camera using picamera2.
Press 'q' to quit, 's' to save snapshot, 'i' to toggle info overlay.
"""

import sys
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Deque, Tuple

import cv2 as cv
import numpy as np
from picamera2 import Picamera2


# Configuration constants
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 768
WINDOW_NAME = 'Live Camera Feed'
FPS_UPDATE_INTERVAL = 30  # Update FPS display every N frames
SNAPSHOT_DIR = "snapshots"

# Overlay styling constants
OVERLAY_FONT = cv.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 0.6
OVERLAY_FONT_THICKNESS = 2
OVERLAY_LINE_HEIGHT = 25
OVERLAY_MARGIN = 10
OVERLAY_ALPHA = 0.6

# Info overlay constants
INFO_OVERLAY_WIDTH = 300
INFO_OVERLAY_COLOR = (0, 255, 0)  # Green (BGR)

# Help overlay constants
HELP_OVERLAY_WIDTH = 400
HELP_OVERLAY_FONT_SCALE = 0.7
HELP_OVERLAY_LINE_HEIGHT = 30
HELP_OVERLAY_MARGIN = 20
HELP_OVERLAY_COLOR = (0, 255, 255)  # Yellow (BGR)
HELP_OVERLAY_ALPHA = 0.8

# Keyboard commands
KEYBOARD_COMMANDS = {
    'q': 'Quit',
    's': 'Save Snapshot',
    'i': 'Toggle Info Overlay',
    'h': 'Toggle This Help'
}


class CameraError(Exception):
    """Custom exception for camera errors."""
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
        CameraError: If camera initialization fails
        ValueError: If width or height are not positive
    """
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Width and height must be positive, got {width}x{height}"
        )

    camera = None
    try:
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (width, height)}
        )
        camera.configure(config)
        camera.start()
        yield camera
    except ValueError:
        raise
    except Exception as e:
        raise CameraError(f"Camera initialization failed: {e}")
    finally:
        if camera is not None:
            try:
                camera.stop()
                camera.close()
            except Exception as e:
                print(f"Warning: Error during camera cleanup: {e}")


def calculate_fps(frame_times: Deque[float], current_time: float) -> float:
    """
    Calculate current FPS based on recent frame times.

    Args:
        frame_times: Deque of recent frame timestamps
        current_time: Current timestamp

    Returns:
        Current FPS
    """
    if len(frame_times) < 2:
        return 0.0

    time_diff = current_time - frame_times[0]
    if time_diff > 0:
        return len(frame_times) / time_diff
    return 0.0


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font: int = OVERLAY_FONT,
    font_scale: float = OVERLAY_FONT_SCALE,
    color: Tuple[int, int, int] = INFO_OVERLAY_COLOR,
    thickness: int = OVERLAY_FONT_THICKNESS
) -> None:
    """
    Draw text on frame (modifies frame in-place).

    Args:
        frame: Frame to draw on
        text: Text to draw
        position: (x, y) position for text
        font: OpenCV font type
        font_scale: Font scale factor
        color: Text color (BGR)
        thickness: Text thickness
    """
    cv.putText(frame, text, position, font, font_scale, color, thickness)


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    width: int,
    height: int,
    frame_count: int
) -> None:
    """
    Draw information overlay on frame (modifies frame in-place).

    Args:
        frame: Input frame (modified in-place)
        fps: Current frames per second
        width: Frame width
        height: Frame height
        frame_count: Total frame count
    """
    # Background for text
    overlay_height = 4 * OVERLAY_LINE_HEIGHT + OVERLAY_MARGIN
    overlay = frame[0:overlay_height, 0:INFO_OVERLAY_WIDTH].copy()
    cv.rectangle(
        overlay, (0, 0), (INFO_OVERLAY_WIDTH, overlay_height),
        (0, 0, 0), -1
    )

    # Blend overlay with original
    blended = cv.addWeighted(
        overlay, OVERLAY_ALPHA,
        frame[0:overlay_height, 0:INFO_OVERLAY_WIDTH],
        1 - OVERLAY_ALPHA, 0
    )
    frame[0:overlay_height, 0:INFO_OVERLAY_WIDTH] = blended

    # Draw text
    y_position = OVERLAY_MARGIN + 20
    draw_text_with_background(
        frame, f"FPS: {fps:.1f}", (OVERLAY_MARGIN, y_position)
    )

    y_position += OVERLAY_LINE_HEIGHT
    draw_text_with_background(
        frame, f"Resolution: {width}x{height}",
        (OVERLAY_MARGIN, y_position)
    )

    y_position += OVERLAY_LINE_HEIGHT
    draw_text_with_background(
        frame, f"Frames: {frame_count}",
        (OVERLAY_MARGIN, y_position)
    )

    y_position += OVERLAY_LINE_HEIGHT
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw_text_with_background(
        frame, timestamp, (OVERLAY_MARGIN, y_position)
    )


def save_snapshot(
    frame: np.ndarray,
    output_dir: str = SNAPSHOT_DIR,
    snapshot_count: int = 0
) -> str:
    """
    Save a snapshot of the current frame.

    Args:
        frame: Frame to save
        output_dir: Directory to save snapshots
        snapshot_count: Counter to prevent filename collisions

    Returns:
        Path to saved snapshot

    Raises:
        CameraError: If saving fails
        ValueError: If output directory cannot be created
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Verify directory is writable
        if not output_path.is_dir():
            raise ValueError(
                f"Output path exists but is not a directory: "
                f"{output_dir}"
            )

        # Add counter and milliseconds to prevent collisions
        timestamp = datetime.now()
        ms = timestamp.microsecond // 1000
        filename = output_path / (
            f"snapshot_{timestamp:%Y-%m-%d_%H-%M-%S}-"
            f"{ms:03d}_{snapshot_count:04d}.png"
        )

        success = cv.imwrite(str(filename), frame)
        if not success:
            raise CameraError("cv.imwrite failed to save image")

        return str(filename)

    except ValueError:
        raise
    except Exception as e:
        raise CameraError(f"Failed to save snapshot: {e}")


def show_help_overlay(frame: np.ndarray) -> np.ndarray:
    """
    Show help overlay with key commands.

    Args:
        frame: Input frame

    Returns:
        Frame with help overlay (new copy)
    """
    result = frame.copy()

    # Build help text from KEYBOARD_COMMANDS
    help_text = ["Keyboard Commands:"]
    for key, description in KEYBOARD_COMMANDS.items():
        help_text.append(f"  {key} - {description}")
    help_text.append("")
    help_text.append("Press any key to continue...")

    # Calculate overlay size
    num_lines = len(help_text)
    overlay_height = (
        num_lines * HELP_OVERLAY_LINE_HEIGHT + 2 * HELP_OVERLAY_MARGIN
    )
    overlay_width = HELP_OVERLAY_WIDTH

    # Center the overlay
    h, w = frame.shape[:2]
    x_start = (w - overlay_width) // 2
    y_start = (h - overlay_height) // 2

    # Draw background
    y_end = y_start + overlay_height
    x_end = x_start + overlay_width
    overlay_region = result[y_start:y_end, x_start:x_end].copy()
    cv.rectangle(
        overlay_region, (0, 0), (overlay_width, overlay_height),
        (0, 0, 0), -1
    )

    # Blend overlay with original
    blended = cv.addWeighted(
        overlay_region, HELP_OVERLAY_ALPHA,
        result[y_start:y_end, x_start:x_end],
        1 - HELP_OVERLAY_ALPHA, 0
    )
    result[y_start:y_end, x_start:x_end] = blended

    # Draw text
    y_position = y_start + HELP_OVERLAY_MARGIN + 20
    for line in help_text:
        cv.putText(
            result, line,
            (x_start + HELP_OVERLAY_MARGIN, y_position),
            OVERLAY_FONT, HELP_OVERLAY_FONT_SCALE,
            HELP_OVERLAY_COLOR, OVERLAY_FONT_THICKNESS
        )
        y_position += HELP_OVERLAY_LINE_HEIGHT

    return result


def display_live_feed(
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT
) -> int:
    """
    Display live camera feed with interactive controls.

    Args:
        width: Camera frame width
        height: Camera frame height

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("Starting live camera feed...")
    print(f"Resolution: {width}x{height}")
    print("\nKeyboard Commands:")
    for key, description in KEYBOARD_COMMANDS.items():
        print(f"  {key} - {description}")
    print()

    frame_count = 0
    frame_times: Deque[float] = deque(maxlen=FPS_UPDATE_INTERVAL)
    fps = 0.0
    show_info = True
    show_help = False
    snapshot_count = 0

    try:
        with camera_context(width, height) as camera:
            print("Camera started successfully!")
            print("Press 'h' to show help overlay\n")

            while True:
                try:
                    # Capture frame
                    frame = camera.capture_array()
                    if frame is None:
                        print("Warning: Failed to capture frame")
                        continue

                    frame_count += 1
                    tick_count = cv.getTickCount()
                    tick_freq = cv.getTickFrequency()
                    current_time = tick_count / tick_freq

                    # Update FPS calculation (O(1) with deque)
                    frame_times.append(current_time)

                    if frame_count % FPS_UPDATE_INTERVAL == 0:
                        fps = calculate_fps(frame_times, current_time)

                    # Prepare display frame
                    if show_help:
                        # Help overlay returns a new copy
                        display_frame = show_help_overlay(frame)
                    else:
                        # For info overlay, modify in-place (copy first)
                        if show_info:
                            display_frame = frame.copy()
                            draw_overlay(
                                display_frame, fps, width, height,
                                frame_count
                            )
                        else:
                            display_frame = frame

                    # Display the frame
                    cv.imshow(WINDOW_NAME, display_frame)

                    # Handle key presses
                    key = cv.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        try:
                            snapshot_path = save_snapshot(
                                frame, SNAPSHOT_DIR, snapshot_count
                            )
                            snapshot_count += 1
                            print(f"Snapshot saved: {snapshot_path}")
                        except (CameraError, ValueError) as e:
                            print(f"Error saving snapshot: {e}")
                    elif key == ord('i'):
                        show_info = not show_info
                        status = "enabled" if show_info else "disabled"
                        print(f"Info overlay {status}")
                    elif key == ord('h'):
                        show_help = not show_help

                except Exception as e:
                    print(f"Warning: Error processing frame: {e}")
                    continue

    except (CameraError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nCamera feed interrupted by user")
    finally:
        cv.destroyAllWindows()
        print(f"\nCamera stopped.")
        print(f"Total frames captured: {frame_count}")
        print(f"Snapshots saved: {snapshot_count}")
        if fps > 0:
            print(f"Final FPS: {fps:.1f}")

    return 0


def main():
    """Main entry point."""
    return display_live_feed(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)


if __name__ == "__main__":
    sys.exit(main())
