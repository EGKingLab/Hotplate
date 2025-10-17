########################################################################
target_Hz = 1
recording_minutes = 10
iso = 400
countdown = 5
########################################################################

# https://forums.raspberrypi.com/viewtopic.php?t=367558

import time
from datetime import datetime
from datetime import timedelta
import board
import busio
import digitalio
import adafruit_max31855

import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

import RPi.GPIO as GPIO

import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import tempfile
import glob
import os
import sys
import signal
import atexit
import subprocess

from picamera2 import Picamera2
import cv2 as cv

# Hardware configuration constants
GPIO_LED_PIN = 16
SPI_CS_MCP3008_PIN = 22  # D22
SPI_CS_THERMOCOUPLE_PIN = 5  # D5

# Data collection configuration constants
ANALOG_CONVERSION_FACTOR = 1000.0
CSV_FLUSH_INTERVAL = 10  # samples
DEFAULT_EXPOSURE_TIME = 50000  # microseconds
MAX_CONSECUTIVE_ERRORS = 10

# Camera configuration constants
CAMERA_RESOLUTION = (1024, 768)

def timestamp():
    """Generate a CSV-formatted timestamp from current datetime."""
    return datetime.now().strftime("%Y,%m,%d,%H,%M,%S,%f")

# Global resources for cleanup
camera = None
csv_file = None
temp_dir = None
gpio_initialized = False

def cleanup_resources():
    """Cleanup all resources safely"""
    global camera, csv_file, temp_dir, gpio_initialized

    if camera is not None:
        try:
            camera.stop()
            print("Camera stopped.")
        except Exception as e:
            print(f"Error stopping camera: {e}")

    if csv_file is not None and not csv_file.closed:
        try:
            csv_file.close()
            print("CSV file closed.")
        except Exception as e:
            print(f"Error closing CSV file: {e}")

    if gpio_initialized:
        try:
            GPIO.cleanup()
            print("GPIO cleaned up.")
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print(f"\n\nReceived signal {sig}. Cleaning up...")
    cleanup_resources()
    sys.exit(0)

def validate_config():
    """Validate configuration parameters"""
    if target_Hz <= 0:
        raise ValueError(f"target_Hz must be positive, got {target_Hz}")
    if recording_minutes <= 0:
        raise ValueError(f"recording_minutes must be positive, got {recording_minutes}")
    if iso < 100 or iso > 1600:
        print(f"Warning: ISO {iso} may be outside typical range (100-1600)")
    if countdown < 0:
        raise ValueError(f"countdown must be non-negative, got {countdown}")

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Validate configuration
try:
    validate_config()
except ValueError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

samples = int(target_Hz * 60.0 * recording_minutes)

print(f'Recording with a target of {target_Hz} Hz '
      f'for {recording_minutes} minutes.\n')

# Directory for holding temporary images
temp_dir = os.path.join(tempfile.gettempdir(), datetime.now().isoformat())
try:
    os.mkdir(temp_dir)
    print(f"Created temporary directory: {temp_dir}")
except OSError as e:
    print(f"Error: Failed to create temporary directory '{temp_dir}': {e}")
    sys.exit(1)

# Camera setup
try:
    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": CAMERA_RESOLUTION})
    camera.configure(config)

    # Set initial controls - ISO converted to AnalogueGain (ISO/100)
    analogue_gain = iso / 100.0
    camera.set_controls({
        "AnalogueGain": analogue_gain,
        "AeEnable": True,  # Enable auto exposure
        "AwbEnable": True  # Enable auto white balance
    })

    # Start camera and wait for settings to settle
    camera.start()
    time.sleep(2)

    # Get current exposure time and AWB gains to lock them
    metadata = camera.capture_metadata()
    if not metadata:
        raise RuntimeError("Failed to capture camera metadata")

    current_exposure = metadata.get("ExposureTime", DEFAULT_EXPOSURE_TIME)  # Default fallback
    current_awb_gains = metadata.get("AwbGains", (1.0, 1.0))  # Default fallback

    # Now lock the exposure and AWB values
    camera.set_controls({
        "ExposureTime": current_exposure,
        "AnalogueGain": analogue_gain,
        "AeEnable": False,
        "AwbEnable": False
    })

    print(f"Camera initialized: Exposure={current_exposure}, Gain={analogue_gain:.2f}")

except Exception as e:
    print(f"Error: Failed to initialize camera: {e}")
    cleanup_resources()
    sys.exit(1)

# Open data file for writing
outfile = datetime.now().isoformat()
csv_filename = outfile + ".csv"

try:
    csv_file = open(csv_filename, "w")
    # Write header
    csv_file.write("Year,Month,Day,Hour,Minute,Second,Microsecond,")
    csv_file.write("Thermistor_Temp,Thermistor_Temp_NIST,Analog\n")
    print(f"CSV file created: {csv_filename}")
except IOError as e:
    print(f"Error: Failed to create CSV file '{csv_filename}': {e}")
    cleanup_resources()
    sys.exit(1)

# Flash LED
try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GPIO_LED_PIN, GPIO.OUT)
    gpio_initialized = True
    print("GPIO initialized")
except Exception as e:
    print(f"Error: Failed to initialize GPIO: {e}")
    cleanup_resources()
    sys.exit(1)

# Initialize SPI and sensors
try:
    spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

    # MCP3008
    cs_mcp3008 = digitalio.DigitalInOut(getattr(board, f'D{SPI_CS_MCP3008_PIN}'))
    mcp = MCP.MCP3008(spi, cs_mcp3008)

    # Thermocouple
    cs_thermocouple = digitalio.DigitalInOut(getattr(board, f'D{SPI_CS_THERMOCOUPLE_PIN}'))

    print("SPI and sensors initialized")
except Exception as e:
    print(f"Error: Failed to initialize SPI/sensors: {e}")
    cleanup_resources()
    sys.exit(1)

# Countdown and start recording
GPIO.output(GPIO_LED_PIN, GPIO.HIGH)
print("Recording in...")
for i in range(countdown, 0, -1):
    print(i)
    time.sleep(1)
GPIO.output(GPIO_LED_PIN, GPIO.LOW)

print("Recording begins.")

start_time = datetime.now()
sample_count = 0
error_count = 0

try:
    while (datetime.now() <= start_time + timedelta(minutes=recording_minutes)):
        now = datetime.now()

        try:
            # Record an image to the temporary directory
            # Note: picamera2 doesn't have annotate_text, timestamp annotation removed
            img_filename = os.path.join(temp_dir, f'img{str(sample_count + 1).zfill(4)}.jpg')
            camera.capture_file(img_filename)

            # Read analog in
            chan0 = AnalogIn(mcp, MCP.P0)
            analog_temp = chan0.value / ANALOG_CONVERSION_FACTOR

            # Read thermocouple (only once, not twice)
            thermocouple = adafruit_max31855.MAX31855(spi, cs_thermocouple)

            print(f'Thermistor Temp.: {thermocouple.temperature} C\t'
                  f'Analog Temp.: {analog_temp}\t'
                  f'{now.strftime("%Y-%m-%d %H:%M:%S.%f")}')

            csv_file.write(f"{timestamp()},{thermocouple.temperature},{thermocouple.temperature_NIST},{analog_temp}\n")

            sample_count += 1

            # Periodic flush to prevent data loss
            if sample_count % CSV_FLUSH_INTERVAL == 0:
                csv_file.flush()

        except Exception as e:
            error_count += 1
            print(f"Warning: Error in sample {sample_count + 1}: {e}")
            # Continue recording despite errors
            if error_count > MAX_CONSECUTIVE_ERRORS:
                print("Error: Too many consecutive errors, stopping recording")
                break

        # Adaptive sleep to maintain target frequency
        loop_duration = (datetime.now() - now).total_seconds()
        target_period = 1.0 / target_Hz  # 1.0 seconds for 1 Hz
        sleep_time = target_period - loop_duration

        if sleep_time > 0:
            time.sleep(sleep_time)
            actual_Hz = target_Hz
        else:
            # Loop took longer than target period
            actual_Hz = 1.0 / loop_duration

        print(f'Loop duration: {loop_duration:.3f}s | Sleep: {max(0, sleep_time):.3f}s | Actual Hz: {actual_Hz:.3f}\n')

finally:
    # Ensure resources are cleaned up
    print(f"\nRecording stopped. Collected {sample_count} samples with {error_count} errors.")
    if csv_file and not csv_file.closed:
        csv_file.close()
        print("CSV file closed.")
    if camera:
        camera.stop()
        print("Camera stopped.")

print("\nRecording complete. Processing video.")

# Load calibration
print('\nChoose the calibration .pkl file.')
Tk().withdraw()
cal_file = askopenfilename(filetypes=[("Pickle files", "*.pkl"),
                                      ("All files", "*.*")])

if not cal_file:
    print("Warning: No calibration file selected. Skipping image undistortion.")
    skip_undistortion = True
else:
    skip_undistortion = False
    try:
        with open(cal_file, 'rb') as F:
            ret = pickle.load(F)
            mtx = pickle.load(F)
            dist = pickle.load(F)
            rvecs = pickle.load(F)
            tvecs = pickle.load(F)
        print(f"Calibration loaded from {cal_file}")
    except (IOError, pickle.UnpicklingError, EOFError) as e:
        print(f"Error: Failed to load calibration file: {e}")
        print("Skipping image undistortion.")
        skip_undistortion = True
    except Exception as e:
        print(f"Error: Unexpected error loading calibration: {e}")
        print("Skipping image undistortion.")
        skip_undistortion = True

start = time.time()

# Load image list
images = glob.glob(os.path.join(temp_dir, 'img*.jpg'))
images.sort()

if not images:
    print("Warning: No images found in temporary directory. Skipping video processing.")
else:
    print(f"Found {len(images)} images to process.")

    # Undistort
    if not skip_undistortion:
        print('Undistorting images.')
        undistort_errors = 0
        for img_path in images:
            try:
                img = cv.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_path}")

                h,  w = img.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

                dst = cv.undistort(img, mtx, dist, None, newcameramtx)
                if not cv.imwrite(img_path, dst):
                    raise ValueError(f"Failed to write undistorted image: {img_path}")

            except Exception as e:
                undistort_errors += 1
                print(f"Warning: Error undistorting {img_path}: {e}")

        if undistort_errors > 0:
            print(f"Warning: {undistort_errors} images failed to undistort")

    # Write movie
    print('Writing movie.')
    video_filename = outfile + '.avi'
    try:
        input_pattern = os.path.join(temp_dir, 'img%04d.jpg')
        result = subprocess.run(
            ['ffmpeg', '-framerate', '1', '-i', input_pattern,
             video_filename, '-y'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: ffmpeg failed with exit code {result.returncode}")
            print(f"Error output: {result.stderr}")
        elif os.path.exists(video_filename):
            print(f"Video created successfully: {video_filename}")
        else:
            print("Warning: ffmpeg command completed but video file not found")

    except Exception as e:
        print(f"Error: Failed to create video: {e}")

# Cleanup
if images:
    cleanup_errors = 0
    for img_path in images:
        try:
            os.remove(img_path)
        except OSError as e:
            cleanup_errors += 1
            print(f"Warning: Failed to remove {img_path}: {e.strerror}")

    if cleanup_errors > 0:
        print(f"Warning: {cleanup_errors} temporary files could not be removed")

try:
    os.rmdir(temp_dir)
    print(f"Temporary directory removed: {temp_dir}")
except OSError as e:
    print(f"Warning: Failed to remove temporary directory '{temp_dir}': {e.strerror}")

# Final summary
print(f'\nRecording complete. Files saved as {outfile}[.csv, .avi]')
print(f'Samples collected: {sample_count}')
if error_count > 0:
    print(f'Errors encountered: {error_count}')

end = time.time()
elapsed_seconds = end - start
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = elapsed_seconds % 60
print(f'Post-processing time: {hours:02d}:{minutes:02d}:{seconds:05.2f}')
