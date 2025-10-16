########################################################################
target_Hz = 1
recording_minutes = 10
iso = 200
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

from picamera2 import Picamera2
import cv2 as cv

def timestamp():
    t = datetime.now()
    return str(t.year) + "," + str(t.month) + "," + str(t.day)  + "," + \
        str(t.hour) + "," + str(t.minute) + "," + str(t.second) + "," + \
        str(t.microsecond)

samples = int(target_Hz * 60.0 * recording_minutes)

print(f'Recording with a target of {target_Hz} Hz '
      f'for {recording_minutes} minutes.\n')

# Directory for holding temporary images
temp_dir = tempfile.gettempdir() + '/' + datetime.now().isoformat()
os.mkdir(temp_dir)

# Camera setup
camera = Picamera2()
config = camera.create_still_configuration(main={"size": (1024, 768)})
camera.configure(config)

# Set initial controls - ISO converted to AnalogueGain (ISO/100)
analogue_gain = iso / 100.0
camera.set_controls({
    "AnalogueGain": analogue_gain,
    "AeEnable": False,  # Disable auto exposure
    "AwbEnable": False  # Disable auto white balance
})

# Start camera and wait for settings to settle
camera.start()
time.sleep(2)

# Get current exposure time and AWB gains to lock them
metadata = camera.capture_metadata()
current_exposure = metadata.get("ExposureTime", 33000)  # Default fallback
current_awb_gains = metadata.get("AwbGains", (1.0, 1.0))  # Default fallback

# Now lock the exposure and AWB values
camera.set_controls({
    "ExposureTime": current_exposure,
    "AnalogueGain": analogue_gain,
    "AeEnable": False,
    "AwbEnable": False
})

# Open data file for writing
outfile = datetime.now().isoformat()
f = open(outfile + ".csv","w")

# Write header
f.write("Year,Month,Day,Hour,Minute,Second,Microsecond,")
f.write("Thermistor_Temp,Thermistor_Temp_NIST,Analog\n")

# Flash LED
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(16, GPIO.OUT)

spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# MCP3008
cs22 = digitalio.DigitalInOut(board.D22)
mcp = MCP.MCP3008(spi, cs22)

# Thermocouple
cs5 = digitalio.DigitalInOut(board.D5)

# Countdown and start recording
GPIO.output(16,GPIO.HIGH)
print("Recording in...")
for i in range(countdown, 0, -1):
    print(i)
    time.sleep(1)
GPIO.output(16,GPIO.LOW)

print("Recording begins.")

start_time = datetime.now()

ii = 1

while (datetime.now() <= start_time + timedelta(minutes=recording_minutes)):
    now = datetime.now()

    # Record an image to the temporary directory
    # Note: picamera2 doesn't have annotate_text, timestamp annotation removed
    camera.capture_file(temp_dir + '/img' + str(ii).zfill(4) + '.jpg')

    # Read analog in
    chan0 = AnalogIn(mcp, MCP.P0)
    analog_temp = chan0.value / 1000.0

    # Read thermocouple
    max31855 = adafruit_max31855.MAX31855(spi, cs5)

    print(f'Thermistor Temp.: {max31855.temperature} C\t'
          f'Analog Temp.: {analog_temp}\t'
          f'{now.strftime("%Y-%m-%d %H:%M:%S.%f")}')
    tc = adafruit_max31855.MAX31855(spi, cs5)

    f.write(timestamp() + "," + str(tc.temperature) + "," + \
            str(tc.temperature_NIST) + "," + str(analog_temp) + "\n")

    # Fix sleep time
    d = datetime.now() - now # time for the loop so far
    obs_Hz = 1 / d.total_seconds()
    print('Observed recording frequency: {:3f}\n'.format(obs_Hz))
#     d_Hz = target_Hz - obs_Hz
#     Hz = target_Hz + d_Hz
#     print('New frequency: {:3f}'.format(Hz))
#     sleep_time = 1 / Hz
#     time.sleep(sleep_time)

    ii = ii + 1

f.close()

# Stop camera
camera.stop()

print("\nRecording complete. Processing video.")

# Load calibration
print('\nChoose the calibration .pkl file.')
Tk().withdraw()
cal_file = askopenfilename()

F = open(cal_file, 'rb')
ret = pickle.load(F)
mtx = pickle.load(F)
dist = pickle.load(F)
rvecs = pickle.load(F)
tvecs = pickle.load(F)
F.close()

# Load image list
images = glob.glob(temp_dir + '/img*.jpg')
images.sort()

# Undistort
print('Undistorting images.')
for f in images:
    img = cv.imread(f)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    cv.imwrite(f, dst)

# Write movie
print('Writing movie.')
os.system('ffmpeg -framerate 1 -i ' + temp_dir +
          '/img%04d.jpg outfile.avi >/dev/null 2>&1')
os.rename('outfile.avi', outfile + '.avi')

# Cleanup
for f in images:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
os.rmdir(temp_dir)

print('\nRecording complete. Files saved as ' + outfile +'[.csv, .avi]')
