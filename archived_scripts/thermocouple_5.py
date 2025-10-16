########################################################################
target_Hz = 1
recording_minutes = 10
iso = 200
countdown = 5
########################################################################

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

from picamera import PiCamera
import cv2 as cv

def timestamp():
    t = datetime.now()
    return str(t.year) + "," + str(t.month) + "," + str(t.day)  + "," + \
        str(t.hour) + "," + str(t.minute) + "," + str(t.second) + "," + \
        str(t.microsecond)

samples = int(target_Hz * 60.0 * recording_minutes)

print('Recording with a target of {} Hz for {} minutes.\n'.format(target_Hz, recording_minutes))

# Directory for holding temporary images
temp_dir = tempfile.gettempdir() + '/' + datetime.now().isoformat()
os.mkdir(temp_dir)

# Camera setup
camera = PiCamera(resolution=(1024, 768), framerate=30)
camera.iso = iso
# Wait for the automatic gain control to settle
time.sleep(2)
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g

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
    #camera.annotate_text = now.isoformat()
    camera.capture(temp_dir + '/img' + str(ii).zfill(4) + '.jpg')
    
    # Read analog in
    chan0 = AnalogIn(mcp, MCP.P0)
    analog_temp = chan0.value / 1000.0

    # Read thermocouple
    max31855 = adafruit_max31855.MAX31855(spi, cs5)

    print('Thermistor Temp.: {} C\tAnalog Temp.: {}\t{}'.format(max31855.temperature,
                                                                analog_temp,
                                                                now.strftime('%Y-%m-%d %H:%M:%S.%f')))
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
os.system('ffmpeg -framerate 1 -i ' + temp_dir + '/img%04d.jpg outfile.avi >/dev/null 2>&1')
os.rename('outfile.avi', outfile + '.avi')

# Cleanup
for f in images:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
os.rmdir(temp_dir)

print('\nRecording complete. Files saved as ' + outfile +'[.csv, .avi]')
