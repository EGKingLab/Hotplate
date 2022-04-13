from time import sleep
from picamera import PiCamera
import pickle
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

camera = PiCamera()
camera.resolution = (1024, 768)

camera.start_preview()
sleep(10)
camera.capture('foo.jpg')
camera.stop_preview()

# Load calibration
print('Choose the calibration .pkl file.')
Tk().withdraw()
cal_file = askopenfilename()

F = open(cal_file, 'rb')
ret = pickle.load(F)
mtx = pickle.load(F)
dist = pickle.load(F)
rvecs = pickle.load(F)
tvecs = pickle.load(F)
F.close()

# Undistort foo
img = cv.imread('foo.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite('foo_undistort.png', dst)
