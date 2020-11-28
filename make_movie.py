base_dir = "2020-11-25_VAL-11466_group1"

import glob
import os
import cv2 as cv
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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
images = glob.glob(base_dir + '/img*.jpg')
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
os.system('ffmpeg -framerate 1 -i ' + base_dir + '/img%04d.jpg outfile.avi >/dev/null 2>&1')
