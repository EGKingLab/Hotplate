import tempfile
import glob
import os
import cv2 as cv
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

temp_dir = "2021-04-06_RNAi-32871.3954-1_group1"
undistort_images = False

if undistort_images:
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
