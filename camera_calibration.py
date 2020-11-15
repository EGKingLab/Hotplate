# camera_calibration.py
# https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

from time import sleep
from picamera import PiCamera
import numpy as np
import cv2 as cv
import glob
import pickle
from datetime import datetime

# Campture images
camera = PiCamera()
camera.resolution = (1024, 768)

n_images = 15
wait_time = 4

print("Taking " + str(n_images) + " calibration images.")

for ii in range(n_images):
    camera.start_preview()
    sleep(wait_time)
    camera.capture('calibrate_image' + str(ii) + '.jpg')

camera.stop_preview()

# Find corners
nrow = 13
ncol = 19

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((nrow*ncol,3), np.float32)
objp[:,:2] = np.mgrid[0:ncol,0:nrow].T.reshape(-1,2)
objpoints = [] 
imgpoints = [] 
images = glob.glob('calibrate_image*.jpg')

print("Found {} images".format(len(images)))
print("Computing camera parameters")

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray, (ncol,nrow), None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        
        cv.drawChessboardCorners(img, (ncol,nrow), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("Total error: {:3f}".format(mean_error/len(objpoints)))

# Save calibration
cal_file = datetime.now().isoformat() + ".pkl"
F = open(cal_file, 'wb')
pickle.dump(ret, F)
pickle.dump(mtx, F)
pickle.dump(dist, F)
pickle.dump(rvecs, F)
pickle.dump(tvecs, F)
F.close()

# Undistort an image
img = cv.imread('calibrate_image0.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite('calibrate_image0_calibresult.png', dst)
