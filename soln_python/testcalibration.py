import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

cameraMatrix = np.array([[2887.13,0,1799.84],
						[0,2883.59,1366.89],
						[0.0,0.0,1.0]])

cameraMatrix2 = np.array([[3200,0,10500],
						[0,3600,1200],
						[0.0,0.0,1.0]])
img = cv.imread('../Photos/chessboard.jpg')
dst = cv.undistort(img, cameraMatrix2, None, None) #first None is dist




plt.figure(2)
plt.imshow(dst)
plt.show()
plt.pause(0)