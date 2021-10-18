import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


#rad1 = 2.52*10^-2
#rad2 = -6.14*10^-2
cameraMatrix = np.array([[2887.13,0,1799.84],
						[0,2883.59,1366.89],
						[0.0,0.0,1.0]])

img = cv.imread('../Photos/IMG_20200325_141911.jpg')
dst = cv.undistort(img, cameraMatrix, None, None, cameraMatrix) #first None is dist


#plt.figure(1)
#plt.imshow(img)

plt.figure(2)
plt.imshow(dst)
print(dst[4,9])
print(img[4,9])
plt.show()
plt.pause(0)