import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

img1 = cv.imread('data/im1.png')
img2 = cv.imread('data/im2.png')
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

#Default
use_sift = True
use_orb = not use_sift

if (len(sys.argv) > 1):
	if (sys.argv[1] == "orb"):
		use_orb = True
		use_sift = False
else:
	print("Set command argument to change detector")

#Detectors
if use_sift:
	print("detector: SIFT")
	#Sift object
	alpha = 0.75
	sift = cv.xfeatures2d.SIFT_create()#(0,3,0.04, 10,1.4)

	#Detection - SIFT
	kp1, des1 = sift.detectAndCompute(gray1,None) #Descriptor also computed
	kp2, des2 = sift.detectAndCompute(gray2,None) #Descriptor also computed

	#Initialize FLANN for SIFT
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
elif use_orb:
	print("detector: ORB")
	alpha = 0.9
	orb = cv.ORB_create()

	#orb for img1
	kp1 = orb.detect(gray1,None)
	kp1, des1 = orb.compute(gray1, kp1)

	#orb for img2
	kp2 = orb.detect(gray2,None)
	kp2, des2 = orb.compute(gray2, kp2)

	#Initialize FLANN for ORB
	FLANN_INDEX_LSH = 6
	index_params= dict(algorithm = FLANN_INDEX_LSH,
	                   table_number = 6, # 12
	                   key_size = 12,     # 20
	                   multi_probe_level = 1) #2search_params = dict(checks=50)   # or pass empty dictionary


# FLANN feature matching
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Lowes ratio test - Only keep good matches (stored in pts1, pts2)
pts1 = []
pts2 = []
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < alpha*n.distance:
        matchesMask[i]=[1,0]
        pts1.append([kp1[n.queryIdx].pt[0], kp1[n.queryIdx].pt[1]])
        pts2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

# Visualize matching
draw_params = dict(#singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

match_img = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**draw_params)

plt.figure(1)
plt.imshow(match_img)

# Remove points that are not fulfilling epipolar constraint
pts1 = np.int32(np.around(pts1))
pts2 = np.int32(np.around(pts2))

F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, ransacReprojThreshold=0.3, confidence=0.7) # 0.05, 0.99999

pts1 = pts1[mask.ravel()==1] #select inlier points
pts2 = pts2[mask.ravel()==1]

if use_orb:
	np.savetxt('data/matchesORB.txt', np.hstack((pts1,pts2)))
elif use_sift:
	np.savetxt('data/matchesSIFT.txt', np.hstack((pts1,pts2)))

####   Visualizing points after fulfulling epipolar constraints
#IMAGE 1
plt.figure(2)
plt.plot(pts1[:,0], pts1[:,1], 'ro')
#numbers in plot
for p in range(len(pts1)):
	plt.text(pts1[p][0], pts1[p][1], str(p+1), color="black", fontsize=10)
plt.imshow(gray1, cmap='gray')

# IMAGE 2
plt.figure(3)
plt.plot(pts2[:,0], pts2[:,1], 'bo')
#numbers in plot
for p in range(len(pts2)):
	plt.text(pts2[p][0], pts2[p][1], str(p+1), color="black", fontsize=10)
plt.imshow(gray2, cmap='gray')

plt.show()
plt.pause(0)

#https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html
#https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
