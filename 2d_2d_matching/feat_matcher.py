import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import yaml



def detect_and_match(gray1, gray2, use_sift, use_orb, detect_num):
	"""
	1) Detects features using SIFT or ORB
	2) Uses FLANN feature matcher
	3) Filter out using Lowes ratio test
	4) Filters out outliers using Ransac (from find fundamental matrix)
	"""

	#Load parameters from param file
	with open("params.yaml", "r") as stream:
		try:
			params = yaml.safe_load(stream)
			print(params)
		except yaml.YAMLError as exc:
			print(exc)

	#Detectors
	if use_sift:
		print("feature detection (SIFT)")

		#Sift object
		sift = cv.xfeatures2d.SIFT_create(contrastThreshold = params["sift"]["contrastThreshold"], sigma = params["sift"]["sigma"]) # 3,  0.04,  10,  1.4

		#Detection - SIFT
		kp1, des1 = sift.detectAndCompute(gray1,None)
		kp2, des2 = sift.detectAndCompute(gray2,None)

		#Initialize FLANN for SIFT
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	elif use_orb:
		print("feature detection (ORB)")
		
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
						   table_number = 6,
						   key_size = 12,
						   multi_probe_level = 1)


	print("FLANN feature matching")
	search_params = dict(checks=50)
	flann = cv.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	#Nearest-neighbor distance ratio test (Lowes ratio test)
	print("Lowes ratio test")
	pts1 = []
	pts2 = []
	matchesMask = [[0,0] for i in range(len(matches))]
	for i,tupl in enumerate(matches):
		m = tupl[0]
		n = tupl[1]
		if m.distance < params["alpha"]*n.distance:
			matchesMask[i]=[1,0]
			pts1.append([kp1[n.queryIdx].pt[0], kp1[n.queryIdx].pt[1]])
			pts2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

	# Visualize matching
	draw_params = dict( matchesMask = matchesMask,
					   flags = cv.DrawMatchesFlags_DEFAULT)

	match_img = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**draw_params)

	plt.figure("All matches")
	plt.imshow(match_img)

	#Remove points that are not fulfilling epipolar constraint
	print("Filtering on fundamental matrix with ransac")
	pts1 = np.int32(np.around(pts1))
	pts2 = np.int32(np.around(pts2))
	_, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, ransacReprojThreshold=params["ransacRepThresh"], confidence=params["conf"])

	#Select inlier points
	pts1 = pts1[mask.ravel()==1]
	pts2 = pts2[mask.ravel()==1]

	print("Writing to file")
	if use_orb:
		np.savetxt('../data/matchesORB'+detect_num+'.txt', np.hstack((pts1,pts2)))
	elif use_sift:
		np.savetxt('../data/matchesSIFT'+detect_num+'.txt', np.hstack((pts1,pts2)))

	return pts1, pts2


#Default
use_sift = True
use_orb = not use_sift

if (len(sys.argv) > 1):
	if (sys.argv[1] == "orb"):
		use_orb = True
		use_sift = False
else:
	print("Set command argument to change detector")

img1 = cv.imread('../data/glosh1.jpg')
img2 = cv.imread('../data/glosh2.jpg')
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

pts1, pts2 = detect_and_match(gray1, gray2, use_sift, use_orb,'1')

####   Visualizing
print("plotting figures")
#IMAGE 1
plt.figure("Detections in image1")
plt.plot(pts1[:,0], pts1[:,1], 'ro')

#numbers in plot
for p in range(len(pts1)):
	plt.text(pts1[p][0], pts1[p][1], str(p+1), color="black", fontsize=10)
plt.imshow(gray1, cmap='gray')

# IMAGE 2
plt.figure("Detections in image2")
plt.plot(pts2[:,0], pts2[:,1], 'bo')

#numbers in plot
for p in range(len(pts2)):
	plt.text(pts2[p][0], pts2[p][1], str(p+1), color="black", fontsize=10)
plt.imshow(gray2, cmap='gray')
plt.show()

plt.pause(0.001)
input("hit something to end")
plt.close('all')

#https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html
#https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
