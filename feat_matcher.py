import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('data/im1.png')
img2 = cv.imread('data/im2.png')
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

#Detectors
sift = cv.xfeatures2d.SIFT_create()#(0,3,0.04, 10,1.4)
orb = cv.ORB_create()

#should change params

#Detection - SIFT
kp1, des1 = sift.detectAndCompute(gray1,None) #Descriptor also computed
kp2, des2 = sift.detectAndCompute(gray2,None) #Descriptor also computed
#print(kp1[0].pt[0])

#orb for img1
kp1_o = orb.detect(gray1,None)
kp1_o, des1_o = orb.compute(gray1, kp1_o)

#orb for img2
kp2_o = orb.detect(gray2,None)
kp2_o, des2_o = orb.compute(gray2, kp2_o)

#FLANN - SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

#SIFT
matches = flann.knnMatch(des1,des2,k=2)
#matches_filterd = []
pts1 = []
pts2 = []

#Try to only use good matches
matchesMask = [[0,0] for i in range(len(matches))]

#Ratio test (SIFT)
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i]=[1,0]
        pts1.append([kp1[n.queryIdx].pt[0], kp1[n.queryIdx].pt[1]])
        pts2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
        good.append([m])


"""
with open('matches_sift.txt', 'a') as f:
    for match in matches_filterd:
        for coordinate in match:
            f.write("%s " % coordinate) #write matcheseach number
        f.write("\n") #end with newline
"""

draw_params = dict(
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

img_sift = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,matches,None,**draw_params)

img_sift2 = cv.drawMatchesKnn(gray1,kp1,gray2,kp2,good, None, flags=2)
plt.figure(4)
plt.imshow(img_sift2)

plt.figure(1)
plt.imshow(img_sift)


# Remove points that are not fulfilling epipolar constraint
pts1 = np.int32(np.around(pts1))
pts2 = np.int32(np.around(pts2))

F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, ransacReprojThreshold=0.3, confidence=0.7) # 0.05, 0.99999

# Select inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

np.savetxt('data/matchesSIFT.txt', np.hstack((pts1,pts2)))


plt.figure(3)
plt.plot(pts2[:,0], pts2[:,1], 'bo')

#numbers in plot
for p in range(len(pts2)):
	plt.text(pts2[p][0], pts2[p][1], str(p+1), color="black", fontsize=10)

plt.imshow(gray2)

#DEBUGGING
plt.figure(2)
plt.plot(pts1[:,0], pts1[:,1], 'ro')

#numbers in plot
for p in range(len(pts1)):
	plt.text(pts1[p][0], pts1[p][1], str(p+1), color="black", fontsize=10)

plt.imshow(gray1)

print("MATCHED POINT TWO")
print(pts1[1])
print(pts2[1])

plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all')

"""
#FLANN - ORB
FLANN_INDEX_LSH = 6
index_params_o= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2search_params = dict(checks=50)   # or pass empty dictionary

flann_o = cv.FlannBasedMatcher(index_params_o,search_params)

matches_o = flann_o.knnMatch(des1_o, des2_o, k=2)

matchesMask_o = [[0,0] for i in range(len(matches_o))]

#Ratio test (ORB)
for i,(m,n) in enumerate(matches_o):
    if m.distance < 0.72*n.distance:
        matchesMask_o[i]=[1,0]

draw_params_o = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask_o,
                   flags = cv.DrawMatchesFlags_DEFAULT)

img_orb = cv.drawMatchesKnn(gray1,kp1_o,gray2,kp2_o,matches_o,None,**draw_params_o)
"""

#plt.subplot(212)
#plt.imshow(img_orb)
#plt.show()

#Display images

"""
cv.imshow('fast',fast_img)
cv.imshow('sift',sift_img)
cv.imshow('original',gray1)
cv.waitKey(0)
cv.destroyAllWindows()
"""


#https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html
