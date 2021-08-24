# Structure-based visual localization

Python implementation of Structure-based visual localization. Features were detected and matched using SIFT. Triangulation was used to create a sparse 3D reconstruction, as is visualized below. 2D-3D matches are established from new query images. Finally, the camera pose is estimated using PnP inside a RANSAC loop. 

<img width="561" alt="3d_reconstruction" src="https://user-images.githubusercontent.com/29915643/129894652-b2557ff0-bdcc-46c8-8442-fc25e7097768.PNG">

## SIFT vs ORB features

A comparison between SIFT and ORB features were initially performed to choose which feature detector and descriptor to go with. Although ORB was significantly faster, too few matches were found in comparison to SIFT, which caused a very sparse 3D reconstruction. 

<img width="498" alt="ORB_SIFT" src="https://user-images.githubusercontent.com/29915643/129895082-88c786db-775d-4729-a52e-3ae7c94aa521.PNG">

## Localization

Below is an example showing a sequence of 2D images. The second image shows the 3d reconstruction and the pose of the camera relative to the book in each frame.
<img width="745" alt="sequence" src="https://user-images.githubusercontent.com/29915643/129897316-f91fb8a2-28a0-4f0e-8cbe-f7ff019b29ec.PNG">
<img width="621" alt="VO_book_sequence" src="https://user-images.githubusercontent.com/29915643/129897326-5e1aff1a-4a10-47b2-97cd-d7a7efcb3814.PNG">


## How to run

### Add dataset

Change the images and camera calibration files in the /data folder. Change im1.png, im2.png to consecutive images in your dataset. Change K1.txt, K2.txt to be the [camera calibration](https://docs.opencv.org/4.5.2/dc/dbb/tutorial_py_calibration.html) matrix of the camera that captured im1.png and im2.png. 

### Feature detection and matching
```
python3 feat_matcher.py
```
This script outputs the data/matches_sift.txt file needed for 3D reconstruction and pose estimation. 

### 3D reconstruction and pose estimation
```
cd soln_python
python3 estimatepose.py
```
