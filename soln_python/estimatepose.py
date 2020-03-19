import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import cv2
from util import *
from eight_point import *
from epipolar_match import *
from choose_solution import *
from linear_triangulation import *
from motion_from_essential import *
from essential_from_fundamental import *
from camera_matrices import *
from numpy.linalg import inv
matches = np.loadtxt('../data/matches.txt')
uv1 = matches[:,:2]
uv2 = matches[:,2:]
n = len(matches)

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
K1 = np.loadtxt('../data/K1.txt')
K2 = np.loadtxt('../data/K2.txt')

F = eight_point(uv1, uv2)

E = essential_from_fundamental(F, K1, K2)
Rts = motion_from_essential(E)
R,t = choose_solution(uv1, uv2, K1, K2, Rts)
P1,P2 = camera_matrices(K1, K2, R, t)

# Uncomment for task 4b
# uv1 = np.loadtxt('../data/goodPoints.txt')
# uv2 = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1)

n = len(uv1)
X = np.array([linear_triangulation(uv1[i], uv2[i], P1, P2) \
    for i in range(n)])


#T=np.eye(4)
X=np.array(X,dtype='float32')
uv2=np.array(uv2,dtype='float32')


def findT(X,uv,K):
    uv=np.array(uv,dtype='float32')
    dist_coeffs = np.zeros((4,1))
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(X, uv2, K2, dist_coeffs,flags=0)
    (success,rotation_vector, translation_vector,_) = cv2.solvePnPRansac(X, uv, K2, dist_coeffs,cv2.SOLVEPNP_UPNP)
    T_r,_=cv2.Rodrigues(rotation_vector)

    x=translation_vector[0]
    y=translation_vector[1]
    z=translation_vector[2]

    T_v=np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])

    T_r=np.array([[T_r[0,0],T_r[0,1],T_r[0,2],0],[T_r[1,0],T_r[1,1],T_r[1,2],0],[T_r[2,0],T_r[2,1],T_r[2,2],0],[0,0,0,1]])
    T_v=T_v.astype(float)
    T_r=T_r.astype(float)

    return T_v@T_r

T1=findT(X,uv1,K1)
T2=findT(X,uv2,K1)

plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
draw_frame(T1,1,ax)

draw_frame(T2@T1,1,ax)

show_point_cloud(X,T1,ax,1,
    xlim=[-0.6,+1], 
    ylim=[-0.6,+1],
    zlim=[-0.6,+5])

