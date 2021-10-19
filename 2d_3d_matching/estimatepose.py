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


def findT(match,pic1,pic2,K):
    matches = np.loadtxt(match)
    uv1 = matches[:,:2]
    uv2 = matches[:,2:]
    n = len(matches)

    I1 = plt.imread(pic1)
    I2 = plt.imread(pic2)
    colors=np.zeros((len(uv1),3))
    print(uv1)
    uv1_int=np.array(uv1,dtype=int)
    print(len(I1))
    print(len(I1[0]))
    for i in range(len(uv1_int)):
        colors[i]=I1[uv1_int[i,1],uv1_int[i,0]]/255.0


    F = eight_point(uv1, uv2)

    E = essential_from_fundamental(F, K, K)
    Rts = motion_from_essential(E)
    R,t = choose_solution(uv1, uv2, K, K, Rts)
    P1,P2 = camera_matrices(K, K, R, t)


    x=t[0]
    y=t[1]
    z=t[2]

    trans_1_2=np.array([[1,0,0,-x],
                        [0,1,0,y],
                        [0,0,1,-z],
                        [0,0,0,1]])

    #Rotation in an array
    rot_1_2=np.array([[R[0,0],R[0,1],R[0,2],0],
                    [R[1,0],R[1,1],R[1,2],0],
                    [R[2,0],R[2,1],R[2,2],0],
                    [0,0,0,1]])
    
    # Uncomment for task 4b
    # uv1 = np.loadtxt('../data/goodPoints.txt')
    # uv2 = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1)

    n = len(uv1)
    X = np.array([linear_triangulation(uv1[i], uv2[i], P1, P2) \
        for i in range(n)])

    X=np.array(X,dtype='float32')
    uv2=np.array(uv2,dtype='float32')

    uv=np.array(uv2,dtype='float32')
    dist_coeffs = np.zeros((4,1))

    (success,rotation_vector, translation_vector,_) = cv2.solvePnPRansac(X, uv, K, dist_coeffs,cv2.SOLVEPNP_UPNP)
    temp_rotation_vector = np.array([rotation_vector[0], rotation_vector[2], rotation_vector[1]]) #change y,z
    T_r,_=cv2.Rodrigues(temp_rotation_vector) #rotation
    print("Success",success)
    
    #Translation
    x=translation_vector[0]
    y=translation_vector[1]
    z=translation_vector[2]
    T_v=np.array([[1,0,0,-x],[0,1,0,y],[0,0,1,-z],[0,0,0,1]])

    #Rotation in an array
    T_r=np.array([[T_r[0,0],T_r[0,1],T_r[0,2],0],[T_r[1,0],T_r[1,1],T_r[1,2],0],[T_r[2,0],T_r[2,1],T_r[2,2],0],[0,0,0,1]])
    
    T_v=T_v.astype(float)
    T_r=T_r.astype(float)

    return T_v@T_r,X,trans_1_2@rot_1_2,colors


#K1 = np.loadtxt('../data/K_p20.txt')
#K2 = K1
K1 = np.loadtxt('../data/K_p20.txt')
K2 = np.loadtxt('../data/K_p20.txt')
T0 = np.eye(4)
T1=np.eye(4)
plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
draw_frame(T1,1,ax,0)
[T2,X,T_1_2,colors]=findT('../data/matchesSIFT'+str(1)+'.txt','../Photos/glosh'+str(1)+'.jpg','../Photos/glosh'+str(1+1)+'.jpg',K1)
show_point_cloud(X,T1,ax,1,colors,
        xlim=[-1.6,+0.6],
        zlim=[-1.6,+0.6],
        ylim=[+2.0,+4.2])
T_c1_cn=np.eye(4)
for i in range(1,2):
    [T2,X,T_1_2,colors]=findT('../data/matchesSIFT'+str(i)+'.txt','../Photos/glosh'+str(i)+'.jpg','../Photos/glosh'+str(i+1)+'.jpg',K1)
    T_c1_cn=T2@T_c1_cn 
    draw_frame(T_c1_cn,1,ax,i)

plt.show()