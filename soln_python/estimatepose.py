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


def findT(match,pic1,pic2,K): #find transform
    matches = np.loadtxt(match)
    uv1 = matches[:,:2]
    uv2 = matches[:,2:]
    n = len(matches)

    I1 = plt.imread(pic1)
    I2 = plt.imread(pic2)

    F = eight_point(uv1, uv2)

    E = essential_from_fundamental(F, K, K)
    Rts = motion_from_essential(E)
    R,t = choose_solution(uv1, uv2, K, K, Rts)
    P1,P2 = camera_matrices(K, K, R, t)

    # Uncomment for task 4b
    # uv1 = np.loadtxt('../data/goodPoints.txt')
    # uv2 = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1)

    n = len(uv1)
    X = np.array([linear_triangulation(uv1[i], uv2[i], P1, P2) \
        for i in range(n)])


    #T=np.eye(4)
    X=np.array(X,dtype='float32')
    uv2=np.array(uv2,dtype='float32')

    uv=np.array(uv2,dtype='float32')
    dist_coeffs = np.zeros((4,1))
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(X, uv2, K2, dist_coeffs,flags=0)
    (success,rotation_vector, translation_vector,_) = cv2.solvePnPRansac(X, uv, K, dist_coeffs,cv2.SOLVEPNP_UPNP)
    temp_rotation_vector = np.array([rotation_vector[0], rotation_vector[2], rotation_vector[1]]) #change y,z
    T_r,_=cv2.Rodrigues(temp_rotation_vector) #rotation

    #Translation
    x=translation_vector[0]
    y=translation_vector[1]
    z=translation_vector[2]

    #Translation in an array
    T_v=np.array([[1,0,0,-x],[0,1,0,y],[0,0,1,-z],[0,0,0,1]])

    #Rotation in an array
    T_r=np.array([[T_r[0,0],T_r[0,1],T_r[0,2],0],[T_r[1,0],T_r[1,1],T_r[1,2],0],[T_r[2,0],T_r[2,1],T_r[2,2],0],[0,0,0,1]])
    
    T_v=T_v.astype(float)
    T_r=T_r.astype(float)

    return T_v@T_r,X #Entire transformation


K1 = np.loadtxt('../data/K_p20.txt')
K2 = K1
T0 = np.eye(4)
T1=np.eye(4)
plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
draw_frame(T1,1,ax)
for i in range(1,4):
    [T2,X]=findT('../data/matchesSIFT'+str(i)+'.txt','../data/'+str(i)+'.jpg','../data/'+str(i+1)+'.jpg',K1)
    #T2=findT(X,uv2,K1)

    #draw_frame(T,scale,ax)

    T_inv=inv(T1)
    X_t=np.zeros((len(X),4))
    for j in range(len(X)):
        x_hom=np.append(X[j],1)
        X_t[j]=T_inv@x_hom.T
    X=X_t

    """
    X_glob = []
    for point in X:
        point_glob = T2@T1@np.column_stack((point,1))
        X_glob.append(point_glob)
    """

    #draw_frame(T1@T0,1,ax)
    draw_frame(T1@T2,1,ax)

    print("HEEEEEEEEEEEEEEEEEY")
    print("ey", i)
    colors = ['r', 'g', 'b']
    show_point_cloud(X,T1,ax,1,
        xlim=[-1.6,+0.6],
        zlim=[-1.6,+0.6],
        ylim=[+2.0,+4.2], color = colors[i-1])
        
    T1=T1@T2


print(X[0])
plt.show()