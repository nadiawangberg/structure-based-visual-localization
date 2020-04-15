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

def homogiphy(X):
    X_hom=np.zeros((len(X),4))
    for i in range(len(X)):
        x_hom=np.append(X[i],1)
        X_hom[i]=x_hom    
    return X_hom
def dehomogiphy(X_hom):
    X=np.zeros((len(X),3))
    for i in range(len(X)):
        x=X_hom[i,:3]
        X[i]=x    
    return X


def findT(match,pic1,pic2,K,T_n_w): #find transform
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
    X_n = np.array([linear_triangulation(uv1[i], uv2[i], P1, P2) \
        for i in range(n)])

    X_n_hom=homogiphy(X_n)
    X_w_hom = np.copy(X_n_hom)
    
    for i in range(len(X_n_hom)):
        X_w_hom[i]=T_n_w@X_n_hom[i]

    #T=np.eye(4)
    X_w=dehomogiphy(X_w_hom)
    X_w=np.array(X_w,dtype='float32')

    #uv2=np.array(uv2,dtype='float32')

    uv2=np.array(uv2,dtype='float32')
    dist_coeffs = np.zeros((4,1))
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(X, uv2, K2, dist_coeffs,flags=0)
    (success,rotation_vector, translation_vector,_) = cv2.solvePnPRansac(X_w, uv2, K, dist_coeffs,cv2.SOLVEPNP_UPNP)
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


    T_1_to_2 = np.vstack( (np.column_stack((R,t)), np.array([0,0,0,1])) )
    print(T_1_to_2)

    return T_v@T_r , X_w, T_1_to_2


#K1 = np.loadtxt('../data/K_p20.txt')
K1=np.loadtxt('../data/K1.txt')
K2 = K1
T0 = np.eye(4)
T1=np.eye(4) #T:N_to_world
plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
scale = 3
draw_frame(T1,scale,ax)
T1=np.eye(4)
"""
[T2,X, T]=findT('../data/matchesSIFT-1.txt','../data/im1.png','../data/im2.png',K1)
colors = ['c', 'r', 'g', 'b', 'y', 'm', 'k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

show_point_cloud(X,ax,
        xlim=[-1.6,+0.6],
        zlim=[-1.6,+0.6],
        ylim=[+2.0,+4.2], color = colors[0])    
T1=T1@T
draw_frame(T1,scale,ax)
"""
T_n_w=np.eye(4)
for i in range(1,8):
    [T_pnp,X_w, T_n_npp]=findT('../data/matchesSIFT'+str(i)+'.txt','../data/'+str(i)+'.jpg','../data/'+str(i+1)+'.jpg',K1,T_n_w)
    T_n_w = T_n_w@inv(T_n_npp)
    #T2 : T_N_to_N+1
    #T2=findT(X,uv2,K1)

    #draw_frame(T,scale,ax)
    """
    # X_t=np.zeros((len(X),4))
    # for j in range(len(X)):
    #     x_hom=np.append(X[j],1)
    #     X_t[j]=T@x_hom
    """
    print(X_t[0])
    print(X[0])
    """
    X_glob = []
    for point in X:
        point_glob = T2@T1@np.column_stack((point,1))
        X_glob.append(point_glob)
    """

    #draw_frame(T1@T0,1,ax)
    colors = ['c', 'r', 'g', 'b', 'y', 'm', 'k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    show_point_cloud(X_w,ax,
        xlim=[-1.6,+0.6],
        zlim=[-1.6,+0.6],
        ylim=[+2.0,+4.2], color = colors[i-1])
        
    T_pnp=T_pnp@T_n_w
    draw_frame(T_pnp,scale,ax)
"""
#print(X[0])
plt.show()