import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from util import *
from eight_point import *
from epipolar_match import *
from choose_solution import *
from linear_triangulation import *
from motion_from_essential import *
from essential_from_fundamental import *
from camera_matrices import *

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

X=np.vstack((X,[0,0,0]))
show_point_cloud(X,0,0,1,
    xlim=[-0.6,+1],
    ylim=[-0.6,+1],
    zlim=[-0.6,+5])
