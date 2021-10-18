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

matches = np.loadtxt('../data/matchesSIFT.txt')
print(matches)
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


#print(X)
#print(len(X))

"""
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)): 
	plt.scatter(X[i][2], X[i][1], X[i][0], marker='o')
	print(X[i][2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#ax.set_zlim3d(-0.01,0.01)
"""

#plt.show()

#for i 
#plt.scatter(x,y,z, 

"""
show_point_cloud(X,
    xlim=[-0.6,+0.6],
    ylim=[-0.6,+0.6],
    zlim=[+3.0,+4.2])
"""

plt.figure(figsize=(6,6))
ax=plt.axes(projection='3d')


# for luddes x,y,z config
"""
show_point_cloud(X,np.eye(4),ax,1,
    xlim=[-0.6,+0.6],
    ylim=[-0.6,+0.6],
    zlim=[+3.0,+4.2])
"""

# for old skool x,y,z config
show_point_cloud(X,np.eye(4),ax,1,
    xlim=[-0.6,+0.6],
    zlim=[-0.6,+0.6],
    ylim=[+3.0,+4.2])
