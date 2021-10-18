import matplotlib.pyplot as plt
import numpy as np
from util import *
from eight_point import *
from epipolar_match import *

matches = np.loadtxt('../data/matches.txt')
uv1 = matches[:,:2]
uv2 = matches[:,2:]
n = len(matches)

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
K1 = np.loadtxt('../data/K1.txt')
K2 = np.loadtxt('../data/K2.txt')

F = eight_point(uv1, uv2)

# np.random.seed(1) # Uncomment if you don't want randomized points

# Choose k random points to visualize
n = len(uv1)
k = 10
sample = np.random.choice(range(n), size=k, replace=False)
uv1 = uv1[sample,:]
uv2 = uv2[sample,:]
uv2_match = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1)

# Draw points in image 1 and matching point in image 2 (true vs epipolar match)
plt.figure(figsize=(6,4))
colors = plt.cm.get_cmap('Set1', k).colors
plt.subplot(121)
plt.imshow(I1)
plt.scatter(uv1[:,0], uv1[:,1], s=100, marker='x', c=colors)
plt.subplot(122)
plt.imshow(I2)
plt.scatter(uv2_match[:,0], uv2_match[:,1], s=100, marker='x', c=colors, linewidths=2, label='Found match')
plt.scatter(uv2[:,0], uv2[:,1], s=100, marker='o', facecolor='none', edgecolors=colors, linewidths=2, label='True match')
plt.legend()
plt.tight_layout()
plt.savefig('out4.png')
