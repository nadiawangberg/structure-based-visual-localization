import matplotlib.pyplot as plt
import numpy as np
from util import *
from eight_point import *

matches = np.loadtxt('../data/matches.txt')
uv1 = matches[:,:2]
uv2 = matches[:,2:]

I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')
K1 = np.loadtxt('../data/K1.txt')
K2 = np.loadtxt('../data/K2.txt')

F = eight_point(uv1, uv2)

# np.random.seed(1) # Uncomment if you don't want randomized points

show_point_matches(I1, I2, uv1, uv2, F)
plt.savefig('out2.png')
