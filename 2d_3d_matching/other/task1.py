import numpy as np
from normalize_points import *
from test_normalize_points import *

matches = np.loadtxt('../data/matches.txt')
pts = matches[:,:2]

print('Checking that the points satisfy the normalization criteria...')
pts_n,T = normalize_points(pts)
test_normalize_points(pts_n)

print('Checking that the transformation matrix performs the same operation...')
pts_n = T@np.column_stack((pts, np.ones(len(pts)))).T
pts_n = (pts_n[:2,:] / pts_n[2,:]).T
test_normalize_points(pts_n)
