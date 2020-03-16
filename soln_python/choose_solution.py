import numpy as np
from linear_triangulation import *
from camera_matrices import *

def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    Chooses among the rotation and translation solutions Rts
    the one which gives the most points in front of both cameras.
    """
    n = len(uv1)
    best = (0, 0)
    for i, (R,t) in enumerate(Rts):
        P1,P2 = camera_matrices(K1, K2, R, t)
        X1 = np.array([linear_triangulation(uv1[j], uv2[j], P1, P2) for j in range(n)])
        X2 = X1 @ R.T + t
        visible = np.logical_and(X1[:,2] > 0, X2[:,2] > 0)
        num_visible = np.sum(visible)
        if num_visible > best[1]:
            best = (i, num_visible)
    print('Choosing solution %d (%d points visible)' % (best[0], best[1]))
    return Rts[best[0]]
