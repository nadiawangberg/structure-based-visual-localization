import numpy as np
from normalize_points import *

def eight_point(uv1, uv2):
    """ Given n >= 8 point matches, (u1 v1) <-> (u2 v2), compute the
    fundamental matrix F that satisfies the equations

        (u2 v2 1)^T * F * (u1 v1 1) = 0

    Args:
        uv1: (n x 2 array) Pixel coordinates in image 1.
        uv2: (n x 2 array) Pixel coordinates in image 2.

    Returns:
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1
             to lines in image 2.

    See HZ Ch. 11.2: The normalized 8-point algorithm (p.281).
    """

    uv1_n, T1 = normalize_points(uv1)
    uv2_n, T2 = normalize_points(uv2)

    # Build A
    n = len(uv1)
    A = np.empty((n, 9))
    for i in range(n):
        u1,v1 = uv1_n[i]
        u2,v2 = uv2_n[i]
        A[i,:] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1]
    A = np.array(A)

    # Solve for f and reshape
    U, s, VT = np.linalg.svd(A)
    F = np.reshape(VT[-1,:], (3,3))

    F = closest_fundamental_matrix(F)

    # Denormalize
    F = T2.T@F@T1

    return F

def closest_fundamental_matrix(F):
    """
    Computes the closest fundamental matrix in the sense of the
    Frobenius norm. See HZ, Ch. 11.1.1 (p.280).
    """
    U, s, VT = np.linalg.svd(F)
    return U@np.diag([s[0],s[1],0])@VT
