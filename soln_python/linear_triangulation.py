import numpy as np

def linear_triangulation(uv1, uv2, P1, P2):
    """
    Compute the 3D position of a single point from 2D correspondences.

    Args:
        uv1:    2D projection of point in image 1.
        uv2:    2D projection of point in image 2.
        P1:     Projection matrix with shape 3 x 4 for image 1.
        P2:     Projection matrix with shape 3 x 4 for image 2.

    Returns:
        X:      3D coordinates of point in the camera frame of image 1.
                (not homogeneous!)

    See HZ Ch. 12.2: Linear triangulation methods (p312)
    """
    A = np.empty((4,4))
    A[0,:] = uv1[0]*P1[2,:] - P1[0,:]
    A[1,:] = uv1[1]*P1[2,:] - P1[1,:]
    A[2,:] = uv2[0]*P2[2,:] - P2[0,:]
    A[3,:] = uv2[1]*P2[2,:] - P2[1,:]
    U,s,VT = np.linalg.svd(A)
    X = VT[3,:]
    return X[:3]/X[3]
