import numpy as np

def motion_from_essential(E):
    """ Computes the four possible decompositions of E into
    a relative rotation and translation.

    See HZ Ch. 9.7 (p259): Result 9.19
    """
    U, s, VT = np.linalg.svd(E)

    # Make sure we return rotation matrices with det(R) == 1
    if np.linalg.det(U) < 0: U = -U
    if np.linalg.det(VT) < 0: VT = -VT

    W = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U@W@VT
    R2 = U@W.T@VT
    t1 = U[:,2]
    t2 = -U[:,2]
    return [(R1,t1), (R1,t2), (R2, t1), (R2, t2)]
