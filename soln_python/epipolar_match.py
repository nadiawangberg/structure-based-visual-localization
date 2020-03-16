import numpy as np

def epipolar_match(I1, I2, F, uv1):
    """
    For each point in uv1, finds the matching point in image 2 by
    an epipolar line search.

    Args:
        I1:  (H x W matrix) Grayscale image 1
        I2:  (H x W matrix) Grayscale image 2
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
        uv1: (n x 2 array) Points in image 1

    Returns:
        uv2: (n x 2 array) Best matching points in image 2.
    """

    # Tips:
    # - Image indices must always be integer.
    # - Use int(x) to convert x to an integer.
    # - Use rgb2gray to convert images to grayscale.
    # - Skip points that would result in an invalid access.
    # - Use I[v-w : v+w+1, u-w : u+w+1] to extract a window of half-width w around (v,u).
    # - Use the np.sum function.

    w = 10
    uv2 = np.zeros(uv1.shape)
    for i, (u1,v1) in enumerate(uv1):
        if u1 < w or v1 < w or u1 > I1.shape[1]-w or v1 > I1.shape[0]-w:
            continue
        l = F@np.array((u1,v1,1))
        W1 = I1[int(v1)-w:int(v1)+w+1, int(u1)-w:int(u1)+w+1]

        best_err = np.inf
        best_u2 = w
        for u2 in range(w, I2.shape[1]-w):
            v2 = int(round(-(l[2] + u2*l[0])/l[1]))
            if v2 < w or v2 > I2.shape[0]-w:
                continue
            W2 = I2[v2-w:v2+w+1, u2-w:u2+w+1]
            err = np.sum(np.absolute(W1 - W2))
            if err < best_err:
                best_err = err
                best_u2 = u2

        uv2[i,0] = best_u2
        uv2[i,1] = -(l[2] + best_u2*l[0])/l[1]
    return uv2
