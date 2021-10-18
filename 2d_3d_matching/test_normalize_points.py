import numpy as np

def test_normalize_points(pts):
    mean = np.mean(pts, axis=0)
    dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    if np.absolute(mean[0]) > 0.01 or np.absolute(mean[1]) > 0.01:
        print('Translation is NOT GOOD, centroid was (%g, %g), should be (0,0).' % (mean[0], mean[1]))
        return
    else:
        print('Translation is GOOD.')
    if np.absolute(dist - np.sqrt(2)) > 0.1:
        print('Scaling is NOT GOOD, mean distance was %g, should be sqrt(2).' % dist)
        return
    else:
        print('Scaling is GOOD.')
