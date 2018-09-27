from __future__ import division
import math
import numpy as np

rad = math.radians
def dist(a, b):
    return np.linalg.norm(a-b)

def action_probabilities(start, dest, R, W, H, granularity=15, beta=1):
    assert 360 % granularity == 0
    assert beta > 0
    std_thetas = np.arange(0, rad(360), rad(granularity))
    dists = circle_dists(center=start, dest=dest, R=R, W=W, H=H,
            granularity=granularity)
    dists /= -beta
    dists = np.exp(dists)
    dists /= np.sum(dists)
    return dists


# TODO: add something like `absorb_threshold` for determining whether R is
# small enough to be considered `ABSORB`
def action_probability(start, end, dest, W, H, granularity=15,
        verbose_return=False, **kwargs):
    R = dist(start, end)
    P = action_probabilities(start=start, dest=dest, R=R, W=W, H=H,
            granularity=granularity, **kwargs)
    theta = math.atan2((end[1] - start[1]), (end[0] - start[0]))
    if theta < 0:
        theta += rad(360)
    assert 0 <= theta < rad(360)
    theta_near_index = int(round(theta/rad(granularity)) % (len(P) - 1))

    if verbose_return:
        return P[theta_near_index], P
    else:
        return P[theta_near_index]


def circle(center, R=1, granularity=15, append_center=True):
    """
    Returns a 2D array, where the ith entry is the x,y coordinates of the ith
    point along the radius R circle centered at `center`.

    append_center: If true, include center as the final point in this array.
    """
    assert center.shape == (2,)

    std_thetas = np.arange(0, rad(360), rad(15))
    res = np.empty([len(std_thetas), 2])
    res[:, 0] = np.cos(std_thetas)
    res[:, 1] = np.sin(std_thetas)
    res *= R
    np.add(res, center, out=res)

    if append_center:
        return np.vstack([res, center])
    else:
        return res

def circle_dists(center, dest, W=None, H=None, **kwargs):
    """
    Returns the distances between `dest` points on a circle centered at
    `center`. If `W` and `H` are given, then points outside the rectangle
    defined by corners (0,0) and (W,H) are considered to have np.inf distance.
    """
    assert dest.shape == (2,)
    points = circle(center=center, **kwargs)
    vec = np.subtract(points, dest)
    res = np.linalg.norm(vec, axis=1)
    if W is not None and H is not None:
        for i, (x, y) in enumerate(points):
            if 0 <= x <= W and 0 <= y <= H:
                continue
            else:
                res[i] = np.inf
    return res
