from __future__ import division
import numpy as np

def destination_transition(n_D, epsilon):
    assert n_D > 0
    if n_D == 1:
        return np.array([[1]])
    P_same = (1 - epsilon) / n_D
    P_other = epsilon / (n_D - 1)

    transition = np.empty([n_D, n_D])
    transition.fill(P_other)
    for i in range(n_D):
        transition[i, i] = P_same
    # assert np.sum(transition) - 1 < 1e-5, np.sum(transition)
    return transition
