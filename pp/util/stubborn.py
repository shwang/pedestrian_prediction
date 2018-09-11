from __future__ import division
import numpy as np

# epsilon-stubborn transition:
#                / 1-epsilon        if j = i
# P(s'=j|s=i) = |
#                \ epsilon / (n-1)  if j =/= i
def epsilon_stubborn_transition(n, epsilon):
    assert n > 0
    if n == 1:
        return np.array([[1]])
    P_same = (1 - epsilon) / n
    P_other = epsilon / (n - 1)

    transition = np.empty([n, n])
    transition.fill(P_other)
    for i in range(n):
        transition[i, i] = P_same
    # assert np.sum(transition) - 1 < 1e-5, np.sum(transition)
    return transition