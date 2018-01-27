from __future__ import division
import numpy as np

def _value(g, s, verbose=False):
    """
    Estimate values as negative euclidean distance from s.
    Ignores rewards while calculating values! Rewards are only considered
    in `q_values` and `action_probabilities`.

    Params:
        mdp [GridWorldMDP]: The MDP.
        s [int]: The state from which euclidean distances are calculated.

    Returns:
        V [np.ndarray]: An `mdp.S`-length vector, where the ith entry is
            the reward of the optimal trajectory from the starting state to
            state i.
    """
    V = np.empty(g.S)
    R, C = g.state_to_coor(s)

    for s_prime in range(g.S):
        r, c = g.state_to_coor(s_prime)
        V[s_prime] = (R-r)**2 + (C-c)**2

    np.sqrt(V, out=V)
    np.multiply(V, -1, out=V)

    return V

forwards_value_iter = backwards_value_iter = _value
