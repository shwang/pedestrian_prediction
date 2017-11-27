from __future__ import division
import numpy as np

from ..softact_shared import q_values as _q_values
from ..softact_shared import action_probabilities \
        as _action_probabilities

def _value(g, s, beta=1, verbose=False):
    """
    Estimate values as negative euclidean distance from s.
    Ignores rewards while calculating values! Rewards are only considered
    in `q_values` and `action_probabilities`.

    Params:
        mdp [GridWorldMDP]: The MDP.
        s [int]: The state from which euclidean distances are calculated.
        beta [float]: The irrationality constant. This value is ignored(!),
            and is left here for compatibility purposes.
            TODO: really necessary?

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

def q_values(g, goal_state):
    return _q_values(g, goal_state, forwards_value_iter)

def action_probabilities(g, goal_state, **kwargs):
    return _action_probabilities(g, goal_state, q_values, **kwargs)
