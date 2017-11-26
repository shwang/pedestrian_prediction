from __future__ import division

import numpy as np

from ...mdp.hardmax import trajectory_probability
from .beta import binary_search

def infer(g, traj, dest_list, beta_guesses=None,
        mk_bin_search=None, mk_traj_prob=None, **kwargs):
    """
    For each destination, computes the beta_hat that maximizes
    the probability of the trajectory given the destination.

    Then normalizes over trajectory probabilities to find
    the probability of each destination.

    Params:
        - g [GridWorldMDP]: The MDP.
        - traj [list of (int, int)]: Trajectory represented by
            state-action pairs.
        - dest_list [listlike of int]: The possible destination
            states. Behavior is undefined if there are duplicate
            or invalid destination states.
        - beta_guesses [listlike of float] (optional): For binary search.

        - mk_{bin_search,traj_prob} (optional): Mock functions for
            unit test purposes.

    Returns:
        - dest_probs [np.ndarray]: The probability of each
            destination.
        - betas [np.ndarray]: The beta_hat associated with
            each destination.
    """
    assert len(traj) > 0

    num = len(dest_list)
    betas = np.zeros(num)
    dest_probs = np.zeros(num)

    _binary_search = mk_bin_search or binary_search
    _trajectory_probability = mk_traj_prob or trajectory_probability

    if beta_guesses is None:
        beta_guesses = [1] * num

    for i, dest in enumerate(dest_list):
        guess = beta_guesses[i]
        betas[i] = _binary_search(g, traj, dest, guess=guess, **kwargs)
        dest_probs[i] = _trajectory_probability(g, dest, traj, betas[i])

    np.divide(dest_probs, sum(dest_probs), out=dest_probs)
    return dest_probs, betas
