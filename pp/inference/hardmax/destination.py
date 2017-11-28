from __future__ import division

import numpy as np

from ...parameters import val_default
from .beta import binary_search

def infer(g, traj, dests, beta_guesses=None, val_mod=val_default,
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
        - dests [listlike of int]: The possible destination
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

    num = len(dests)
    betas = np.zeros(num)
    dest_probs = np.zeros(num)

    _binary_search = mk_bin_search or binary_search
    _trajectory_probability = mk_traj_prob or val_mod.trajectory_probability

    if beta_guesses is None:
        beta_guesses = [1] * num

    for i, dest in enumerate(dests):
        guess = beta_guesses[i]
        betas[i] = _binary_search(g, traj, dest, guess=guess, **kwargs)
        dest_probs[i] = _trajectory_probability(g, dest, traj,
                beta=betas[i])

    # XXX: fix this ugly hack for guaranteeing dest_prob[0] = 1
    # even when raw traj_prob = 0.
    if num == 1:
        dest_probs[0] = 1
    np.divide(dest_probs, sum(dest_probs), out=dest_probs)
    return dest_probs, betas
