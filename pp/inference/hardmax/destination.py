from __future__ import division

import numpy as np

from ...parameters import val_default
from .beta import binary_search

def _mle_betas(g, traj, dests, beta_guesses, mk_bin_search=None, **kwargs):
    _binary_search = mk_bin_search or binary_search

    num = len(dests)
    betas = np.empty(num)
    if beta_guesses is None:
        beta_guesses = [1] * num

    if len(traj) == 0:
        return np.array(beta_guesses)

    for i, dest in enumerate(dests):
        guess = beta_guesses[i]
        betas[i] = _binary_search(g, traj, dest, guess=guess, **kwargs)

    return betas

def infer(g, traj, dests, beta_guesses=None, val_mod=val_default,
        mk_bin_search=None, mk_traj_prob=None, bin_search_opts={}):
    """
    For each destination, computes the beta_hat that maximizes
    the probability of the trajectory given the destination.

    Then normalizes over trajectory probabilities to find
    the probability of each destination.

    Params:
        - g [GridWorldMDP]: The MDP.
        - traj [list of (int, int)]: Trajectory represented by state-action
            pairs.
        - dests [listlike of int]: The possible destination
            states. Behavior is undefined if there are duplicate
            or invalid destination states.
        - beta_guesses [listlike of float] (optional): For binary search.

        - mk_{bin_search,traj_prob} (optional): Mock functions for
            unit test purposes.

    Returns:
        - dest_probs [np.ndarray]: The probability of each destination.
        - betas [np.ndarray]: The beta_hat associated with each destination.
    """
    assert len(traj) > 0
    _trajectory_probability = mk_traj_prob or val_mod.trajectory_probability
    betas = _mle_betas(g, traj, dests, beta_guesses, mk_bin_search,
            **bin_search_opts)

    dest_probs = np.zeros(len(dests))
    for i, dest in enumerate(dests):
        dest_probs[i] = _trajectory_probability(g, dest, traj, beta=betas[i])

    # XXX: fix this ugly hack for guaranteeing dest_prob[0] = 1
    # even when raw traj_prob = 0.
    if len(dests) == 1:
        dest_probs[0] = 1
    np.divide(dest_probs, sum(dest_probs), out=dest_probs)
    return dest_probs, betas

def hmm_infer(g, traj, dests, epsilon=0.05, beta_guesses=None,
        dest_prob_priors=None,
        val_mod=val_default, verbose_return=False,
        mk_bin_search=None, mk_act_probs=None, bin_search_opts={}):
    """
    For each destination, computes the beta_hat that maximizes
    the probability of the trajectory given the destination.

    Then runs the forward algorithm to calculate the probability
    of each destination at each timestep, given an epsilon-stubborn
    agent and the MLE betas associated with each destination.

    Params:
        - g [GridWorldMDP]: The MDP.
        - traj [list of (int, int)]: Trajectory represented by
            state-action pairs.
        - dests [listlike of int]: The possible destination
            states. Behavior is undefined if there are duplicate
            or invalid destination states.
        - beta_guesses [listlike of float] (optional): For binary search.
        - verbose_output [bool] (optional): If True, then return the probability
            of each destination at every timestep, rather than just at the last
            timestep.

        - mk_{bin_search,traj_prob} (optional): Mock functions for
            unit testing purposes.

    Returns:
        - dest_probs [np.ndarray]: The probability of each
            destination at the last timestep. If verbose_output is True, then
            this outputs destination probabilities at every timestep as a 2D
            array where the first index is time and the second index is
            the destination.
        - betas [np.ndarray]: The beta_hat associated with
            each destination.
    """
    action_probabilities = mk_act_probs or val_mod.action_probabilities
    L = len(dests)
    assert L > 0

    # Unpack dest_prob.
    if dest_prob_priors is None:
        dest_prob_priors = [1] * L
    assert len(dest_prob_priors) == L
    dest_prob_priors = np.array(dest_prob_priors) / sum(dest_prob_priors)

    # Cache MLE beta for entire trajectory given a particular destination.
    betas = _mle_betas(g, traj, dests, beta_guesses, mk_bin_search,
        **bin_search_opts)

    # Cache action probabilities for each destination and corresponding betas.
    P_a = np.empty([L, g.S, g.A])
    for i, (dest, beta) in enumerate(zip(dests, betas)):
        P_a[i] = action_probabilities(g, dest, beta=beta)

    # Cache 'epsilon-stubborn' transition probabilities
    if L == 1:
        T = np.array([[1]])
    else:
        T = np.empty([L, L])
        for d in range(L):
            for d_prime in range(L):
                if d == d_prime:
                    T[d, d_prime] = 1 - epsilon
                else:
                    T[d, d_prime] = epsilon/(L - 1)

    # Execute forward algorithm.
    if verbose_return:
        P_D_all = np.empty([len(traj), L])
    P_D = np.copy(dest_prob_priors)
    P_D_prime = np.empty(L)

    for t, (s, a) in enumerate(traj):
        if t == 0:
            for d_now in range(L):
                P_D_prime[d_now] = P_a[d_now, s, a] * P_D[d_now]
        else:
            for d_now in range(L):
                for d_prev in range(L):
                    P_D_prime[d_now] += (P_D[d_prev] * T[d_prev, d_now] *
                            P_a[d_now, s, a])

        # Normalize probabilities
        P_D_prime /= sum(P_D_prime)
        if verbose_return:
            P_D_all[t] = P_D
        # Reuse P_D as buffer for next P_D_prime.
        P_D, P_D_prime = P_D_prime, P_D
        P_D_prime.fill(0)

    if verbose_return:
        return P_D_all, betas
    else:
        return P_D, betas
