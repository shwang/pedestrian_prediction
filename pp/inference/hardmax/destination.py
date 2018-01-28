from __future__ import division

import numpy as np

from ...mdp import gridless as gridless
from ...parameters import val_default
from .beta import binary_search
from ...util.dest import destination_transition
from sklearn.preprocessing import normalize

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
    _trajectory_probability = mk_traj_prob or g.trajectory_probability
    betas = _mle_betas(g, traj, dests, beta_guesses, mk_bin_search,
            **bin_search_opts)

    dest_probs = np.zeros(len(dests))
    for i, dest in enumerate(dests):
        dest_probs[i] = _trajectory_probability(dest, traj, beta=betas[i])

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
    action_probabilities = mk_act_probs or g.action_probabilities
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
        P_a[i] = action_probabilities(dest, beta=beta)

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

def infer_joint(g, dests, betas, priors=None, traj=[], epsilon=0.02,
        verbose_return=False, use_gridless=False):
    # Process parameters
    if not use_gridless:
        T = len(traj)
    else:
        T = max(len(traj) - 1, 0)
    n_D = len(dests)
    n_B = len(betas)
    if priors is None:
        priors = np.ones([n_D, n_B])
    else:
        priors = np.array(priors)

    assert priors.shape == (n_D, n_B)
    assert np.sum(priors) != 0
    assert np.all(priors >= 0)
    priors = priors / np.sum(priors)

    P_joint_DB_all = np.empty([T+1, n_D, n_B])
    P_joint_DB_all[0] = priors
    # Trivial case, don't compute action probabilities.
    if T == 0:
        if verbose_return == False:
            return P_joint_DB_all[-1]
        else:
            return P_joint_DB_all[-1], P_joint_DB_all

    # Precompute
    AXIS_D, AXIS_B, AXIS_S, AXIS_A = 0, 1, 2, 3
    boltzmann = np.empty([n_D, n_B, g.S, g.A])
    for index_d, d in enumerate(dests):
        for index_b, b in enumerate(betas):
            act_probs = boltzmann[index_d, index_b]
            act_probs[:] = g.action_probabilities(goal=d, beta=b)

    dest_trans = destination_transition(n_D, epsilon)

    # t=0 (priors)
    P_beta = np.sum(priors, axis=0)
    assert P_beta.shape == (n_B,)
    P_dest_given_beta = normalize(priors, axis=AXIS_D, norm='l1')

    def calc_joint():
        res =  np.multiply(P_dest_given_beta, P_beta.reshape(1, n_B))
        assert np.sum(res) - 1 < 1e-5, np.sum(res)
        return res

    if not use_gridless:
        emissions = traj
    else:
        emissions = []
        for i in range(len(traj) - 1):
            s, s_prime = np.array(traj[i]), np.array(traj[i+1])
            s = np.array(s, dtype=float)
            s_prime = np.array(s_prime, dtype=float)
            assert s.shape == (2,)
            assert s_prime.shape == (2,)
            emissions.append([s, s_prime])

    # Main loop: t >= 1
    for i, emission in enumerate(emissions):
        # Calculate emission probability given each (dest, beta) pair
        if use_gridless:
            s, s_prime = emission
            assert s.shape == (2,), s.shape
            assert s_prime.shape == (2,), s_prime.shape
            boltzmann = np.empty([n_D, n_B])
            for index_d, d in enumerate(dests):
                for index_b, b in enumerate(betas):
                    d_coor = g.state_to_real_coor(d)
                    boltzmann[index_d, index_b] = gridless.action_probability(
                            start=s, end=s_prime, dest=d_coor, beta=b,
                            W=g.rows, H=g.cols)
        else:
            s, a = emission
            boltzmann = np.empty([n_D, n_B])
            for index_d, d in enumerate(dests):
                for index_b, b in enumerate(betas):
                    boltzmann[index_d, index_b] = g.action_probabilities(
                            goal=d, beta=b)[s, a]

        t = i + 1
        if t == 1:
            P_dest_given_beta = np.multiply(P_dest_given_beta,
                    boltzmann, out=P_dest_given_beta)
        else:
            P_dest_given_beta_predict = np.matmul(dest_trans, P_dest_given_beta)
            normalize(P_dest_given_beta_predict, axis=AXIS_B, copy=False,
                    norm='l1')
            assert P_dest_given_beta_predict.shape == (n_D, n_B)

            P_dest_given_beta = np.multiply(P_dest_given_beta_predict,
                    boltzmann, out=P_dest_given_beta)

        # Use unnormalized P_dest_given_beta for P_beta.
        P_beta = np.multiply(P_beta, np.sum(P_dest_given_beta, axis=AXIS_D),
                out=P_beta)
        np.divide(P_beta, np.sum(P_beta), out=P_beta)
        assert P_beta.shape == (n_B,)

        # Now that P_beta is computed, we can normalize P_dest_given_beta.
        normalize(P_dest_given_beta, axis=AXIS_D, copy=False, norm='l1')
        assert P_dest_given_beta.shape == (n_D, n_B)

        P_joint_DB_all[t] = calc_joint()

    if verbose_return == False:
        return P_joint_DB_all[-1]
    else:
        return P_joint_DB_all[-1], P_joint_DB_all
