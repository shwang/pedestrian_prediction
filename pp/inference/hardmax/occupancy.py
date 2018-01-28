from __future__ import division

import numpy as np
import destination
import state

from ...parameters import val_default
from ...util.args import unpack_opt_list

def infer_simple(g, init_state, dest, T, beta=1, action_prob=None,
        val_mod=val_default):
    """
    Calculate expected occupancy for a `beta`-rational agent over trajectories
    of length T, given a fixed destination `dest`.

    Return the expected number of times that the agent will enter each state
    as an 1D array of dimension (`g.S`).
    """
    P = state.infer_simple(g, init_state, dest, T, beta, action_prob, val_mod)
    D = np.sum(P[1:], axis=0)
    D[dest] = 1  # This value is fixed.
    return D


def infer_bayes(g, dest, T, betas, traj=[], init_state=None, priors=None,
        action_prob=None, val_mod=val_default, verbose_return=False):
    """ Single dest, multiple betas Bayesian inference."""
    assert betas is not None
    occ_res, occ_all, P_beta = state.infer_bayes(g, dest, T, betas, traj=traj,
            init_state=init_state,
            priors=priors, action_prob=action_prob, verbose_return=True)
    D = np.sum(occ_res[1:], axis=0)
    D[dest] = 1  # This value is fixed.

    if verbose_return:
        return D, occ_res, occ_all, P_beta
    else:
        return D

def infer_joint(*args, **kwargs):
    """
    Multi dest, multi beta Bayesian inference.
    Params: The same as state.infer_state.

    Returns:
        occ_res [np.ndarray]: A (T+1 x S) array, where the `t`th entry is the
            probability of state S in `t` timesteps from now.
        occ_all [np.ndarray]: A (|betas| x T+1 x S) array, where the `b`th entry
            is the (T+1 x S) expected states probabilities if it were the case
            that `beta_star == beta[b]`.
        P_joint_DB [np.ndarray]: A (|dests| x |betas|) dimension array, where
            the `b`th entry is the posterior probability associated with
            `betas[b]`.
    """
    occ_res, occ_all, P_joint_DB = state.infer_joint(*args, verbose_return=True,
            **kwargs)

    D = np.sum(occ_res[1:], axis=0)
    D[dest] = 1  # This value is fixed.

    return D, occ_res, occ_all, P_beta


def infer_from_start(g, init_state, dest_or_dests, dest_probs=None,
        T=None, verbose=False, beta_or_betas=1, cached_action_probs=None,
        verbose_return=False):
    """
    Use prior to calculate the expected number of times that the agent will
    enter each state.

    Params:
        g [GridWorldMDP] -- The MDP in which the agent resides.
        init_state [int] -- The agent's current state.
        dest_or_dests [int] -- The agent's goal, or a list of potential goals.
        T [int] -- Make an inference for state probabilities at this many
            timesteps in the future.
        dest_probs [np.ndarray] (optional) -- The posterior probability of each
            destination. By default, uniform over each possible destination.
        beta_or_betas [float] (optional) -- The irrationality coefficient, or
            a list of irrationality coefficients coresponding to each of the
            agent's potential goals.
        cached_action_probs [np.ndarray] (optional) -- Cached results from
            `MDP.action_probabilities()`.
        verbose_return [bool] (optional) -- If True, return extra information.

    Returns:
    If verbose_return is True, then returns D, D_dests, dest_probs, betas
    If verbose_return is False, then returns D

    D, a 2D array, is a weighted sum of the occupancies in D_dests. The weight
        for each occupancy grid is equal to the posterior probability of the
        associated destination.
    D_dests is a list of 2D arrays. The ith array is the expected occupancy
        given that the true destination is the ith destination.
    dest_probs, a 1D array, is the posterior probability of each destination.
    betas is a 1D array containing the MLE beta for each destination.
    """
    if T is None:
        T = g.rows + g.cols

    dests = unpack_opt_list(dest_or_dests)
    L = len(dests)

    betas = np.array(unpack_opt_list(beta_or_betas, extend_to=L))
    assert len(betas) == L, betas

    # Unpack cached_action_probs, if applicable.
    if cached_action_probs is not None:
        act_probs = unpack_opt_list(cached_action_probs)
    else:
        act_probs = [None] * L
    assert len(act_probs) == L, act_probs

    # Unpack dest_prob
    if dest_probs is None:
        dest_probs = [1] * L
    assert len(dest_probs) == L
    dest_probs = np.array(dest_probs) / sum(dest_probs)

    # Take the weighted sum of the occupancy given each individual destination.
    D = np.zeros(g.S)
    D_dests = []  # Only for verbose_return
    for dest, beta, act_prob, dest_prob in zip(
            dests, betas, act_probs, dest_probs):
        D_dest = infer_simple(g, init_state, dest, T, beta=beta,
                action_prob=act_prob)
        if verbose_return:
            D_dests.append(np.copy(D_dest))
        np.multiply(D_dest, dest_prob, out=D_dest)
        np.add(D, D_dest, out=D)

    if not verbose_return:
        return D
    else:
        return D, D_dests, dest_probs, betas


def infer(g, traj, dest_or_dests, T=None, verbose=False, beta_or_betas=None,
        hmm=False, hmm_opts={},
        beta_guesses=None, bin_search_opts={},
        **kwargs):
    """
    Using the trajectory as evidence for MLE beta and the probability of each
    destination in the destination set, calculate the expected number of times
    that the agent will enter each state.

    Params:
        g [GridWorldMDP] -- The MDP in which the agent resides.
        init_state [int] -- The agent's current state.
        dest_or_dests [int] -- The agent's goal, or a list of potential goals.
        T [int] -- Make an inference for state probabilities at this many
            timesteps in the future.
        dest_probs [np.ndarray] -- The posterior probability of each
            destination.
        beta_or_betas [float] (optional) -- The irrationality coefficient, or
            a list of irrationality coefficients coresponding to each of the
            agent's potential goals.
        cached_action_probs [np.ndarray] (optional) -- Cached results from
            `MDP.action_probabilities()`. Mainly useful for testing, since
            these results are already cached by default.
        verbose_return [bool] (optional): If True, then this function also
            returns the MLE beta and calculated destination probabilities.
    Returns:
        See the documentation for `infer_from_start`.
    """
    assert len(traj) > 0, traj
    s_b = g.transition(*traj[-1])
    dest_list = unpack_opt_list(dest_or_dests)
    if beta_or_betas is not None:
        betas = unpack_opt_list(beta_or_betas)
        dest_probs = None
    else:
        if hmm:
            dest_probs, betas = destination.hmm_infer(g, traj, dest_list,
                    beta_guesses=beta_guesses, bin_search_opts=bin_search_opts,
                    **hmm_opts)
        else:
            dest_probs, betas = destination.infer(g, traj, dest_list,
                    beta_guesses=beta_guesses,
                    bin_search_opts=bin_search_opts)

    return infer_from_start(g, s_b, dest_list, dest_probs=dest_probs,
            T=T, verbose=verbose, beta_or_betas=betas, **kwargs)


def _main():
    from mdp import GridWorldMDP
    from util import display
    N = 50
    default_reward = -5
    g = GridWorldMDP(N, N, default_reward=default_reward, euclidean_rewards=True)
    for beta in [1, 2, 3, 4]:
        print("Expected occupancies for beta={}").format(beta)
        D = infer_from_start(g, 0, N*N-1, T=N, beta=beta).reshape(N, N)
        print(D)


if __name__ == '__main__':
    _main()
