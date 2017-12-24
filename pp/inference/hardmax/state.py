from __future__ import division

import numpy as np
import destination

from ...parameters import val_default
from ...util.args import unpack_opt_list

def infer_simple(g, init_state, dest, T, beta=1, action_prob=None,
        val_mod=val_default):
    """
    Calculate expected states for a `beta`-rational agent over each timestep
    from 0 to T inclusive, given a fixed destination `dest`.

    Return the probability the agent is in each state during each timestep
    as a 2D array of dimension (T+1, g.S).
    """
    if action_prob is None:
        action_prob = val_mod.action_probabilities(g, dest, beta=beta)

    P_t = np.zeros([T+1, g.S])
    P_t[0][init_state] = 1
    for t in range(1, T+1):
        P = P_t[t-1]
        P_prime = P_t[t]
        for s in range(g.S):
            if P[s] == 0:
                continue
            for a, s_prime in g.neighbors[s]:
                P_prime[s_prime] += P[s] * action_prob[s, a]
    return P_t


def infer_from_start(g, init_state, dest_or_dests, T, dest_probs=None,
        verbose=False, beta_or_betas=1,
        cached_action_probs=None, all_steps=True, val_mod=val_default):
    """
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
            `inference.hardmax.action_probabilities()`.
        all_steps [bool] (optional) -- If True, then return all state
            probabilities up to T in a 2D array.
    Returns:
        If `all_steps` is True, then returns all state probabilities for
        timesteps 0,1,...,T in a 2D array. (with dimension (T+1) x mdp.S)

        Otherwise, returns a 1D array with the state probabilities for the
        Tth timestep. (with dimension mdp.S)
    """
    if T is None:
        T = g.rows + g.cols

    dests = unpack_opt_list(dest_or_dests)
    L = len(dests)

    betas = np.array(unpack_opt_list(beta_or_betas, extend_to=L))
    assert len(betas) == L, betas

    # Unpack cached_action_probs, if applicable.
    if cached_action_probs is not None:
        if len(cached_action_probs.shape) == 2:
            act_probs = [cached_action_probs] * L
    else:
        act_probs = [None] * L
    assert len(act_probs) == L, act_probs

    # Unpack dest_prob
    if dest_probs is None:
        dest_probs = [1] * L
    assert len(dest_probs) == L
    dest_probs = np.array(dest_probs) / sum(dest_probs)

    P = np.zeros([T+1, g.S])
    for dest, dest_prob, beta, act_prob in \
            zip(dests, dest_probs, betas, act_probs):
        P_D = infer_simple(g, init_state, dest, T, beta=beta,
                action_prob=act_prob, val_mod=val_mod)
        assert P.shape == P_D.shape
        np.multiply(P_D, dest_prob, out=P_D)
        np.add(P, P_D, out=P)

    if all_steps:
        return P
    else:
        return P[T]


def infer(g, traj, dest_or_dests, T=None, verbose=False, beta_or_betas=None,
        hmm=False, hmm_opts={},
        auto_beta=True, beta_guesses=None, bin_search_opts={},
        **kwargs):
    """
    If beta_or_betas not provided, then use MLE beta.
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
