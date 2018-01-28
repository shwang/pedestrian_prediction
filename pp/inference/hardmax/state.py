from __future__ import division

import numpy as np
import destination
import beta as bt

from ...mdp import GridWorldExpanded

from ...parameters import val_default
from ...util.args import unpack_opt_list

def infer_joint(g, dests, betas, T, use_gridless=False, traj=[],
        init_state=None, priors=None, k=None,
        verbose_return=False, epsilon=0.02):
    """
    Calculate the expected state probabilties by taking a linear combination
    over the state probabilities associated with each dest-beta pair. The
    weights in this linear combination correspond to the joint posterior
    probability of (beta, dest) given that `beta` is a member of `betas`,
    `dest` is a member of `dests`, and the observed trajectory `traj`.

    Params:
        g [GridWorldMDP or GridWorldExpanded]: The MDP.
        dests [list of ints]: The states that are possible destinations.
        betas [list of floats]: The possible rationality constants.
        T [int]: The number of timesteps to predict into the future.
        use_gridless [bool]: If True, then the required format for the `traj`
            parameter is changed. As a sanity check, `use_gridless=True`
            requires that `g` is an instance of `GridWorldExpanded`.
        traj [list] (optional):
            If `use_gridless` is False, then `traj` should be a list of
            (state, action) pairs describing the Human's motion so far.

            If `use_gridless` is True, then `traj` should be a list of
            (x, y) pairs, where x and y are floats. The ith entry of `traj`
            describes the Human's observed location at time `i`. It is assumed
            that timesteps are chosen appropriately so that every timestep,
            the Human either moves about 1 unit of Euclidean distance or is
            standing in place.
        init_state [int] (optional):
            If `traj` is not given, then `init_state` must be set.
            `init_state` provides the Human's current position when this
            information cannot be inferred from `traj`.
        priors [np.ndarray] (optional): If D is the size of `dests` and B is
            the size of `betas`, then priors is a D x B array. The `(d, b)`th
            entry of `priors` is the prior joint probability:
                P(dest=dests[d], beta=betas[b]).
            If priors is not given, then a uniform prior is assumed.
        k [int] (optional): The trajectory-forgetting parameter. If k is given,
            then only consider the last k timesteps of the trajectory when
            performing compuations.  XXXXX Not yet implemented XXXXX
        verbose_return [bool]: If True, then return additional results, as
            described below.

    Returns:
        occ_res [np.ndarray]: A (T+1 x S) array, where the `t`th entry is the
            probability of state S in `t` timesteps from now.
    Verbose Returns:
        occ_all [np.ndarray]: A (|betas| x T+1 x S) array, where the `b`th entry
            is the (T+1 x S) expected states probabilities if it were the case
            that `beta_star == beta[b]`.
        P_joint_DB [np.ndarray]: A (|dests| x |betas|) dimension array, where
            the `b`th entry is the posterior probability associated with
            `betas[b]`.
    """
    assert len(traj) > 0 or init_state is not None
    if use_gridless:
        assert isinstance(g, GridWorldExpanded)
    if len(traj) > 0:
        if not use_gridless:
            init_state = g.transition(*traj[-1])
        else:
            x = int(round(traj[-1][0] - 0.5))
            y = int(round(traj[-1][1] - 0.5))
            init_state = g.coor_to_state(x, y)

    assert dests is not None
    assert betas is not None

    P_joint_DB = destination.infer_joint(g, dests=dests, betas=betas, traj=traj,
            priors=priors, verbose_return=False, epsilon=epsilon,
            use_gridless=use_gridless)
    n_D, n_B = len(dests), len(betas)
    assert P_joint_DB.shape == (n_D, n_B)

    # State prediction
    occ_all = np.empty([n_D, n_B, T+1, g.S])
    occ_res = np.zeros([T+1, g.S])
    for i, dest in enumerate(dests):
        for j, beta in enumerate(betas):
            occ_all[i, j] = infer_simple(g, init_state, dest=dest, T=T,
                    beta=beta)

    weighted = np.multiply(occ_all, P_joint_DB.reshape(n_D, n_B, 1, 1))
    # occ_res = np.add(occ_res, occ_all[i,j] * P_joint_DB[i, j], out=occ_res)
    occ_res = np.sum(np.sum(weighted, axis=0), axis=0)
    assert occ_res.shape == (T+1, g.S)

    if verbose_return:
        return occ_res, occ_all, P_joint_DB
    else:
        return occ_res

def infer_bayes(g, dest, T, betas, traj=[], init_state=None, priors=None,
        k=None,
        action_prob=None, verbose_return=False):
    """
    Calculate the expected state probabilties by taking a linear combination
    over the state probabilities associated with each beta in `betas`. The
    weights in this linear combination correspond to the posterior probability
    of each beta given that `beta_star` is actually in `betas` and the observed
    trajectory `traj`.

    Returns:
        occ_res [np.ndarray]: A (T+1 x S) array, where the `t`th entry is the
            probability of state S in `t` timesteps from now.
    Verbose Returns:
        occ_all [np.ndarray]: A (|betas| x T+1 x S) array, where the `b`th entry
            is the (T+1 x S) expected states probabilities if it were the case
            that `beta_star == beta[b]`.
        P_betas [np.ndarray]: A (|betas|) dimension array, where the `b`th entry
            is the posterior probability associated with `betas[b]`.
    """
    assert len(traj) > 0 or init_state is not None
    if len(traj) > 0:
        init_state = g.transition(*traj[-1])

    assert betas is not None
    P_beta = bt.calc_posterior_over_set(g, traj=traj, goal=dest, betas=betas,
            k=k, priors=priors)

    occ_all = np.empty([len(betas), T+1, g.S])
    occ_res = np.zeros([T+1, g.S])
    for i, beta in enumerate(betas):
        occ_all[i] = infer_simple(g, init_state, dest=dest, T=T,
                action_prob=action_prob, beta=beta)
        occ_res = np.add(occ_res, occ_all[i] * P_beta[i], out=occ_res)

    if verbose_return:
        return occ_res, occ_all, P_beta
    else:
        return occ_res


def infer_simple(g, init_state, dest, T, beta=1, action_prob=None,
        val_mod=val_default):
    """
    Calculate expected states for a `beta`-rational agent over each timestep
    from 0 to T inclusive, given a fixed destination `dest`.

    Return the probability the agent is in each state during each timestep
    as a 2D array of dimension (T+1, g.S).
    """
    assert init_state is not None
    g.set_goal(dest)
    M = g.transition_probabilities(beta=beta, act_probs_cached=action_prob)

    P_t = np.zeros([T+1, g.S])
    P_t[0][init_state] = 1
    for t in range(1, T+1):
        P = P_t[t-1]
        P_prime = P_t[t]
        P_prime[:] = np.matmul(M, P)
    return P_t


def infer_from_start(g, init_state, dest_or_dests, T, dest_probs=None,
        verbose=False, beta_or_betas=1,
        cached_action_probs=None, verbose_return=True, val_mod=val_default):
    """
    Infer state probabilities over the next T timesteps before evidence for the
    true destination and irrationality coefficent of the agent is observed.

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
            `inference.hardmax.action_probabilities()`.
        verbose_return [bool] (optional) -- If True, then return all state
            probabilities up to T in a 2D array, the MLE betas and
            destination probabilities.

    Returns:
        If `verbose_return` is False, returns `P`, a 1D array with the state
        probabilities for the Tth timestep. (with dimension mdp.S)

        If `verbose_return` is True, then returns (P, betas, dest_probs).
        P [np.ndarray] -- the state probabilities for timesteps 0,1,...,T in a
            2D array (with dimension (T+1) x mdp.S).
        betas [np.ndarray]-- a 1D array which contains the beta associated with
            each destination.
        dest_probs [np.ndarray]-- a 1D array which contains the probability of
            each destination being the true destination.
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

    if verbose_return:
        return P, betas, dest_probs
    else:
        return P[T]


def infer(g, traj, dest_or_dests, T=None, verbose=False, beta_or_betas=None,
        hmm=False, hmm_opts={},
        beta_guesses=None, bin_search_opts={},
        **kwargs):
    """
    Infer the state probabilities over the next T timesteps and a nonempty
    trajectory, which may be used to estimate destination probabilities and
    the MLE beta associated with each destination.

    Params:
    g [GridWorldMDP] -- The agent's MDP.
    traj [list of (int, int)] -- A list of state-action pairs describing the
        agent's trajectory.
    dest_or_dests [int or np.ndarray]: A single destination, or a list of
        possible destinations.
    T [int]: The number of timesteps to predict into the future.
    beta_or_betas [int or np.ndarray] (optional): A fixed beta, or a list of
        betas corresponding to each destination. If not provided, then
        automatically compute MLE betas from the trajectory for each
        destination.
    beta_guesses [list of float] (optional): A list of initial guesses for the
        MLE beta corresponding to each destination.
    hmm [bool] (optional): Whether or not to use the 'epsilon-greedy' HMM model
        for the agent's destination.

    verbose_return [bool] (optional): If True, then this function also returns
        the MLE beta and calculated destination probabilities.

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
