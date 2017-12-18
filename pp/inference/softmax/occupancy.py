from __future__ import division

import numpy as np

from itertools import imap

from .destination import infer_destination
from ...mdp.softmax import backwards_value_iter, forwards_value_iter
from ...mdp.hardmax import backwards_value_iter as dijkstra
from ...util import normalize

def infer_occupancies(mdp, traj, beta=1, gamma=1, prior=None, dest_set=None,
        vi_precision=1e-7,
        nachum=False, backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the expected number of times each state will be occupied given the
    trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        nachum [bool]: (optional) (experimental) If true, then use Nachum-style Bellman
            updates. Otherwise, use shwang-style Bellman updates.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
        dest_set [set]: (optional) A set of states that could be the destination
            state. Equivalent to setting the priors of states not in this set to
            zero and renormalizing. By default, the dest_set contains all possible
            destinations.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        D_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the expected occupancy of state i, given the provided
            trajectory.
    """
    assert len(traj) > 0
    for s, a in traj:
        assert s >= 0 and s < mdp.S, s
        assert a >= 0 and a < mdp.A, a
    if prior != None:
        assert len(prior) == mdp.S, len(prior)
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S

    if dest_set != None:
        all_set = set(xrange(mdp.S))
        assert dest_set.issubset(all_set), (dest_set, mdp.S)
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0
    prior = normalize(prior)

    S_a = traj[0][0]
    V_a = backwards_value_iter_fn(mdp, S_a, beta=beta, gamma=gamma,
            update_threshold=vi_precision, verbose=verbose)
    S_b = mdp.transition(*traj[-1])
    V_b = backwards_value_iter_fn(mdp, S_b, beta=beta, gamma=gamma,
            update_threshold=vi_precision, verbose=verbose)

    P_dest = infer_destination(mdp, traj, beta=beta, prior=prior, dest_set=dest_set,
            V_a_cached=V_a, V_b_cached=V_b,
            backwards_value_iter_fn=backwards_value_iter_fn)

    D_dest = np.zeros(mdp.S)
    for C in xrange(mdp.S):
        if prior[C] == 0:
            continue

        if not nachum:
            goal_val = -V_b[C] + np.log(P_dest[C])
        else:
            goal_val = -V_b[C]/beta + np.log(P_dest[C])


        D_dest += forwards_value_iter(mdp, C, beta=beta,
                    fixed_init_val=goal_val, verbose=verbose)

    # The paper says to multiply by exp(V_a), but exp(V_b) gets better results
    # and seems more intuitive.
    D_dest += V_b
    if not nachum:
        return np.exp(D_dest)
    else:
        res = np.exp(D_dest)
        if dest_set is not None and len(dest_set) == 1:
            # True when only one destination.
            assert np.max(res) == res[list(dest_set)[0]]
        return res / np.max(res)

# XXX: Cached occupancies?

def infer_temporal_occupancies(mdp, traj, T, c_0, sigma_0, sigma_1,
        beta=1, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Given a softmax agent's prior trajectory, approximate the probability
    that a state will be occupied by the softmax agent at a given timestep
    for the next T timesteps, assuming that all trajectories by the agent
    will be equal to T actions in length.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        c_0 [float]: The expected mean reward collected by the agent every timestep.
            This should be a negative number, and should be different for different
            MDPs. (See Ziebart paper).
        sigma_0, sigma_1 [float]: Numbers describing the variance of reward collected
            by the agent over time. (See Ziebart paper).
        T [int]: A positive number indicating the number of timesteps to calculate,
            and also presumed length of any trajectory from the agent.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
        dest_set [set]: (optional) A set of states that could be the destination
            state. Equivalent to setting the priors of states not in this set to
            zero and renormalizing. By default, the dest_set contains all possible
            destinations.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        D_dest_t [np.ndarray]: A mdp.S x T matrix, where the t-th column is a
            normalized vector whose ith entry is the expected occupancy of
            state i at time t+1, given the provided trajectory.
    """
    assert c_0 < 0, c_0
    assert T > 0, T

    S_b = mdp.transition(*traj[-1])
    D_s = infer_occupancies(mdp, traj, beta=beta, prior=prior, dest_set=dest_set,
            backwards_value_iter_fn=backwards_value_iter, verbose=verbose)
    R_star_b = dijkstra(mdp, S_b, verbose=verbose)

    P_dest_t = np.ndarray([T, mdp.S])

    for t in xrange(1, T + 1):
        numer = -np.square(c_0*t - R_star_b)
        denom = 2*(sigma_0**2 + t*sigma_1**2)
        P_dest_t[t-1] = np.exp(numer/denom)

    for s, occupancy in enumerate(D_s):
        P_dest_t[:, s] = occupancy * normalize(P_dest_t[:, s])

    return P_dest_t

def infer_occupancies_from_start(mdp, init_state, beta=1, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the expected number of times each state will be occupied given the
    trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        init_state [int]: The agent's initial state.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
        dest_set [set]: (optional) A set of states that could be the destination
            state. Equivalent to setting the priors of states not in this set to
            zero and renormalizing. By default, the dest_set contains all possible
            destinations.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        D_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the expected occupancy of state i, given the provided
            trajectory.
    """
    assert init_state >= 0 and init_state < mdp.S, init_state
    if prior != None:
        assert len(prior) == mdp.S, len(prior)
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S

    if dest_set != None:
        all_set = set(xrange(mdp.S))
        assert dest_set.issubset(all_set), (dest_set, mdp.S)
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0
    prior = normalize(prior)

    V = backwards_value_iter_fn(mdp, init_state, beta=beta, verbose=verbose)

    D_dest = np.zeros(mdp.S)
    for C in xrange(mdp.S):
        if prior[C] == 0:
            continue

        goal_val = -V[C] + np.log(prior[C])
        # XXX: This temporary implementation will break if there is more than
        # one possible destination. For more information, review Ziebart.
        # [Don't care about softmax for now. We probably aren't going to use it
        # because it is too slow.]
        D_dest += forwards_value_iter(mdp, C, beta=beta,
                    fixed_init_val=goal_val, verbose=verbose)

    D_dest += V
    return np.exp(D_dest)

def infer_temporal_occupancies_from_start(mdp, init_state, T, c_0, sigma_0, sigma_1,
        beta=1, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Given a softmax agent's prior trajectory, approximate the probability
    that a state will be occupied by the softmax agent at a given timestep
    for the next T timesteps, assuming that all trajectories by the agent
    will be equal to T actions in length.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        c_0 [float]: The expected mean reward collected by the agent every timestep.
            This should be a negative number, and should be different for different
            MDPs. (See Ziebart paper).
        sigma_0, sigma_1 [float]: Numbers describing the variance of reward collected
            by the agent over time. (See Ziebart paper).
        T [int]: A positive number indicating the number of timesteps to calculate,
            and also presumed length of any trajectory from the agent.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
        dest_set [set]: (optional) A set of states that could be the destination
            state. Equivalent to setting the priors of states not in this set to
            zero and renormalizing. By default, the dest_set contains all possible
            destinations.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        D_dest_t [np.ndarray]: A mdp.S x T matrix, where the t-th column is a
            normalized vector whose ith entry is the expected occupancy of
            state i at time t+1, given the provided trajectory.
    """
    assert c_0 < 0, c_0
    assert T > 0, T

    D_s = infer_occupancies_from_start(mdp, init_state, beta=beta, prior=prior,
            dest_set=dest_set,
            backwards_value_iter_fn=backwards_value_iter, verbose=verbose)
    R_star_b = dijkstra(mdp, init_state, verbose=verbose)

    P_dest_t = np.ndarray([T, mdp.S])

    for t in xrange(1, T + 1):
        numer = -np.square(c_0*t - R_star_b)
        denom = 2*(sigma_0**2 + t*sigma_1**2)
        P_dest_t[t-1] = np.exp(numer/denom)

    for s, occupancy in enumerate(D_s):
        P_dest_t[:, s] = occupancy * normalize(P_dest_t[:, s])

    return P_dest_t
