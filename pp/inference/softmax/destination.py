from __future__ import division
import numpy as np
from numpy import random

from ...mdp.softmax import backwards_value_iter, forwards_value_iter

from ...util import sum_rewards, normalize

def infer_destination(mdp, traj, beta=1, prior=None, dest_set=None,
        V_a_cached=None, V_b_cached=None, vi_precision=1e-5, nachum=False,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the probability of each destination given the trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
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
        V_a_cached [list-like]: (optional) An `mdp.S`-length list with the backwards
            softmax values of each state, given the initial state matches that of
            the trajectory `traj`.
        V_b_cached [list-like]: (optional) An `mdp.S`-length list with the backwards
            softmax values of each state, given the initial state matches
            the state that the last state-action pair in `traj` would transition into.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        P_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the probability that the agent's destination is state i,
            given the provided trajectory.
    """
    assert beta >= 0
    assert len(traj) > 0
    for s, a in traj:
        assert s >= 0 and s < mdp.S, s
        assert a >= 0 and a < mdp.A, a
    if prior is not None:
        assert len(prior) == mdp.S, len(prior)
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S
    assert V_a_cached is None or len(V_a_cached) == mdp.S, V_a_cached
    assert V_b_cached is None or len(V_b_cached) == mdp.S, V_b_cached

    if dest_set != None:
        all_set = set(i for i in xrange(mdp.S))
        assert dest_set.issubset(all_set), (dest_set, mdp.S)

        # If there is only one dest, all probability goes to that dest.
        if len(dest_set) == 1:
            res = np.zeros(mdp.S)
            for d in dest_set:
                res[d] = 1
            return res

        # Remove probability from nondestinations
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0

    # XXX: Check for ABSORB. A state where ABSORB action was taken is automatically
    # the goal state.  [low-priority because softmax is out of fashion]

    traj_reward = sum_rewards(mdp, traj)

    if traj_reward == -np.inf:
        # Something is probably wrong with our model if we observed agent
        # choosing an illegal action other than ABSORB
        print u"Warning: -inf traj_reward in infer_destination."

    def _calc_values(init_state):
        return backwards_value_iter_fn(mdp, init_state, beta=beta,
                update_threshold=vi_precision, verbose=verbose)

    if V_a_cached is None:
        S_a = traj[0][0]
        V_a = _calc_values(S_a)
    else:
        V_a = V_a_cached

    if V_b_cached is None:
        S_b = mdp.transition(*traj[-1])
        V_b = _calc_values(S_b)
    else:
        V_b = V_b_cached

    # XXX: correct numerical errors due to large magnitude before exp
    # updatable = (prior > 0)
    # P_dest = traj_reward + V_b - V_a
    P_dest = np.zeros(mdp.S)
    for C in xrange(mdp.S):
        if nachum:
            P_dest[C] = np.exp((traj_reward + V_b[C] - V_a[C])/beta)
        else:
            P_dest[C] = np.exp(traj_reward + V_b[C] - V_a[C])
        if prior is not None:
            P_dest[C] *= prior[C]
    return normalize(P_dest)
