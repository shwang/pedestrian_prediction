from __future__ import division

import numpy as np

from ...mdp.hardmax import action_probabilities

def infer_from_start(mdp, init_state, dest, T, verbose=False, beta=1,
        cached_action_prob=None, all_steps=True):
    """
    Params:
        mdp [mdp.GridWorldMDP] -- The MDP in which the agent resides.
        init_state [int] -- The agent's current state.
        dest [int] -- The agent's goal.
        T [int] -- Make an inference for state probabilities at this many
            timesteps in the future.
        beta [float] (optional) -- The irrationality coefficient.
        cached_action_prob [np.ndarray] (optional) -- Cached results from
            `inference.hardmax.action_probabilities()`.
        all_steps [bool] (optional) -- If True, then return all state
            probabilities up to T in a 2D array.
    Returns:
        If `all_steps` is True, then returns all state probabilities for
        timesteps 0,1,...,T in a 2D array. (with dimension (T+1) x S, where
        S is the number of states)

        Otherwise, returns a 1D array with the state probabilities for the
        Tth timestep. (with dimension S, where S is the number of states).


    [The commented out code is scaffolding for future support of multiple
    destinations. (currently only supports a single destination).]
    """
    # if prior is None:
    #     prior = np.ones(mdp.S) / mdp.S
    # if dest_set is not None:
    #     for s in range(mdp.S):
    #         if s not in dest_set:
    #             prior[s] = 0
    #     prior /= sum(prior)

    if cached_action_prob is not None:
        action_prob = cached_action_prob
        assert action_prob.shape == (mdp.S, mdp.A)
        # for v in action_prob.values():
        #     assert action_prob.shape[:1] == (mdp.S, mdp.A)
    else:
        # for dest in range(mdp.S):
        #     if prior[dest] == 0:
        #         continue
        #     action_prob = {}
        #     action_prob[dest] = action_probabilities(mdp, dest)
        action_prob = action_probabilities(mdp, dest, beta=beta)

    res = np.zeros([T+1, mdp.S])
    res[0][init_state] = 1
    for t in range(1, T+1):
        P = res[t-1]
        P_prime = res[t]
        # import pdb; pdb.set_trace()
        # TODO: loop over dest
        for s in range(mdp.S):
            if P[s] == 0:
                continue
            for a, s_prime in mdp.neighbors[s]:
                # P_prime += prior[dest] * P[s] * action_prob[dest, s, a]
                P_prime[s_prime] += P[s] * action_prob[s, a]

    if all_steps:
        return res
    else:
        return res[T]

def infer(mdp, traj, dest, **kwargs):
    """
    Like infer_from_start, but uses the end of `traj` as the initial state.
    """
    assert len(traj) > 0
    s_a = traj[0][0]
    s_b = mdp.transition(traj[-1])
    return infer_from_start(mdp, s_b, dest, **kwargs)
