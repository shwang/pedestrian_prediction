import numpy as np

from value_iter import backwards_value_iter

def _sum_rewards(mdp, traj):
    return sum(map(lambda x: mdp.rewards[x[0], x[1]], traj))

def _normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def infer_destination(mdp, traj, prior=None):
    """
    Calculate the probability of each destination given the trajectory so far.

    Params:
        mdp [MDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
    Return:
        P_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the probability that the agent's destination is state i,
            given the provided trajectory.
    """
    assert len(traj) > 0
    for s, a in traj:
        assert s >= 0 and s < mdp.S, s
        assert a >= 0 and a < mdp.A, a
    if prior != None:
        assert len(prior) == mdp.S, len(prior)
        assert sum(prior) - 1.0 < 1e-7, (sum(prior), prior)

    traj_reward = _sum_rewards(mdp, traj)

    S_a = traj[0][0]
    # TODO(shwang): remove max_iters when backwards_value_iter converges
    V_a = backwards_value_iter(mdp, S_a, max_iters=mdp.S)
    S_b = mdp.transition(*traj[-1])
    V_b = backwards_value_iter(mdp, S_b, max_iters=mdp.S)

    P_dest = np.zeros(mdp.S)
    for C in range(mdp.S):
        P_dest[C] = np.exp(traj_reward + V_b[C] - V_a[C])
        if prior != None:
            P_dest[C] *= prior[C]
    return _normalize(P_dest)

def infer_state_frequency(traj):
    """
    Calculate the probability of occupying each state given the trajectory so far.
    """
