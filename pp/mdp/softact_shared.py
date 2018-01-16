from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize

def q_values(mdp, goal_state, forwards_value_iter):
    """
    Calculate a hardmax agent's Q values for each state action pair.
    For hardmax forwards_value_iter only. In other words, these q_values
    correspond to maximum future value from the state-action pair.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state, where the agent is forced to choose
            the absorb action, and whose state value is 0.
    """
    if goal_state in mdp.q_cache:
        return np.copy(mdp.q_cache[goal_state])

    mdp.set_goal(goal_state)
    V = forwards_value_iter(mdp, goal_state)

    Q = np.empty([mdp.S, mdp.A])
    Q.fill(-np.inf)
    for s in range(mdp.S):
        if s == goal_state:
            Q[s, mdp.Actions.ABSORB] = 0
        else:
            for a in range(mdp.A):
                Q[s,a] = mdp.rewards[s,a] + V[mdp.transition(s,a)]
    assert Q.shape == (mdp.S, mdp.A)

    mdp.q_cache[goal_state] = Q
    return np.copy(Q)

# TODO: get rid of manualy goal-setting
def action_probabilities(mdp, goal_state, q_values, beta=1, q_cached=None):
    """
    At each state, calculate the softmax probability of each action
    using hardmax Q values.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state. At the goal state, the agent
            always chooses the ABSORB action at no cost.
    """
    assert beta > 0, beta
    if q_cached is not None:
        Q = np.copy(q_cached)
    else:
        Q = q_values(mdp, goal_state)

    np.divide(Q, beta, out=Q)
    # Use amax to mitigate numerical errors
    amax = np.amax(Q, axis=1, keepdims=1)
    np.subtract(Q, amax, out=Q)

    np.exp(Q, out=Q)
    return normalize(Q, norm='l1', copy=False)

def transition_probabilities(g, action_probabilities, beta=1,
        act_probs_cached=None):
    """
    Calculate the SxS state probability transition matrix `T` for a
    beta-irrational agent.
    """
    assert beta > 0, beta
    if act_probs_cached is None:
        P = action_probabilities(g, g.goal, beta=beta, q_cached=None)
    else:
        P = act_probs_cached

    tup_ref = g.transition_cached_t

    T = np.zeros([g.S, g.S])
    for s in range(g.S):
        for a in range(g.A):
            s_prime = tup_ref[s*g.A + a]
            T[s_prime, s] += P[s, a]
    return T

# TODO: Optimization: don't calculate all the action probabilities when I
#       only need a small subset of them. Over-calculating becomes a big
#       runtime problem when N is large (200). ^[citation needed]
def trajectory_probability(mdp, goal_state, traj, action_probabilities,
        beta=1, cached_act_probs=None):
    """
    Calculate the product of the probabilities of each
    state-action pair in this trajectory given an mdp,
    a goal_state, and beta.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state. At the goal state, the agent
            always chooses the ABSORB action at no cost.
        traj [list of (int, int)]: A list of state-action pairs. If this
            is an empty list, return traj_prob=1.
        beta [float] (optional): Irrationality constant.
        cached_act_probs [ndarray] (optional): Cached results of
            action_probabilities. Mainly for testing purposes.
    Return:
        traj_prob [float].
    """
    if len(traj) == 0:
        return 1

    if cached_act_probs is None:
        P = action_probabilities(mdp, goal_state, beta=beta)
    else:
        P = cached_act_probs

    traj_prob = 1
    for s, a in traj:
        traj_prob *= P[s, a]
    return traj_prob
