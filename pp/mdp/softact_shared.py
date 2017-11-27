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

    mdp.q_cache[goal_state] = np.copy(Q)
    return Q

def action_probabilities(mdp, goal_state, q_values, beta=1, q_cached=None):
    """
    At each state, calculate the softmax probability of each action
    using hardmax Q values.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state. At the goal state, the agent
            always chooses the ABSORB action at no cost.
    """
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
