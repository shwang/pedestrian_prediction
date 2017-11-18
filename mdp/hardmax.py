from __future__ import division
import numpy as np
from sklearn.preprocessing import normalize
import Queue

def forwards_value_iter(mdp, goal_state, *args, **kwargs):
    kwargs["forwards"] = True
    return _value_iter(mdp, goal_state, *args, **kwargs)

def backwards_value_iter(mdp, init_state, *args, **kwargs):
    kwargs["forwards"] = False
    return _value_iter(mdp, init_state, *args, **kwargs)

def _value_iter(mdp, s, forwards, beta=1, verbose=False):
    """
    If forwards is False,
    Calculate the reward of the optimal trajectory to each state, starting from
    `s`.

    If forwards is True,
    Calculate the reward of the optimal trajectory from each state to `s`

    Runs in |S| log |S| time, where |S| is the number of states.

    Params:
        mdp [GridWorldMDP]: The MDP.
        s [int]: The init_state or the goal_state, depending on the
            value of forwards. In either case, V[s] = 0.
        forwards [bool]: Described in the function summary.
        lazy_s [bool]: TODO

    Returns:
        V [np.ndarray]: An `mdp.S`-length vector, where the ith entry is
            the reward of the optimal trajectory from the starting state to
            state i.
    """
    # Dijkstra is only valid if all costs are nonnegative.
    assert (mdp.rewards <= 0).all()

    V = np.ndarray(mdp.S)
    pq = Queue.PriorityQueue()
    visited = set()
    # entry := (cost, node)
    pq.put((0, s))
    while not pq.empty():
        cost, state = pq.get()
        if state in visited:
            continue
        V[state] = cost
        visited.add(state)

        if forwards:
            s_prime = state
            for (a, s) in mdp.reverse_neighbors[s_prime]:
                reward = mdp.rewards[s, a] / beta
                if reward == -np.inf or s in visited:
                    continue
                pq.put((-reward + cost, s))
        else:
            for a in mdp.Actions:
                reward = mdp.rewards[state, a] / beta
                s_prime = mdp._transition(state, a)
                if reward == -np.inf or s_prime in visited:
                    continue
                pq.put((-reward + cost, s_prime))

    return -V

def q_values(mdp, goal_state, beta=1):
    """
    Calculate a hardmax agent's Q values for each state action pair.
    For hardmax forwards_value_iter only. In other words, these q_values
    correspond to maximum future value from the state-action pair.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state, where the agent is forced to choose
            the absorb action, and whose state value is 0.
    """
    mdp = mdp.copy()
    mdp.set_goal(goal_state)
    V = forwards_value_iter(mdp, goal_state, beta=beta)

    Q = np.empty([mdp.S, mdp.A])
    Q.fill(-np.inf)
    for s in range(mdp.S):
        if s == goal_state:
            Q[s, mdp.Actions.ABSORB] = 0
        else:
            for a in range(mdp.A):
                Q[s,a] = mdp.rewards[s,a]/beta + V[mdp.transition(s,a)]
    assert Q.shape == (mdp.S, mdp.A)
    return Q

def action_probabilities(mdp, goal_state, beta=1, q_cached=None):
    """
    At each state, calculate the softmax probability of each action
    using hardmax Q values.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: The goal state. At the goal state, the agent
            always chooses the ABSORB action at no cost.
    """
    if q_cached is not None:
        Q = q_cached
    else:
        Q = q_values(mdp, goal_state, beta=beta)

    # Use amax to mitigate numerical errors
    amax = np.amax(Q, axis=1, keepdims=1)
    np.subtract(Q, amax, out=Q)

    np.exp(Q, out=Q)
    return normalize(Q, norm='l1', copy=False)
