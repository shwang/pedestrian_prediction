from __future__ import division
import numpy as np
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
                s_prime = mdp.transition(state, a)
                if reward == -np.inf or s_prime in visited:
                    continue
                pq.put((-reward + cost, s_prime))

    return -V
