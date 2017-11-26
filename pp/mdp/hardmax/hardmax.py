from __future__ import division
import numpy as np
import Queue

from ..softact_shared import q_values as _q_values
from ..softact_shared import action_probabilities \
        as _action_probabilities

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

def q_values(mdp, goal_state):
    return _q_values(mdp, goal_state, forwards_value_iter)

def action_probabilities(mdp, goal_state, **kwargs):
    return _action_probabilities(mdp, goal_state, q_values, **kwargs)

def trajectory_probability(mdp, goal_state, traj, beta=1, cached_act_probs=None):
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
