import numpy as np
import Queue

def dijkstra(mdp, init_state, verbose=False):
    """
    Calculate the reward of the optimal trajectory to each state, starting from
    `init_state`. Runs in |S| log |S| time, where |S| is the number of states.

    Params:
        mdp [GridWorldMDP]: The MDP.
        init_state [int]: The starting state. The reward of the optimal trajectory
            to the starting state is 0.

    Returns:
        R_star [np.ndarray]: An `mdp.S`-length vector, where the ith entry is
            the reward of the optimal trajectory from the starting state to
            state i.
    """
    # Dijkstra is only valid if all costs are nonnegative.
    assert (mdp.rewards <= 0).all()

    R_star = np.ndarray(mdp.S)
    pq = Queue.PriorityQueue()
    visited = set()
    # entry := (cost, node)
    pq.put((0, init_state))
    while not pq.empty():
        cost, state = pq.get()
        if state in visited:
            continue
        R_star[state] = cost
        visited.add(state)

        for a in mdp.Actions:
            reward = mdp.rewards[state, a]
            s_prime = mdp._transition(state, a)
            if reward == -np.inf or s_prime in visited:
                continue
            pq.put((-reward + cost, s_prime))

    return -R_star
