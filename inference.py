import numpy as np
from numpy import random
import random

from value_iter import backwards_value_iter, forwards_value_iter

def _sum_rewards(mdp, traj):
    return sum(map(lambda x: mdp.rewards[x[0], x[1]], traj))

def _normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def _display(mdp, traj, init_state, goal_state, overlay=False):
    init_state = mdp.state_to_coor(init_state)
    goal_state = mdp.state_to_coor(goal_state)
    visited = {mdp.state_to_coor(s) for s, a in traj}
    for r in range(mdp.rows):
        line = ['_'] * mdp.cols
        for c in range(mdp.cols):
            if (r, c) in visited:
                line[c] = '#'
        if overlay:
            if r == init_state[0]:
                line[init_state[1]] = 'A' if init_state in visited else 'a'
            if r == goal_state[0]:
                line[goal_state[1]] = 'G' if goal_state in visited else 'g'
        print(line)

def simulate(mdp, initial_state, goal_state, path_length=None):
    """
    Generate a sample trajectory of a softmax agent's behavior.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        initial_state [int]: The state that the agent starts in.
        dest_state [int]: The agent's goal state.
        path_length [int]: (optional) If the returned trajectory
            has length more than `path_length`, the return
            value is truncated.

    Return:
        traj [list]: A list of (int, int) pairs, representing
            the agent's state and action at each timestep.
    """
    mdp = mdp.copy()
    mdp.set_goal(goal_state)

    # V(state->goal)
    V = forwards_value_iter(mdp, goal_state, max_iters=1000)

    if path_length == None:
        path_length = float('inf')

    traj = []
    s = initial_state
    while len(traj) < path_length:
        a = sample_action(mdp, s, goal_state, V)
        traj.append([s, a])
        if a == mdp.Actions.ABSORB:
            break
        else:
            s = mdp.transition(s, a)
    return traj

def sample_action(mdp, state, goal, cached_values=None):
    """
    Choose an action probabilistically, like a softmax agent would.
    Params:
        mdp [GridWorldMDP]: The MDP that the agent is playing.
        state [int]: The state that the agent is in.
        cached_values [np.ndarray]: (optional) Precalculated values from all states to goal.
            (Calculated using forward value iteration)
    Return:
        a [int]: An action.
    """
    mdp.set_goal(goal)

    if cached_values != None:
        V = cached_values
    else:
        V = forwards_value_iter(mdp, goal, max_iters=1000)

    P = np.zeros(mdp.A)
    for a in range(mdp.A):
        s_prime = mdp.transition(state, a)
        P[a] = mdp.rewards[state, a] + V[s_prime] - V[state]

    P = np.exp(P)
    P = P / sum(P)
    return np.random.choice(list(range(mdp.A)), p=P)

def infer_destination(mdp, traj, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the probability of each destination given the trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
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
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S

    if dest_set != None:
        all_set = {i for i in range(mdp.S)}
        assert dest_set.issubset(all_set), (dest_set, mdp.S)
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0

    # TODO: Check for ABSORB. A state where ABSORB action was taken is automatically
    # the goal state.

    traj_reward = _sum_rewards(mdp, traj)

    if traj_reward == -np.inf:
        # Something is probably wrong with our model if we observed agent
        # choosing an illegal action other than ABSORB
        print("Warning: -inf traj_reward in infer_destination.")

    S_a = traj[0][0]
    V_a = backwards_value_iter_fn(mdp, S_a, verbose=verbose)
    S_b = mdp.transition(*traj[-1])
    V_b = backwards_value_iter_fn(mdp, S_b, verbose=verbose)

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
