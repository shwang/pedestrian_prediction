import numpy as np
from numpy import random
import random

from value_iter import backwards_value_iter

def _sum_rewards(mdp, traj):
    return sum(map(lambda x: mdp.rewards[x[0], x[1]], traj))

def _normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def simulate(mdp, initial_state, goal_state, path_length=None):
    """
    Generate a sample trajectory of a softmax agent's behavior.

    Params:
        mdp [MDP]: The world that the agent is acting in.
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

    if path_length == None:
        path_length = float('inf')

    traj = []
    s = initial_state
    while len(traj) < path_length:
        a = sample_action(mdp, s, goal_state)
        traj.append([s, a])
        if a == mdp.Actions.ABSORB:
            break
        else:
            s = mdp.transition(s, a)
    return traj

def sample_action(mdp, state, goal):
    """
    Choose an action probabilistically, like a softmax agent would.
    Params:
        mdp [MDP]: The MDP that the agent is playing.
        state [int]: The state that the agent is in.
    Return:
        a [int]: An action.
    """
    mdp.set_goal(goal)
    V = backwards_value_iter(mdp, state, goal)
    P = np.zeros(mdp.A)
    for a in range(mdp.A):
        s_prime = mdp.transition(state, a)
        P[a] = mdp.rewards[state, a] + V[s_prime] - V[state]

    P = np.exp(P)
    P = P / sum(P)
    return np.random.choice(list(range(mdp.A)), p=P)

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
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)

    traj_reward = _sum_rewards(mdp, traj)

    S_a = traj[0][0]
    V_a = backwards_value_iter(mdp, S_a)
    S_b = mdp.transition(*traj[-1])
    V_b = backwards_value_iter(mdp, S_b)

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
