import numpy as np

from ...mdp.softmax import backwards_value_iter, forwards_value_iter

def simulate(mdp, initial_state, goal_state, beta=1, path_length=None):
    """
    Generate a sample trajectory of a softmax agent's behavior.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        initial_state [int]: The state that the agent starts in.
        goal_state [int]: The agent's goal state.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        path_length [int]: (optional) If the returned trajectory
            has length more than `path_length`, the return
            value is truncated.

    Returns:
        traj [list]: A list of (int, int) pairs, representing
            the agent's state and action at each timestep.
    """
    assert beta >= 0, beta

    mdp = mdp.copy()
    mdp.set_goal(goal_state)

    # V(state->goal)
    V = forwards_value_iter(mdp, goal_state, beta=beta, max_iters=1000)

    if path_length == None:
        path_length = float(u'inf')

    traj = []
    s = initial_state
    while len(traj) < path_length:
        a = sample_action(mdp, s, goal_state, beta=beta, cached_values=V)
        traj.append([s, a])
        if a == mdp.Actions.ABSORB:
            break
        else:
            s = mdp.transition(s, a)
    return traj

def sample_action(mdp, state, goal, beta=1, nachum=False, cached_values=None):
    """
    Choose an action probabilistically, like a softmax agent would.
    Params:
        mdp [GridWorldMDP]: The MDP that the agent is playing.
        state [int]: The state that the agent is in.
        goal [int]: The agent's goal state.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        cached_values [np.ndarray]: (optional) Precalculated values from all states to goal.
            (Calculated using forward value iteration)
    Return:
        a [int]: An action.
    """
    assert beta >= 0, beta

    if beta == np.inf:
        # Use uniform choice; would otherwise result in P = 0/0 = NaN.
        return np.random.choice(range(mdp.A))

    if cached_values is not None:
        V = cached_values
        mdp.set_goal(goal)
    else:
        V = forwards_value_iter(mdp, goal, beta=beta, max_iters=1000)

    P = np.zeros(mdp.A)
    for a in xrange(mdp.A):
        s_prime = mdp.transition(state, a)
        if not nachum:
            P[a] = mdp.rewards[state, a]/beta + V[s_prime] - V[state]
        else:
            P[a] = (mdp.rewards[state, a] + V[s_prime] - V[state])/beta

    if beta == 0:
        # Use hardmax choice; would otherwise result in P = inf/inf = NaN.
        argmax = np.array(np.argmax(P))
        if argmax.shape == tuple():
            return argmax
        else:
            return np.random.choice(argmax)

    P = np.exp(P)
    P = P / sum(P)
    return np.random.choice(range(mdp.A), p=P)
