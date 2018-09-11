import numpy as np

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

    P = mdp.action_probabilities(goal_state, beta=beta)

    if path_length == None:
        path_length = np.inf

    traj = []
    s = initial_state
    while len(traj) < path_length:
        a = sample_action(mdp, s, goal_state, beta=beta, cached_probs=P)
        assert a is not None
        traj.append([s, a])
        if a == mdp.Actions.ABSORB:
            break
        else:
            s = mdp.transition(s, a)
    return traj

def sample_action(mdp, state, goal, beta=1, cached_probs=None,
        absorb_only_on_goal=True):
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
    assert beta > 0 and beta != np.inf, "beta={} not supported".format(beta)

    if cached_probs is not None:
        P = cached_probs
    else:
        P = mdp.action_probabilitilies(goal, beta=beta)

    if absorb_only_on_goal:
            choice = mdp.Actions.ABSORB
            if state != goal:
                while choice == mdp.Actions.ABSORB:
                    choice = np.random.choice(range(mdp.A), p=P[state])
    else:
        choice = np.random.choice(range(mdp.A), p=P[state])
    return choice

def _main():
    from util import display
    from mdp.mdp import GridWorldMDP

    init = 0
    goal = 35
    default_reward = -2
    beta = 10
    g = GridWorldMDP(6, 6, default_reward=default_reward, euclidean_rewards=True)

    traj = simulate(g, 0, goal, beta=beta)
    print "Testing hardmax.simulate:"
    print "  * default_reward={}, beta={}".format(default_reward, beta)
    print "  * traj: {}".format([(g.state_to_coor(s), g.Actions(a)) for s, a in traj])
    display(g, traj, init, goal, overlay=True)

if __name__ == '__main__':
    _main()
