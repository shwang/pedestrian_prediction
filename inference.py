import numpy as np
from numpy import random
import random

from value_iter import backwards_value_iter, forwards_value_iter, dijkstra

def _sum_rewards(mdp, traj):
    return sum(map(lambda x: mdp.rewards[x[0], x[1]], traj))

def _normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def _display(mdp, traj, init_state, goal_state, overlay=False):
    init_state = mdp.state_to_coor(init_state)
    goal_state = mdp.state_to_coor(goal_state)

    visited = {mdp.state_to_coor(s) for s, a in traj}
    if len(traj) > 0:
        visited.add(mdp.state_to_coor(mdp.transition(*traj[-1])))
    else:
        visited.add(init_state)

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

def simulate(mdp, initial_state, goal_state, beta=1, path_length=None):
    """
    Generate a sample trajectory of a softmax agent's behavior.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        initial_state [int]: The state that the agent starts in.
        dest_state [int]: The agent's goal state.
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
        path_length = float('inf')

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

def sample_action(mdp, state, goal, beta=1, cached_values=None):
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
    # TODO(sirspinach): Unit tests, especially for beta edge cases. (low priority)
    assert beta >= 0, beta

    # TODO(sirspinach): This uniform choice also chooses ABSORB.
    #                   Get rid of this possibility for non-goal states.
    if beta == np.inf:
        # Use uniform choice; would otherwise result in P = 0/0 = NaN.
        return np.random.choice(list(range(mdp.A)))

    if cached_values is not None:
        V = cached_values
        mdp.set_goal(goal)
    else:
        V = forwards_value_iter(mdp, goal, max_iters=1000)

    P = np.zeros(mdp.A)
    for a in range(mdp.A):
        s_prime = mdp.transition(state, a)
        P[a] = mdp.rewards[state, a]/beta + V[s_prime] - V[state]

    if beta == 0:
        # Use hardmax choice; would otherwise result in P = inf/inf = NaN.
        argmax = np.array(np.argmax(P))
        if argmax.shape == tuple():
            return argmax
        else:
            return np.random.choice(argmax)

    P = np.exp(P)
    P = P / sum(P)
    return np.random.choice(list(range(mdp.A)), p=P)

def infer_destination(mdp, traj, beta=1, prior=None, dest_set=None,
        V_a_cached=None, V_b_cached=None, vi_precision=1e-5,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the probability of each destination given the trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        prior [list-like]: (optional) A normalized vector with length mdp.S, where
            the ith entry is the prior probability that the agent's destination is
            state i. By default, the prior probability is uniform over all states.
        dest_set [set]: (optional) A set of states that could be the destination
            state. Equivalent to setting the priors of states not in this set to
            zero and renormalizing. By default, the dest_set contains all possible
            destinations.
        V_a_cached [list-like]: (optional) An `mdp.S`-length list with the backwards
            softmax values of each state, given the initial state matches that of
            the trajectory `traj`.
        V_b_cached [list-like]: (optional) An `mdp.S`-length list with the backwards
            softmax values of each state, given the initial state matches
            the state that the last state-action pair in `traj` would transition into.
        backwards_value_iter_fn [function]: (optional) Set this parameter to use
            a custom version of backwards_value_iter. Used for testing.
    Return:
        P_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the probability that the agent's destination is state i,
            given the provided trajectory.
    """
    assert beta >= 0
    assert len(traj) > 0
    for s, a in traj:
        assert s >= 0 and s < mdp.S, s
        assert a >= 0 and a < mdp.A, a
    if prior is not None:
        assert len(prior) == mdp.S, len(prior)
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S
    assert V_a_cached is None or len(V_a_cached) == mdp.S, V_a_cached
    assert V_b_cached is None or len(V_b_cached) == mdp.S, V_b_cached

    if dest_set != None:
        all_set = {i for i in range(mdp.S)}
        assert dest_set.issubset(all_set), (dest_set, mdp.S)

        # If there is only one dest, all probability goes to that dest.
        if len(dest_set) == 1:
            res = np.zeros(mdp.S)
            for d in dest_set:
                res[d] = 1
            return res

        # Remove probability from nondestinations
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

    if V_a_cached is None:
        S_a = traj[0][0]
        V_a = backwards_value_iter_fn(mdp, S_a, beta=beta, update_threshold=vi_precision,
                verbose=verbose)
    else:
        V_a = V_a_cached

    if V_b_cached is None:
        S_b = mdp.transition(*traj[-1])
        V_b = backwards_value_iter_fn(mdp, S_b, beta=beta, update_threshold=vi_precision,
                verbose=verbose)
    else:
        V_b = V_b_cached

    # TODO: correct numerical errors due to large magnitude before exp
    # updatable = (prior > 0)
    # P_dest = traj_reward + V_b - V_a
    P_dest = np.zeros(mdp.S)
    for C in range(mdp.S):
        P_dest[C] = np.exp(traj_reward + V_b[C] - V_a[C])
        if prior is not None:
            P_dest[C] *= prior[C]
    return _normalize(P_dest)

def infer_occupancies(mdp, traj, beta=1, prior=None, dest_set=None, vi_precision=1e-7,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the expected number of times each state will be occupied given the
    trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
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
        D_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the expected occupancy of state i, given the provided
            trajectory.
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
        all_set = set(range(mdp.S))
        assert dest_set.issubset(all_set), (dest_set, mdp.S)
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0
    prior = _normalize(prior)

    S_a = traj[0][0]
    V_a = backwards_value_iter_fn(mdp, S_a, beta=beta, update_threshold=vi_precision,
            verbose=verbose)
    S_b = mdp.transition(*traj[-1])
    V_b = backwards_value_iter_fn(mdp, S_b, beta=beta, update_threshold=vi_precision,
            verbose=verbose)

    P_dest = infer_destination(mdp, traj, beta=beta, prior=prior, dest_set=dest_set,
            V_a_cached=V_a, V_b_cached=V_b,
            backwards_value_iter_fn=backwards_value_iter_fn)

    D_dest = np.zeros(mdp.S)
    for C in range(mdp.S):
        if prior[C] == 0:
            continue

        goal_val = -V_b[C] + np.log(P_dest[C])

        D_dest += forwards_value_iter(mdp, C, beta=beta,
                    fixed_init_val=goal_val, verbose=verbose)

    # The paper says to multiply by exp(V_a), but exp(V_b) gets better results
    # and seems more intuitive.
    D_dest += V_b
    return np.exp(D_dest)

# XXX: Cached occupancies?

def infer_temporal_occupancies(mdp, traj, T, c_0, sigma_0, sigma_1,
        beta=1, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Given a softmax agent's prior trajectory, approximate the probability
    that a state will be occupied by the softmax agent at a given timestep
    for the next T timesteps, assuming that all trajectories by the agent
    will be equal to T actions in length.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        traj [list-like]: A nonempty list of (state, action) tuples describing
            the agent's trajectory so far. The current state of the agent is
            inferred to be `mdp.transition(*traj[-1])`.
        c_0 [float]: The expected mean reward collected by the agent every timestep.
            This should be a negative number, and should be different for different
            MDPs. (See Ziebart paper).
        sigma_0, sigma_1 [float]: Numbers describing the variance of reward collected
            by the agent over time. (See Ziebart paper).
        T [int]: A positive number indicating the number of timesteps to calculate,
            and also presumed length of any trajectory from the agent.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
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
        D_dest_t [np.ndarray]: A mdp.S x T matrix, where the t-th column is a
            normalized vector whose ith entry is the expected occupancy of
            state i at time t+1, given the provided trajectory.
    """
    assert c_0 < 0, c_0
    assert T > 0, T

    S_b = mdp.transition(*traj[-1])
    D_s = infer_occupancies(mdp, traj, beta=beta, prior=prior, dest_set=dest_set,
            backwards_value_iter_fn=backwards_value_iter, verbose=verbose)
    R_star_b = dijkstra(mdp, S_b, verbose=verbose)

    P_dest_t = np.ndarray([T, mdp.S])

    for t in range(1, T + 1):
        numer = -np.square(c_0*t - R_star_b)
        denom = 2*(sigma_0**2 + t*sigma_1**2)
        P_dest_t[t-1] = np.exp(numer/denom)

    for s, occupancy in enumerate(D_s):
        P_dest_t[:, s] = occupancy * _normalize(P_dest_t[:, s])

    return P_dest_t

def infer_occupancies_from_start(mdp, init_state, beta=1, prior=None, dest_set=None,
        backwards_value_iter_fn=backwards_value_iter, verbose=False):
    """
    Calculate the expected number of times each state will be occupied given the
    trajectory so far.

    Params:
        mdp [GridWorldMDP]: The world that the agent is acting in.
        init_state [int]: The agent's initial state.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
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
        D_dest [np.ndarray]: A normalized vector with length mdp.S, where the ith
            entry is the expected occupancy of state i, given the provided
            trajectory.
    """
    assert init_state >= 0 and init_state < mdp.S, init_state
    if prior != None:
        assert len(prior) == mdp.S, len(prior)
        assert abs(sum(prior) - 1.0) < 1e-7, (sum(prior), prior)
    else:
        prior = [1] * mdp.S

    if dest_set != None:
        all_set = set(range(mdp.S))
        assert dest_set.issubset(all_set), (dest_set, mdp.S)
        impossible_set = all_set - dest_set
        for d in impossible_set:
            prior[d] = 0
    prior = _normalize(prior)

    V = backwards_value_iter_fn(mdp, init_state, beta=beta, verbose=verbose)

    D_dest = np.zeros(mdp.S)
    for C in range(mdp.S):
        if prior[C] == 0:
            continue

        goal_val = -V[C] + np.log(prior[C])
        # TODO: This temporary implementation will break if there is more than
        # one possible destination. For more information, review Ziebart.
        D_dest += forwards_value_iter(mdp, C, beta=beta,
                    fixed_init_val=goal_val, verbose=verbose)

    D_dest += V
    return np.exp(D_dest)
