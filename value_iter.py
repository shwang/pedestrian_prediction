import numpy as np
import queue
import warnings

def _calc_max_update(V, V_prime):
    """
    Use this rather than max(np.absolute(V_prime - V)) because an unchanged value
    of float('-inf') would result in a NaN change, when it should actually result
    in a 0 change.
    """

    assert V.shape == V_prime.shape, (V, V_prime)
    assert float('nan') not in V, V
    assert float('nan') not in V_prime, V_prime

    max_update = float('-inf')
    for v, v_p in zip(V, V_prime):
        if v == v_p == float('-inf'):
            # v - v_p would return NaN, but in our simulation, this is actually
            # a zero update.
            max_update = max(max_update, 0)
        else:
            max_update = max(max_update, abs(v - v_p))

    return max_update

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
    pq = queue.PriorityQueue()
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

def forwards_value_iter(mdp, goal_state, update_threshold=1e-7, max_iters=None,
        fixed_goal=True, fixed_goal_val=0, beta=1, verbose=False):
    """
    Approximate the softmax value of various initial states, given the goal state.

    Params:
        mdp [GridWorldMDP]: The MDP.
        goal_state [int]: A goal state, the only state in which the Absorb action is legal
            and the initial value is 0. All other states start with initial value of
            -inf.
        update_threshold [float]: (optional) When the magnitude of all value updates is
            less than update_threshold, value iteration will return its approximate solution.
        max_iters [int]: (optional) An upper bound on the number of value iterations to
            perform. If this upper bound is reached, then iteration will cease regardless
            of whether `update_threshold`'s condition is met.
        fixed_goal [bool]: (optional) Fix goal_state's value at `fixed_goal_val`.
        fixed_goal_val [float]: (optional) If `fixed_goal` is True, then fix the
            goal_state's value at this value.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        verbose [bool]: (optional) If true, then print the result of each iteration.

    Returns:
        value [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """
    assert beta >= 0, beta
    assert goal_state >= 0 and goal_state < mdp.S, goal_state

    V = np.full(mdp.S, float('-inf'))
    V[goal_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    # Reconfigure rewards to allow ABSORB action at goal_state.
    mdp = mdp.copy()
    mdp.set_goal(goal_state)

    max_update = float('inf')
    it = 0
    while max_update > update_threshold and it < max_iters:
        if verbose:
            print(it, V.reshape(mdp.rows, mdp.cols))
        V_prime = np.zeros(mdp.S)
        for s in range(mdp.S):
            for a in range(mdp.A):
                if fixed_goal and s == goal_state:
                    continue
                s_prime = mdp.transition(s, a)
                V_prime[s] += np.exp(mdp.rewards[s, a]/beta + V[s_prime])

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        V_prime = np.log(V_prime)
        warnings.resetwarnings()

        if fixed_goal:
            V_prime[goal_state] = fixed_goal_val

        max_update = _calc_max_update(V, V_prime)
        it += 1
        V = V_prime

    return V

# TODO: get rid of this goal_state, fixed_init stuff. That should always be true.
def backwards_value_iter(mdp, init_state, goal_state=None, update_threshold=1e-8, max_iters=None,
        fixed_init=True, fixed_init_val=0, beta=1, verbose=False):
    """
    Approximate the softmax value of reaching various destination states, starting
    from a given initial state.

    Params:
        mdp [GridWorldMDP]: The MDP to run backwards_value_iter in. Maybe I should change this
            make backwards_value_iter part of the GridWorldMDP class... anyways the
            coupling is kind of uncomfortable right now.
        init_state [int]: A starting state, whose initial value will be set to 0. All other
            states will be initialized with value float('-inf').
        goal_state [int]: A goal state, the only state in which the Absorb action is legal.
                            Or provide -1 to allow Absorb at every state.
                            Or provide None to allow Absorb at no state.
                            [!!! Experimentally, we have found that the default values:
                                goal_state=None, fixed_init=True, and fixed_init_val=0
                                work best.]
        update_threshold [float]: (optional) When the magnitude of all value updates is
            less than update_threshold, value iteration will return its approximate solution.
        max_iters [int]: (optional) An upper bound on the number of value iterations to
            perform. If this upper bound is reached, then iteration will cease regardless
            of whether `update_threshold`'s condition is met.
        fixed_init [bool]: (optional) Fix initial state's value at `fixed_init_val`.
        fixed_init_val [float]: (optional) If `fixed_init` is True, then fix the initial
            state's value at this value.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        verbose [bool]: (optional) If true, then print the result of each iteration.

    Returns:
        value [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """
    assert beta >= 0, beta
    assert init_state >= 0 and init_state < mdp.S, init_state
    assert goal_state == None or (goal_state >= -1 and goal_state < mdp.S), goal_state

    V = np.full(mdp.S, float('-inf'))
    V[init_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    # Reconfigure rewards to allow ABSORB action at goal_state.
    mdp = mdp.copy()
    if goal_state == -1:
        mdp.set_all_goals()
    else:
        mdp.set_goal(goal_state)

    # Set up the dirty bitfield. When caching is True, this is used to determine
    # which states to update.
    dirty = np.full(mdp.S, False)
    dirty[init_state] = True
    updatable = np.empty(mdp.S, dtype=bool)

    max_update = np.inf
    it = 0
    V_prime = np.full(mdp.S, -np.inf)
    V_prime[init_state] = 0

    while it < max_iters:
        if verbose:
            print(it, V.reshape(mdp.rows, mdp.cols))

        # If a state is dirty or has dirty neighbours, then it is updatable.
        updatable[:] = dirty
        for s_prime, dirt in enumerate(dirty):
            if not dirt:
                continue
            for a in range(mdp.A):
                s = mdp.transition(s_prime, a)
                updatable[s] = True

        # Set updateable values to 0 so we can actually update later.
        # Leave other values as is.
        for s, flag in enumerate(updatable):
            if flag:
                V_prime[s] = 0

        if verbose:
            print(it, updatable.reshape(mdp.rows, mdp.cols))

        for s_prime in range(mdp.S):
            for a in range(mdp.A):
                s = mdp.transition(s_prime, a)
                if not updatable[s]:
                    continue
                if fixed_init and s == init_state:
                    continue
                V_prime[s] += np.exp(mdp.rewards[s_prime, a]/beta + V[s_prime])

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        np.log(V_prime, out=V_prime, where=updatable)
        warnings.resetwarnings()

        if fixed_init:
            V_prime[init_state] = fixed_init_val

        max_update = _calc_max_update(V, V_prime)
        if verbose:
            print("max_update", max_update)
        if max_update < update_threshold:
            break

        # Various warnings for subtracting -inf from -inf and processing the
        # resulting nan.
        warnings.filterwarnings("ignore", "invalid value encountered in abs")
        warnings.filterwarnings("ignore", "invalid value encountered in subtract")
        warnings.filterwarnings("ignore", "invalid value encountered in greater")

        # XXX: This can be optimized slightly. By storing subtract result and abs
        # result in the same array.
        # If a state updates by more than update_threshold, then it is dirty.
        np.greater(np.abs(V_prime - V), update_threshold, out=dirty)

        warnings.filterwarnings("ignore", "invalid value encountered in abs")
        warnings.filterwarnings("ignore", "invalid value encountered in subtract")
        warnings.filterwarnings("ignore", "divide by zero encountered in greater")

        it += 1
        V[:] = V_prime

    return V
