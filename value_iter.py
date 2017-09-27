import numpy as np
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

def forwards_value_iter(mdp, init_state, goal_state, update_threshold=1e-7, max_iters=None,
        fixed_goal=False):
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
        fixed_goal [bool]: (experimental) Fix goal_state's value at 0.

    Returns:
        value [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """

    assert goal_state >= 0 and goal_state < mdp.S, goal_state
    V = np.array([float('-inf')] * mdp.S)
    V[goal_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    # Reconfigure rewards to allow ABSORB action at goal_state.
    mdp = mdp.copy()
    mdp.set_goal(goal_state)

    max_update = float('inf')
    it = 0
    while max_update > update_threshold and it < max_iters:
        V_prime = np.zeros(mdp.S)
        for s in range(mdp.S):
            for a in range(mdp.A):
                if fixed_goal and s == goal_state:
                    continue
                s_prime = mdp.transition(s, a)
                V_prime[s] += np.exp(mdp.rewards[s, a] + V[s_prime])

        if fixed_goal:
            V_prime[goal_state] = 1  # becomes 0 after log

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        V_prime = np.log(V_prime)
        warnings.resetwarnings()


        max_update = _calc_max_update(V, V_prime)
        it += 1
        V = V_prime

    return V

def backwards_value_iter(mdp, init_state, goal_state, update_threshold=1e-7, max_iters=None,
        fixed_init=False):
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
        update_threshold [float]: (optional) When the magnitude of all value updates is
            less than update_threshold, value iteration will return its approximate solution.
        max_iters [int]: (optional) An upper bound on the number of value iterations to
            perform. If this upper bound is reached, then iteration will cease regardless
            of whether `update_threshold`'s condition is met.
        fixed_init [bool]: (experimental) Fix initial state's value at 0.

    Returns:
        value [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """
    assert init_state >= 0 and init_state < mdp.S, init_state
    assert goal_state == None or (goal_state >= -1 and goal_state < mdp.S), goal_state
    V = np.array([float('-inf')] * mdp.S)
    V[init_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    # Reconfigure rewards to allow ABSORB action at goal_state.
    mdp = mdp.copy()
    if goal_state == -1:
        mdp.set_all_goals()
    else:
        mdp.set_goal(goal_state)

    max_update = float('inf')
    it = 0
    while max_update > update_threshold and it < max_iters:
        V_prime = np.zeros(mdp.S)
        for s_prime in range(mdp.S):
            for a in range(mdp.A):
                s = mdp.transition(s_prime, a)
                if fixed_init and s == init_state:
                    continue
                V_prime[s] += np.exp(mdp.rewards[s_prime, a] + V[s_prime])

        if fixed_init:
            V_prime[init_state] = 1  # becomes 0 after log

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        V_prime = np.log(V_prime)
        warnings.resetwarnings()


        max_update = _calc_max_update(V, V_prime)
        it += 1
        V = V_prime

    return V
