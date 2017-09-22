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

def backwards_value_iter(mdp, init_state, states=None, update_threshold=1e-7, max_iters=None,
        softmax=True):
    """
    Approximate the softmax value of reaching various destination states, starting
    from a given initial state.

    Params:
        S [int]: The number of states.
        A [int]: The number of actions.
        init_state [int]: A starting state, whose initial value will be set to 0. All other
            states will be initialized with value float('-inf').
        rewards [np.ndarray]: a SxA array where rewards[s, a] is the reward
            received from taking action a at state s.
        transition [function]: The state transition function for the deterministic MDP.
            transition(s, a) returns the state that results from taking action a at state s.
        states [list]: (optional) An ordered list of destination states to calculate
            value for. By default, calculate the value of all states.
        update_threshold [float]: (optional) When the magnitude of all value updates is
            less than update_threshold, value iteration will return its approximate solution.
        max_iters [int]: (optional) An upper bound on the number of value iterations to
            perform. If this upper bound is reached, then iteration will cease regardless
            of whether `update_threshold`'s condition is met.

    Returns:
        value [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """
    assert init_state >= 0 and init_state < mdp.S, init_state
    V = np.array([float('-inf')] * mdp.S)
    V[init_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    max_update = float('inf')
    it = 0
    while max_update > update_threshold and it < max_iters:

        if softmax:
            V_prime = np.zeros(mdp.S)
            for s_prime in range(mdp.S):
                for a in range(mdp.A):
                    s = mdp.transition(s_prime, a)
                    V_prime[s] += np.exp(mdp.rewards[s_prime, a] + V[s_prime])

            warnings.filterwarnings("ignore", "divide by zero encountered in log")
            V_prime = np.log(V_prime)
            warnings.resetwarnings()
        else:
            V_prime = np.array([float('-inf')] * mdp.S)
            for s_prime in range(mdp.S):
                for a in range(mdp.A):
                    s = mdp.transition(s_prime, a)
                    V_prime[s] = max(V_prime[s], mdp.rewards[s_prime, a] + V[s_prime])

        max_update = _calc_max_update(V, V_prime)
        it += 1
        V = V_prime

    return V
