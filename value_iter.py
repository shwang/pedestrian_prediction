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
def forwards_value_iter(*args, **kwargs):
    kwargs['forwards'] = True
    return _value_iter(*args, **kwargs)

def backwards_value_iter(*args, **kwargs):
    kwargs['forwards'] = False
    return _value_iter(*args, **kwargs)

def _value_iter(mdp, init_state, update_threshold=1e-8, max_iters=None,
        fixed_init_val=0, beta=1, forwards=False, verbose=False, super_verbose=False):
    """
    Approximate the softmax value of reaching various destination states, starting
    from a given initial state.

    Params:
        mdp [GridWorldMDP]: The MDP to run backwards_value_iter in. Maybe I should change this
            make backwards_value_iter part of the GridWorldMDP class... anyways the
            coupling is kind of uncomfortable right now.
        init_state [int]: A starting state, whose initial value will be set to 0. All other
            states will be initialized with value float('-inf').
        update_threshold [float]: (optional) When the magnitude of all value updates is
            less than update_threshold, value iteration will return its approximate solution.
        max_iters [int]: (optional) An upper bound on the number of value iterations to
            perform. If this upper bound is reached, then iteration will cease regardless
            of whether `update_threshold`'s condition is met.
        fixed_init_val [float]: (optional) Fix the initial state's value at this value.
        forwards [bool]: (optional) Choose between forwards or backwards value iteration.
            By default, False, indicating backwards value iteration.
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

    V = np.full(mdp.S, float('-inf'))
    V[init_state] = 0
    if max_iters == None:
        max_iters = float('inf')

    mdp = mdp.copy()

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
        if verbose or super_verbose:
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

        if super_verbose:
            print(it, updatable.reshape(mdp.rows, mdp.cols))

        temp = np.empty(mdp.S)
        exp_max = np.zeros(mdp.S)
        if forwards:
            for s in range(mdp.S):
                if not updatable[s]:
                    continue

                # XXX: optimization: cache this length
                N = len(mdp.neighbors[s])
                exp_max[s] = -np.inf

                for i, (a, s_prime) in enumerate(mdp.neighbors[s]):
                    temp[i] = mdp.rewards[s, a]/beta + V[s_prime]
                    if temp[i] == -np.inf:
                        continue
                    if temp[i] > exp_max[s]:
                        exp_max[s] = temp[i]

                before_exp = temp[:N]
                if exp_max[s] > -np.inf:
                    before_exp -= exp_max[s]
                V_prime[s] = sum(np.exp(before_exp))
        else:
            for s_prime in range(mdp.S):
                if not updatable[s_prime]:
                    continue
                N = len(mdp.reverse_neighbors[s_prime])
                exp_max[s_prime] = -np.inf

                for i, (a, s) in enumerate(mdp.reverse_neighbors[s_prime]):
                    temp[i] = mdp.rewards[s, a]/beta + V[s]
                    if temp[i] == -np.inf:
                        continue
                    if temp[i] > exp_max[s_prime]:
                        exp_max[s_prime] = temp[i]

                before_exp = temp[:N]
                if exp_max[s_prime] > -np.inf:
                    before_exp -= exp_max[s_prime]
                V_prime[s_prime] = sum(np.exp(before_exp))

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        np.log(V_prime, out=V_prime, where=updatable)
        np.add(V_prime, exp_max, out=V_prime, where=updatable)
        warnings.resetwarnings()

        V_prime[init_state] = fixed_init_val

        max_update = _calc_max_update(V, V_prime)
        if super_verbose:
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
