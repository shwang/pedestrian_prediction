from __future__ import division
import numpy as np
import warnings
from itertools import izip

def _calc_max_update(V, V_prime):
    """
    Use this rather than max(np.absolute(V_prime - V)) because an unchanged value
    of float('-inf') would result in a NaN change, when it should actually result
    in a 0 change.
    """

    assert V.shape == V_prime.shape, (V, V_prime)
    assert float(u'nan') not in V, V
    assert float(u'nan') not in V_prime, V_prime

    max_update = float(u'-inf')
    for v, v_p in izip(V, V_prime):
        if v == v_p == float(u'-inf'):
            # v - v_p would return NaN, but in our simulation, this is actually
            # a zero update.
            max_update = max(max_update, 0)
        else:
            max_update = max(max_update, abs(v - v_p))

    return max_update

def forwards_value_iter(*args, **kwargs):
    kwargs[u'forwards'] = True
    if 'lazy_init_state' not in kwargs:
        kwargs['lazy_init_state'] = True
    # XXX: Does absorbing at goal cause Divergence (TM)?
    # # We should be able to absorb at the goal.
    # if "absorb" not in kwwargs:
    #     kwargs['absorb'] = True
    return _value_iter(*args, **kwargs)

def backwards_value_iter(*args, **kwargs):
    kwargs[u'forwards'] = False
    if 'lazy_init_state' not in kwargs:
        kwargs['lazy_init_state'] = False
    return _value_iter(*args, **kwargs)

def _value_iter(mdp, init_state, update_threshold=1e-8, max_iters=None, nachum=False,
        fixed_init_val=0, beta=1, forwards=False, lazy_init_state=False, absorb=False,
        verbose=False, super_verbose=False, init_vals=None, gamma=1, debug_iters=False):
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
        nachum [bool]: (optional) (experimental) If true, then use Nachum-style Bellman
            updates. Otherwise, use shwang-style Bellman updates.
        fixed_init_val [float]: (optional) Fix the initial state's value at this value.
        beta [float]: (optional) The softmax agent's irrationality constant.
            Beta must be nonnegative. If beta=0, then the softmax agent
            always acts optimally. If beta=float('inf'), then the softmax agent
            acts completely randomly.
        forwards [bool]: (optional) Choose between forwards or backwards value iteration.
            By default, False, indicating backwards value iteration.
        lazy_init_state [bool]: (optional) (debugging) If False, perform a final mini-Bellman
            update on V[init_state] after values have converged.  Otherwise, the
            value of `V[init_state]` will be init_state value.
        absorb [bool]: (optional) If True, then during value iteration, allow the Absorb
            action at the init_state and remove the Absorb action from all other
            states. This operation has no effect on the results if lazy_init_val
            is True.
        verbose [bool]: (optional) If true, then print the result of each iteration.

    Returns:
        V [np.ndarray]: If `states` is not given, a length S array, where the ith
            element is the value of reaching state i starting from init_state. If
            `states` is given, a length `len(states)` array where the ith element
            is the value of reaching state `states[i]` starting from init_state.
    """
    assert beta >= 0, beta
    assert init_state >= 0 and init_state < mdp.S, init_state

    mdp = mdp.copy()

    if init_vals is None:
        V = np.full(mdp.S, float(u'-inf'))
        V[init_state] = 0
        dirty = np.full(mdp.S, False)
        dirty[init_state] = True
        updatable = np.empty(mdp.S, dtype=bool)
    else:
        assert init_vals.shape == (mdp.S,)
        V = init_vals
        dirty = np.full(mdp.S, True)
        updatable = np.empty(mdp.S, dtype=bool)
    if max_iters == None:
        max_iters = float(u'inf')

    # Set up the dirty bitfield. When caching is True, this is used to determine
    # which states to update.

    it = 0
    V_prime = np.full(mdp.S, -np.inf)
    V_prime[init_state] = 0

    # Execute one iteration. Returns True if converged.
    # Modifies dirty, updatable, V, and V_prime.
    #
    # If immutable_init_state is True, then fix V[init_state] at fixed_init_val.
    def _step(immutable_init_state=True):
        if verbose or super_verbose:
            print it, V.reshape(mdp.rows, mdp.cols)

        # If a state is dirty or has dirty neighbours, then it is updatable.
        updatable[:] = dirty
        for s_prime, dirt in enumerate(dirty):
            if not dirt:
                continue
            for a in xrange(mdp.A):
                s = mdp.transition(s_prime, a)
                updatable[s] = True

        # Set updateable values to 0 so we can actually update later.
        # Leave other values as is.
        for s, flag in enumerate(updatable):
            if flag:
                V_prime[s] = 0

        if super_verbose:
            print it, updatable.reshape(mdp.rows, mdp.cols)

        temp = np.empty(mdp.S)
        exp_max = np.zeros(mdp.S)  # log-sum-exp trick to prevent exp overflow.
        if forwards:
            for s in xrange(mdp.S):
                if not updatable[s]:
                    continue

                # XXX: optimization: cache this length
                N = len(mdp.neighbors[s])
                exp_max[s] = -np.inf

                for i, (a, s_prime) in enumerate(mdp.neighbors[s]):
                    if not nachum:
                        temp[i] = mdp.rewards[s, a]/beta + gamma*V[s_prime]
                    else:
                        temp[i] = mdp.rewards[s, a]/beta + gamma*V[s_prime]/beta
                    if temp[i] == -np.inf:
                        continue
                    if temp[i] > exp_max[s]:
                        exp_max[s] = temp[i]

                before_exp = temp[:N]
                if exp_max[s] > -np.inf:
                    before_exp -= exp_max[s]
                V_prime[s] = sum(np.exp(before_exp))
        else:
            for s_prime in xrange(mdp.S):
                if not updatable[s_prime]:
                    continue
                N = len(mdp.reverse_neighbors[s_prime])
                exp_max[s_prime] = -np.inf

                for i, (a, s) in enumerate(mdp.reverse_neighbors[s_prime]):
                    if not nachum:
                        temp[i] = mdp.rewards[s, a]/beta + gamma*V[s]
                    else:
                        temp[i] = mdp.rewards[s, a]/beta + gamma*V[s]/beta

                    if temp[i] == -np.inf:
                        continue
                    if temp[i] > exp_max[s_prime]:
                        exp_max[s_prime] = temp[i]

                before_exp = temp[:N]
                if exp_max[s_prime] > -np.inf:
                    before_exp -= exp_max[s_prime]
                V_prime[s_prime] = sum(np.exp(before_exp))

        # This warning will appear when taking the log of float(-inf) in V_prime.
        warnings.filterwarnings(u"ignore", u"divide by zero encountered in log")
        np.log(V_prime, out=V_prime, where=updatable)
        np.add(V_prime, exp_max, out=V_prime, where=updatable)
        warnings.resetwarnings()

        if nachum:
            # XXX: Hmmmm... I wonder how beta should interact with V_prime[init_state]
            np.multiply(V_prime, beta, out=V_prime, where=updatable)

        if immutable_init_state:
            V_prime[init_state] = fixed_init_val

        max_update = _calc_max_update(V, V_prime)
        if super_verbose:
            print u"max_update", max_update
        if max_update < update_threshold:
            return True

        # Various warnings for subtracting -inf from -inf and processing the
        # resulting nan.
        warnings.filterwarnings(u"ignore", u"invalid value encountered in abs")
        warnings.filterwarnings(u"ignore", u"invalid value encountered in subtract")
        warnings.filterwarnings(u"ignore", u"invalid value encountered in greater")

        # XXX: This can be optimized slightly. By storing subtract result and abs
        # result in the same array.
        # If a state updates by more than update_threshold, then it is dirty.
        np.greater(np.abs(V_prime - V), update_threshold, out=dirty)

        warnings.filterwarnings(u"ignore", u"invalid value encountered in abs")
        warnings.filterwarnings(u"ignore", u"invalid value encountered in subtract")
        warnings.filterwarnings(u"ignore", u"divide by zero encountered in greater")

        V[:] = V_prime

        return False

    done = False
    while not done and it < max_iters:
        done = _step()
        it += 1

    # Evaluate V[init_val]
    if not lazy_init_state:
        dirty.fill(False)
        dirty[init_state] = True
        _step(False)

        if verbose or super_verbose:
            print "*", V.reshape(mdp.rows, mdp.cols)

    if debug_iters:
        return V, it
    return V
