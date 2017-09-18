import numpy as np
import warnings

def backwards_value_iter(mdp, init_state, iters=None):
    """
    Calculate the value of reaching all states, starting from a given initial state.

    Params:
        S [int]: The number of states.
        A [int]: The number of actions.
        init_state [int]: A starting state, whose initial value will be set to 0. All other
            states will be initialized with value float('-inf').
        rewards [np.ndarray]: a SxA array where rewards[s, a] is the reward
            received from taking action a at state s.
        transition [function]: The state transition function for the deterministic MDP.
            transition(s, a) returns the state that results from taking action a at state s.
        iters [int]: (optional) The number of value iterations to perform. By default,
            iters will be set to S.

    Returns:
        value [np.ndarray]: A length S array, where the ith element is the value of
            reaching state i starting from init_state.
    """
    assert init_state >= 0 and init_state < mdp.S, init_state
    V = np.array([float('-inf')] * mdp.S)
    V[init_state] = 0
    if iters == None:
        iters = mdp.S

    rewards_by_state = np.zeros(mdp.S)
    for i in range(iters):
        V_prime = np.zeros(mdp.S)
        for s_prime in range(mdp.S):
            for a in range(mdp.A):
                s = mdp.transition(s_prime, a)
                V_prime[s] += np.exp(mdp.rewards[s_prime, a] + V[s_prime])

        warnings.filterwarnings("ignore", "divide by zero encountered in log")
        V = np.log(V_prime)
        warnings.resetwarnings()

    return V
