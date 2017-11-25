from __future__ import division
from __future__ import absolute_import

import numpy as np

from mdp.hardmax import action_probabilities

def infer_from_start(mdp, init_state, dest, T=None, verbose=False, beta=1,
        cached_action_prob=None):
    if T is None:
        T = mdp.rows + mdp.cols

    if cached_action_prob is not None:
        action_prob = cached_action_prob
        assert action_prob.shape == (mdp.S, mdp.A)
    else:
        action_prob = action_probabilities(mdp, dest, beta=beta)

    res = np.zeros([T+1, mdp.S])
    res[0][init_state] = 1
    for t in range(1, T+1):
        P = res[t-1]
        P_prime = res[t]
        for s in range(mdp.S):
            if P[s] == 0:
                continue
            for a, s_prime in mdp.neighbors[s]:
                # P_prime += prior[dest] * P[s] * action_prob[dest, s, a]
                P_prime[s_prime] += P[s] * action_prob[s, a]

    D = np.sum(res[1:], axis=0)
    D[dest] = max(1, D[dest])  # Don't want this to grow absurdly large.
    return D

def infer(mdp, traj, dest, T=None, verbose=False, beta=1,
        cached_action_prob=None):

    assert len(traj) > 0
    s_a = traj[0][0]
    s_b = mdp.transition(*traj[-1])
    return infer_from_start(mdp, s_b, dest, T=T, verbose=verbose, beta=beta,
            cached_action_prob=cached_action_prob)

def _main():
    from mdp import GridWorldMDP
    from util import display
    N = 50
    default_reward = -5
    g = GridWorldMDP(N, N, default_reward=default_reward, euclidean_rewards=True)
    for beta in [1, 2, 3, 4]:
        print("Expected occupancies for beta={}").format(beta)
        D = infer_from_start(g, 0, N*N-1, T=N, beta=beta).reshape(N, N)
        print(D)

if __name__ == '__main__':
    _main()
