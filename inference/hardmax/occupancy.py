from __future__ import division
from __future__ import absolute_import

import numpy as np

from mdp.hardmax import action_probabilities

def infer_from_start(mdp, init_state, dest, T=5, verbose=False, beta=1,
        cached_action_prob=None, all_steps=False, combine=False):
    # if prior is None:
    #     prior = np.ones(mdp.S) / mdp.S
    # if dest_set is not None:
    #     for s in range(mdp.S):
    #         if s not in dest_set:
    #             prior[s] = 0
    #     prior /= sum(prior)

    if cached_action_prob is not None:
        action_prob = cached_action_prob
        assert action_prob.shape == (mdp.S, mdp.A)
        # for v in action_prob.values():
        #     assert action_prob.shape[:1] == (mdp.S, mdp.A)
    else:
        # for dest in range(mdp.S):
        #     if prior[dest] == 0:
        #         continue
        #     action_prob = {}
        #     action_prob[dest] = action_probabilities(mdp, dest)
        action_prob = action_probabilities(mdp, dest, beta=beta)

    res = np.zeros([T+1, mdp.S])
    res[0][init_state] = 1
    for t in range(1, T+1):
        P = res[t-1]
        P_prime = res[t]
        # import pdb; pdb.set_trace()
        # TODO: loop over dest
        for s in range(mdp.S):
            if P[s] == 0:
                continue
            for a, s_prime in mdp.neighbors[s]:
                # P_prime += prior[dest] * P[s] * action_prob[dest, s, a]
                P_prime[s_prime] += P[s] * action_prob[s, a]

    if all_steps:
        return res
    elif combine:
        return np.sum(res[1:], axis=0)/T
    else:
        return res[T]
