from __future__ import division

import numpy as np
import destination

from ...parameters import val_default
from ...util.args import unpack_opt_list


def _infer(g, init_state, dest, T, beta=1, action_prob=None, val_mod=val_default):
    if action_prob is None:
        action_prob = val_mod.action_probabilities(g, dest, beta=beta)

    res = np.zeros([T+1, g.S])
    res[0][init_state] = 1
    for t in range(1, T+1):
        P = res[t-1]
        P_prime = res[t]
        for s in range(g.S):
            if P[s] == 0:
                continue
            for a, s_prime in g.neighbors[s]:
                P_prime[s_prime] += P[s] * action_prob[s, a]

    D = np.sum(res[1:], axis=0)
    D[dest] = 1  # This value is fixed.
    return D


def infer_from_start(g, init_state, dest_or_dests, dest_probs=None,
        T=None, verbose=False, beta_or_betas=1, cached_action_probs=None,
        verbose_return=False):
    """
    If verbose_return: returns D, D_dests, dest_probs, betas
    else: returns D
    """
    if T is None:
        T = g.rows + g.cols

    dests = unpack_opt_list(dest_or_dests)
    L = len(dests)

    betas = np.array(unpack_opt_list(beta_or_betas, extend_to=L))
    assert len(betas) == L, betas

    # Unpack cached_action_probs, if applicable.
    if cached_action_probs is not None:
        act_probs = unpack_opt_list(cached_action_probs)
    else:
        act_probs = [None] * L
    assert len(act_probs) == L, act_prob_list

    # Unpack dest_prob
    if dest_probs is None:
        dest_probs = [1] * L
    assert len(dest_probs) == L
    dest_probs = np.array(dest_probs) / sum(dest_probs)

    # Take the weighted sum of the occupancy given each individual destination.
    D = np.zeros(g.S)
    D_dests = []  # Only for verbose_return
    for dest, beta, act_prob, dest_prob in zip(
            dests, betas, act_probs, dest_probs):
        D_dest = _infer(g, init_state, dest, T, beta=beta,
                action_prob=act_prob)
        D_dests.append(np.copy(D_dest))
        np.multiply(D_dest, dest_prob, out=D_dest)
        np.add(D, D_dest, out=D)

    if not verbose_return:
        return D
    else:
        return D, D_dests, dest_probs, betas


# TODO: bin_search_opt is sketchy
def infer(g, traj, dest_or_dests, T=None, verbose=False, beta_or_betas=None,
        auto_beta=True, beta_guesses=None, bin_search_opt={}, **kwargs):
    """
    If beta_or_betas not provided, then use MLE beta.
    """
    assert len(traj) > 0, traj
    s_b = g.transition(*traj[-1])
    dest_list = unpack_opt_list(dest_or_dests)
    if beta_or_betas is not None:
        betas = unpack_opt_list(beta_or_betas)
        dest_probs = None
    else:
        dest_probs, betas = destination.infer(g, traj, dest_list,
                beta_guesses=beta_guesses, **bin_search_opt)

    return infer_from_start(g, s_b, dest_list, dest_probs=dest_probs,
            T=T, verbose=verbose, beta_or_betas=betas, **kwargs)

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
