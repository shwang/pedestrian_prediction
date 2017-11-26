from __future__ import division

import numpy as np
import destination

from ...parameters import val_default

# TODO: Figure out the imports nonsense and move this to util.
#
# don't use from notation. Maybe consider using sys.modules.
def unpack_opt_list(iter_or_scalar, require_not_empty=True, extend_to=1):
    """
    Converts the iter to a list, or the scalar to a length-1 list.
    """
    try:
        iter(iter_or_scalar)
        res = list(iter_or_scalar)
        if require_not_empty:
            assert len(res) > 0, res
    except TypeError:
        res = [iter_or_scalar] * extend_to
    return res


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
    If verbose_return: returns D, D_dest_list, dest_probs, beta_list
    else: returns D
    """
    if T is None:
        T = g.rows + g.cols

    # TODO: use some variable instead of len(dest_list)
    # Convert scalars arguments into single-element lists, or keep list.
    dest_list = unpack_opt_list(dest_or_dests)
    beta_list = unpack_opt_list(beta_or_betas, extend_to=len(dest_list))
    beta_list = np.array(beta_list)
    assert len(beta_list) == len(dest_list), beta_list

    # Unpack cached_action_prob, if applicable.
    if cached_action_probs is not None:
        act_prob_list = unpack_opt_list(cached_action_probs)
    else:
        act_prob_list = [None] * len(dest_list)
    assert len(act_prob_list) == len(dest_list), act_prob_list

    # Unpack dest_prob
    # TODO: incongrous dest_prob name
    if dest_probs is None:
        dest_probs = [1] * len(dest_list)
    assert len(dest_probs) == len(dest_list)
    dest_probs = np.array(dest_probs) / sum(dest_probs)

    # Take the weighted sum of the occupancy given each individual destination.
    D = np.zeros(g.S)
    D_dest_list = []  # Only for verbose_return
    # TODO: just use a big zip(..)
    for i, dest in enumerate(dest_list):
        D_dest = _infer(g, init_state, dest, T, beta=beta_list[i],
                action_prob=act_prob_list[i])
        D_dest_list.append(np.copy(D_dest))
        np.multiply(D_dest, dest_probs[i], out=D_dest)
        np.add(D, D_dest, out=D)

    if not verbose_return:
        return D
    else:
        return D, D_dest_list, dest_probs, beta_list


# TODO: bin_search_opt is sketchy
def infer(g, traj, dest_or_dests, T=None, verbose=False, beta_or_betas=None,
        auto_beta=True, beta_guesses=None, bin_search_opt={}, **kwargs):
    """
    If auto_beta, then use MLE beta. Otherwise, use given betas (beta_or_betas).
    """
    assert len(traj) > 0, traj
    s_b = g.transition(*traj[-1])
    dest_list = unpack_opt_list(dest_or_dests)
    if not auto_beta:
        betas = unpack_opt_list(beta_or_betas)
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
