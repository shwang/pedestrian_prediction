import numpy as np

from ..mdp import GridWorldMDP
from ..util import sum_rewards, display, build_traj_from_actions
from ..parameters import inf_default

from .common import *

def _occ_starter(N, R, mode):
    g = GridWorldMDP(N, N, {}, default_reward=R)
    T = N+N

    diag = g.S - 1
    diag_top = g.coor_to_state(N//2, N-1)
    bot = g.coor_to_state(N-1, 0)
    top = g.coor_to_state(0, N-1)
    mid = g.coor_to_state(N//2, N//2)

    if mode == "diag+bot":
        start = 0
        dest_list = [diag, bot]
    elif mode == "diag+diag_top":
        start = 0
        dest_list = [diag, diag_top]
    elif mode == "nondiag":
        start = 0
        dest_list = [bot, top]
    elif mode == "diag+mid":
        start = 0
        dest_list = [mid, diag]
    elif mode == "tri":
        start = 0
        dest_list = [top, bot, diag]
    else:
        raise Exception("invalid mode: {}".format(mode))

    return g, T, start, dest_list


def _traj_starter(N, init_state, mode):
    A = Actions
    g = GridWorldMDP(N, N)
    if mode == "diag":
        start = 0
        actions = [A.UP_RIGHT] * (N-1)
    elif mode == "horizontal":
        start = g.coor_to_state(0, N//2)
        actions = [A.RIGHT] * (N-1)
    elif mode == "horizontal_origin":
        start = 0
        actions = [A.RIGHT] * (N-1)
    elif mode == "vertical":
        start = g.coor_to_state(N//2, 0)
        actions = [A.UP] * (N-1)
    elif mode == "diag-crawl":
        start = 0
        actions = [A.RIGHT] * (N-1) + [A.UP] * (N-1)
    elif mode == "diag-fickle":
        start = 0
        w = (N-1)//2
        W = N - 1 - w
        actions = [A.RIGHT] * w + [A.UP_RIGHT] * W + \
                [A.UP] * w
    elif mode == "diag-fickle2":
        start = 0
        w = (N-1)//2
        W = N - 1 - w
        actions = [A.UP_RIGHT] * W + [A.DOWN_RIGHT] * w \
                + [A.DOWN]
    else:
        raise Exception("invalid mode: {}".format(mode))
    actions += [A.ABSORB]
    return build_traj_from_actions(g, start, actions)


def _traj_beta_inf_loop(on_loop, g, traj, dest_list, inf_mod=inf_default,
        beta_guesses=None, min_beta=0.01, max_beta=100, verbose=True):
    # This choice of range implicitly ignores the final action (ABSORB)
    # TODO: let's change the semantics -- don't ignore the final action.
    for i in xrange(len(traj)):
        if i == 0:
            start = traj[0][0]
            tr = [(start, Actions.ABSORB)]
            infer_from_start = inf_mod.occupancy.infer_from_start
            D, D_dest_list, dest_probs, betas = infer_from_start(
                    g, start, dest_list, verbose=verbose, verbose_return=True)
        else:
            tr = traj[:i]
            opt=dict(min_beta=min_beta, max_beta=max_beta)
            infer = inf_mod.occupancy.infer
            D, D_dest_list, dest_probs, betas = infer(g, tr, dest_list,
                    beta_guesses=beta_guesses, bin_search_opt=opt,
                    verbose=verbose, verbose_return=True)
            beta_guesses = np.copy(betas)
            if verbose:
                print "dest_probs={}".format(dest_probs)
                print "betas={}".format(betas)

        np.around(dest_probs, 3, out=dest_probs)
        np.around(betas, 3, out=betas)

        on_loop(tr, D, D_dest_list, dest_probs, betas, i)


def multidest_traj_inf(traj_or_traj_mode="diag", mode="diag", N=30, R=-6,
        title=None, inf_mod=inf_default, zmin=-5, zmax=0, **kwargs):
    g, T, start, dest_list = _occ_starter(N, R, mode)

    # TODO: Use util.unpack once that is available.
    if type(traj_or_traj_mode) is str:
        traj = _traj_starter(N, start, traj_or_traj_mode)
    else:
        traj = traj_or_traj_mode

    beta_hat = beta_fixed = 1

    occ = inf_mod.occupancy
    def on_loop(traj, D, D_dest_list, dest_probs, betas, t):
        occ_list = list(D_dest_list)
        subplot_titles = []
        stars_grid = []

        for dest, dest_prob, beta_hat in zip(
                dest_list, dest_probs, betas):
            subplot_titles.append("dest_prob={}, beta_hat={}".format(
                dest_prob, beta_hat))
            stars_grid.append([dest])

        # Weighted average of all heat maps
        occ_list.append(D)
        subplot_titles.append("net occupancy")
        stars_grid.append(dest_list)

        _title = title or "euclid expected occupancies t={t} R={R}"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=_title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, dest_list)
