import numpy as np

from ..mdp import GridWorldMDP
from ..inference.softmax.destination import infer_destination
from ..util import build_traj_from_actions
from ..util.args import unpack_opt_list
from ..parameters import inf_default

from .common import *
from .common import _traj_starter, _occ_starter
from .common_multi import _occ_starter as _occ_starter_multi

Actions = GridWorldMDP.Actions


def _traj_beta_inf_loop(on_loop, g, traj, traj_lens, goal,
        inf_mod=inf_default, guess=1,
        min_beta=0.01, max_beta=100, verbose=True):

    for i in xrange(len(traj) + 1):
        trajs = []
        beta_hats = []
        for traj_len in traj_lens:
            if i == 0:
                start = traj[0][0]
                tr = [(start, Actions.ABSORB)]
                beta_hat = guess
            else:
                tr = traj[:i]
                if len(tr) > traj_len:
                    tr = tr[-traj_len:]
                beta_hat = inf_mod.beta.binary_search(g, tr, goal,
                        guess=beta_hat, verbose=verbose, min_beta=min_beta,
                        max_beta=max_beta)
                if verbose:
                    print "{}: beta_hat={}".format(i+1, beta_hat)
            trajs.append(tr)
            beta_hats.append(beta_hat)
        on_loop(trajs, np.round(beta_hats, 3), i)


def traj_inf(traj_or_traj_mode="diag", mode="diag", N=30,
        R=-1, title=None, inf_mod=inf_default, zmin=-5, zmax=0,
        traj_lens=None, **kwargs):
    # We don't care about model_goal. The star we show is always `goal`.
    g, T, start, goal, _ = _occ_starter(N, R, mode)
    if type(traj_or_traj_mode) is str:
        traj = _traj_starter(N, start, traj_or_traj_mode)
    else:
        traj = traj_or_traj_mode

    beta_hat = beta_fixed = 1

    traj_lens = unpack_opt_list(traj_lens)
    for i, l in enumerate(traj_lens):
        if l is None:
            traj_lens[i] = np.inf

    occ = inf_mod.occupancy
    def on_loop(trajs, beta_hats, t):
        stars_grid = [goal]

        subplot_titles = []
        occ_list = []

        for traj, beta_hat, traj_len in zip(trajs, beta_hats, traj_lens):
            occ_list.append(
                    occ.infer(g, traj, beta_or_betas=beta_hat,
                    T=T, dest_or_dests=goal))
            subplot_titles.append(
                    "<br>traj length limit={}<br>beta_hat={}".format(
                        traj_len, beta_hat))

        _title = title or "effect of forgetting on euclidean expected occupancies"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, trajs, occ_list, subplot_titles, title=_title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, traj_lens, goal)

def _traj_beta_inf_loop_multi(on_loop, g, traj, dest_list, inf_mod=inf_default,
        beta_guesses_grid=None, min_beta=0.01, max_beta=100, traj_len=None,
        verbose=True):
    traj_len = traj_len or np.inf

    for i in xrange(len(traj) + 1):
        if i == 0:
            start = traj[0][0]
            tr = [(start, Actions.ABSORB)]
            infer_from_start = inf_mod.occupancy.infer_from_start
            D, D_dest_list, dest_probs, betas = infer_from_start(
                    g, start, dest_list, verbose=verbose, verbose_return=True)
        else:
            tr = traj[:i]
            if len(tr) > traj_len:
                tr = tr[-traj_len:]
            opt=dict(min_beta=min_beta, max_beta=max_beta)
            infer = inf_mod.occupancy.infer
            D, _, dest_probs, betas = infer(g, tr, dest_list,
                    beta_guesses=beta_guesses, bin_search_opt=opt,
                    verbose=verbose, verbose_return=True)
            D_list.append(D)
            beta_guesses[j] = np.copy(betas)
            if verbose:
                print "dest_probs={}".format(dest_probs)
                print "betas={}".format(betas)

        np.around(dest_probs, 3, out=dest_probs)
        np.around(betas, 3, out=betas)

        on_loop(traj_list, D_list, dest_grid, betas, i)

def traj_inf_multi(traj_mode="diag", mode="diag", N=30,
        R=-1, title=None, inf_mod=inf_default, zmin=-5, zmax=0,
        traj_lens=None, **kwargs):
    g, T, start, dest_list = _occ_starter_multi(N, R, mode)
    traj = _traj_starter(N, start, traj_mode)

    beta_hat = beta_fixed = 1

    traj_lens = unpack_opt_list(traj_lens)
    for i, l in enumerate(traj_lens):
        if l is None:
            traj_lens[i] = np.inf

    occ = inf_mod.occupancy
    def on_loop(traj_list, D_list, dest_grid, betas, t):
        dest_grid = np.round(dest_grid)

        occ_list = list(D_list)
        subplot_titles = []
        stars_grid = []

        for beta_hat, dest_probs in zip(betas, dest_grid):
            subplot_titles.append("dest_probs={}, beta_hat={}".format(
                list(dest_probs), beta_hat))
            stars_grid.append([dest])

        # Weighted average of all heat maps
        occ_list.append(D)
        subplot_titles.append("net occupancy")
        stars_grid.append(dest_list)

        _title = title or "euclid expected occupancies t={t} R={R}"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=_title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, traj_lens, dest_list)
