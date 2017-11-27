import numpy as np

from ..mdp import GridWorldMDP
from ..inference.softmax.destination import infer_destination
from ..inference import hardmax

from ..util import sum_rewards, display, build_traj_from_actions

from ..parameters import inf_default

from .common import *

def simple_ground_truth_inf(mode="diag", N=30, R=-6, true_beta=5,
        zmin=-5, zmax=0, inf_mod=inf_default, title=None, **kwargs):
    g, T, start, goal, model_goal = _occ_starter(N, R, mode)

    traj = simulate(g, start, goal, beta=true_beta)
    beta_fixed = 1
    beta_hat = 1

    occ = inf_mod.occupancy
    def on_loop(traj, beta_hat, t):
        occupancies = occ.infer(g, traj, beta=beta_hat, T=T, dest=model_goal)
        fixed_occupancies = occ.infer(g, traj, beta=beta_fixed, T=T, dest=model_goal)
        true_occupancies = occ.infer(g, traj, beta=true_beta, T=T, dest=goal)

        occ_list = [occupancies, fixed_occupancies, true_occupancies]
        stars_grid = [[model_goal], [model_goal], [goal]]

        subplot_titles = (
                    "beta_hat={}".format(beta_hat),
                    "beta={}".format(beta_fixed),
                    "ground truth beta={}".format(true_beta),
                    )
        _title = title or "Euclid expected occupancies R={R}" + \
                " (for trajectories of length {T}) <br>t={t}"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, goal)
