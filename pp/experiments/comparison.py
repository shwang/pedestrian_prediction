from __future__ import division

import math
import numpy as np
from ..plot import common as plot
from scenarios import scenario_starter

from ..util.hardmax.simulate import simulate
from ..util.args import unpack_opt_list
from ..util import display
from .robot import compare2
from .simulate import simulate_vanilla
from .context import HRContext


def obvious_experiment(ctx, beta_stars, max_trials, min_trials=10, k=None,
        fixed_betas=1, do_plot=True, verbose=True, blob=None, blob2=None,
        plot=False):
    """ Output: list of ExResult """
    beta_stars = unpack_opt_list(beta_stars)
    fixed_betas = unpack_opt_list(fixed_betas)

    # `blob` is a dictionary that contains result data in case the experiment
    # crashes midway.
    if blob is None:
        blob = {}
    for fb in fixed_betas:
        blob[fb] = []

    est_ignore=False
    for fixed_beta in fixed_betas:
        results = blob[fixed_beta]
        for beta_star in beta_stars:
            res = mega_trial(beta_star=beta_star, ctx=ctx,
                    fixed_beta=fixed_beta,
                    min_trials=min_trials, est_ignore=est_ignore,
                    max_trials=max_trials, verbose=verbose, plot=plot, k=k)
            results.append(res)

        if do_plot:
            basic_plot(ctx, results, k=k, fixed_beta=fixed_beta)
        est_ignore=True
    return blob

def obvious_experiment_bayes():
    """ Same as obvious experiment, except we compare MLE vs Bayes """
    # `blob` is a dictionary that contains result data in case the experiment
    # crashes midway.
    if blob is None:
        blob = {}
    for fb in fixed_betas:
        blob[fb] = []

    est_ignore=False
    for fixed_beta in fixed_betas:
        results = blob[fixed_beta]
        for beta_star in beta_stars:
            res = mega_trial(beta_star=beta_star, ctx=ctx,
                    fixed_beta=fixed_beta,
                    min_trials=min_trials, est_ignore=est_ignore,
                    max_trials=max_trials, verbose=verbose, plot=plot, k=k)
            results.append(res)

        if do_plot:
            basic_plot(ctx, results, k=k, fixed_beta=fixed_beta)
        est_ignore=True
    return blob


def basic_plot(ctx, ex_results, title=None, save_png=False,
        k=None, fixed_beta="unknown"):
    # Really, this k should go in the second title.
    title = title or ("Estimating-/Fixed-beta Robot Navigating w/ Boltzmann-" +\
            "Rational Human<br>k={k} grid_size={gs} fixed_beta={fb}"
            ).format(mode=ctx.mode, gs=ctx.N, k=k, fb=fixed_beta)
    trace_fixed = dict(x=[], y=[], error_y=dict(visible=True, array=[]),
            name="fixed beta", mode="markers")
    trace_est = dict(x=[], y=[], error_y=dict(visible=True, array=[]),
            name="MLE beta", mode="markers")
    for res in ex_results:
        x = res.beta_star
        y_fixed = np.mean(res.reward_fixed)
        se_fixed = np.std(res.reward_fixed)/math.sqrt(len(res.reward_fixed))
        y_est = np.mean(res.reward_est)
        se_est = np.std(res.reward_est)/math.sqrt(len(res.reward_est))
        trace_fixed['x'].append(x)
        trace_fixed['y'].append(y_fixed)
        trace_fixed['error_y']['array'].append(se_fixed)
        trace_est['x'].append(x)
        trace_est['y'].append(y_est)
        trace_est['error_y']['array'].append(se_est)

    plot.show_plot([trace_fixed, trace_est], title=title,
            xtitle="ground truth beta", ytitle="mean accumulated reward",
            save_png=save_png)


# TODO (?): Merge mega_ and run_trial()
def mega_trial(ctx, beta_star, fixed_beta, min_trials, max_trials,
        tolerance=0.5, est_ignore=False,
        k=None, verbose=False, plot=False):
    """ Output: ExResult """
    res = ExResult(beta_star=beta_star, ctx=ctx,
            fixed_beta=fixed_beta)
    if verbose:
        print("*******beta_star={:.3f}, fixed_beta={:.3f}************".format(
            beta_star, fixed_beta))
    for i in range(max_trials):
        r_fixed, r_est, col_fixed, col_est = run_trial(ctx=ctx,
                beta_star=beta_star, fixed_beta=fixed_beta, verbose=verbose,
                plot=plot, est_ignore=est_ignore)
        plot = False
        if verbose:
            print("summary: trial={}, r_fixed={:.3f}, r_est={:.3f}".format(
                i, r_fixed, r_est))
        # TODO: col_fixed/est aren't doing anything right now.
        # Figure out whether I want to remove or keep. XXX
        res.append(r_fixed, r_est)

        se_fixed = np.std(res.reward_fixed)/math.sqrt(len(res.reward_fixed))
        se_est = np.std(res.reward_est)/math.sqrt(len(res.reward_est))
        if i >= min_trials and max(se_fixed, se_est) < tolerance:
            break
    return res


def run_trial(ctx, beta_star, fixed_beta=1, T_max=30, k=None,
        verbose=False, plot=False, est_ignore=False):
    """ Output: r_fixed, r_est, num_collide_{fixed,est} """
    traj_H = simulate(mdp=ctx.g_H, initial_state=ctx.start_H,
            goal_state=ctx.g_H.goal, path_length=T_max, beta=beta_star)
    ctx.traj_H = traj_H


    res_fixed = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
            calc_beta=False, verbose=False)
    r_fixed = sum(res_fixed.rewards)
    num_collisions_fixed = res_fixed.collide_count


    if est_ignore:
        # A hack. TODO: fix this.
        res_est = res_fixed
    else:
        res_est = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
                calc_beta=True, verbose=False)

    if verbose and plot:
        print("Robot_fixed: ^")
        display(ctx.g_H, ctx.traj_H, ctx.start_H, ctx.g_H.goal,
                traj_aux=res_fixed.traj_R, failures=res_fixed.collisions[-1])
        print("------")
        print("Robot_est: ^")
        display(ctx.g_H, ctx.traj_H, ctx.start_H, ctx.g_H.goal,
                traj_aux=res_est.traj_R, failures=res_est.collisions[-1])
        print("------")

    if plot:
        compare2(ctx, plan_res1=res_est, plan_res2=res_fixed,
                title="beta_star={}".format(beta_star), save_png=False, skip=-1)
    r_est = sum(res_est.rewards)
    num_collisions_est = res_est.collide_count

    return r_fixed, r_est, num_collisions_fixed, num_collisions_est


class ExResult(object):
    def __init__(self, ctx, beta_star, est_beta_min=0.1, est_beta_max=6,
            fixed_beta=1):
        self.beta_star = beta_star
        self.fixed_beta = fixed_beta
        self.est_beta_min = est_beta_min
        self.est_beta_max = est_beta_max
        self.ctx = ctx  # Is self.ctx painful? Consider superclassing and using
                        # python's copy library. XXX
        self.reward_fixed = []
        self.reward_est = []
        self.num_trials = 0

    def append(self, r_fixed, r_est):
        self.reward_fixed.append(r_fixed)
        self.reward_est.append(r_est)
        self.num_trials += 1
