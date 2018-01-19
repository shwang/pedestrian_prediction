from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..parameters import inf_default
from ..plot import common as plot

from .context import HRContext
from scenarios import scenario_starter
from .simulate import *

def experiment_k(mode, N=10, fixed_beta=1, k1=None, k2=None,
        collide_radius=2, collide_penalty=10, **kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res1 = simulate_vanilla(ctx=ctx, k=k1, beta_guess=fixed_beta,
            **kwargs)

    plan_res2 = simulate_vanilla(
            ctx=ctx, beta_guess=fixed_beta, k=k2, **kwargs)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}").format(
                    name=mode, t="{t}", c_radius=collide_radius,
                    c_penalty=collide_penalty)
    compare2(ctx, plan_res1, plan_res2, title=title)


def experiment_mle_vs_fixed(mode, N=10, fixed_beta=1,
        collide_radius=2, collide_penalty=10, **kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res1 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
            **kwargs)

    plan_res2 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
            calc_beta=False, **kwargs)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}")
    compare2(ctx, plan_res1, plan_res2, title=title)

def experiment_mle_vs_bayes(mode, N=10, betas=[0.2, 0.5, 0.8, 1, 2, 5, 10, 20],
        priors=None,
        collide_radius=2, collide_penalty=20, **kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res1 = simulate_vanilla(ctx=ctx, **kwargs)

    plan_res2 = simulate_bayes(ctx=ctx, betas=betas, priors=priors, **kwargs)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}")
    compare25(ctx, plan_res1, plan_res2, title=title)

def experiment_3(mode, N=10, betas=[0.2, 0.5, 0.8, 1, 2, 5],
        priors=None, fixed_beta=1,
        collide_radius=2, collide_penalty=10, **kwargs):

    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res1 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta, **kwargs)

    plan_res2 = simulate_bayes(ctx=ctx, betas=betas, priors=priors, **kwargs)

    plan_res3 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
            calc_beta=False, **kwargs)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}")
    compare3(ctx, plan_res1, plan_res2, plan_res3, title=title)


def compare2(ctx, plan_res1, plan_res2, title, save_png=True, skip=0):
    plots1 = plan_res1.gen_all_subplots()
    plots2 = plan_res2.gen_all_subplots()
    rng = list(range(max(len(plots1), len(plots2))))
    for t in rng[skip:]:
        title1, data1, shapes1 = plots1[min(t, len(plots1) - 1)]
        title2, data2, shapes2 = plots2[min(t, len(plots2) - 1)]
        seen = set()
        for d in [data1, data2]:
            for trace in d:
                if "name" in trace:
                    if trace['name'] == "collision":
                        if len(trace['x']) == 0:
                            continue
                    if trace['name'] not in seen:
                        seen.add(trace['name'])
                    else:
                        trace.update(showlegend=False)
                else:
                    trace.update(showlegend=False)
        title_f = title.format(
                    name=ctx.mode, t=t, c_radius=ctx.collide_radius,
                    c_penalty=ctx.collide_penalty)
        plot.subplots([data1, data2], [title1, title2], title=title_f,
                shapes_list=[shapes1, shapes2],
                legend_settings=dict(orientation='h', x=0.5, borderwidth=1),
                save_png=save_png)

def compare25(ctx, simres1, simres_bayes, title, save_png=True, skip=0):
    plots1 = simres1.gen_all_subplots()
    plots2 = simres_bayes.gen_all_subplots()
    bars = simres_bayes.gen_all_barplots()
    rng = list(range(max(len(plots1), len(plots2))))
    for t in rng[skip:]:
        title1, data1, shapes1 = plots1[min(t, len(plots1) - 1)]
        title2, data2, shapes2 = plots2[min(t, len(plots2) - 1)]
        title3, data3 = "Bayes P(beta)", [bars[min(t, len(bars) - 1)]]
        seen = set()
        for d in [data1, data2]:
            for trace in d:
                if "name" in trace:
                    if trace['name'] == "collision":
                        if len(trace['x']) == 0:
                            continue
                    if trace['name'] not in seen:
                        seen.add(trace['name'])
                    else:
                        trace.update(showlegend=False)
                else:
                    trace.update(showlegend=False)
        title_f = title.format(
                    name=ctx.mode, t=t, c_radius=ctx.collide_radius,
                    c_penalty=ctx.collide_penalty)
        plot.subplots([data1, data2, data3], [title1, title2, title3],
                title=title_f,
                shapes_list=[shapes1, shapes2],
                legend_settings=dict(orientation='h', x=0.5, borderwidth=1),
                save_png=save_png)


def compare3(ctx, simres1, simres_bayes, simres3, title, save_png=True, skip=0):
    plots1 = simres1.gen_all_subplots()
    plots2 = simres_bayes.gen_all_subplots()
    plots3 = simres3.gen_all_subplots()
    rng = list(range(max(len(plots1), len(plots2), len(plots3))))
    for t in rng[skip:]:
        title1, data1, shapes1 = plots1[min(t, len(plots1) - 1)]
        title2, data2, shapes2 = plots2[min(t, len(plots2) - 1)]
        title3, data3, shapes3 = plots3[min(t, len(plots3) - 1)]
        seen = set()
        for d in [data1, data2, data3]:
            for trace in d:
                if "name" in trace:
                    if trace['name'] == "collision":
                        if len(trace['x']) == 0:
                            continue
                    if trace['name'] not in seen:
                        seen.add(trace['name'])
                    else:
                        trace.update(showlegend=False)
                else:
                    trace.update(showlegend=False)
        title_f = title.format(
                    name=ctx.mode, t=t, c_radius=ctx.collide_radius,
                    c_penalty=ctx.collide_penalty)
        plot.subplots([data1, data2, data3], [title1, title2, title3],
                title=title_f,
                shapes_list=[shapes1, shapes2, shapes3],
                legend_settings=dict(orientation='h', x=0.5, borderwidth=1),
                save_png=save_png)
