from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..parameters import inf_default
from ..plot import common as plot

from .context import HRContext
from scenarios import scenario_starter
from .simulate import simulate_bayes

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


def experiment_est_vs_fixed(mode, N=10, fixed_beta=1,
        collide_radius=2, collide_penalty=10, **kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res1 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta,
            **kwargs)
    return

    plan_res2 = simulate_vanilla(ctx=ctx, beta_guess=fixed_beta, calc_beta=False,
            **kwargs)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}")
    compare2(ctx, plan_res1, plan_res2, title=title)

def experiment_est_vs_bayes(mode, N=10, fixed_beta=1,
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
        title = title.format(
                    name=ctx.mode, t=t, c_radius=ctx.collide_radius,
                    c_penalty=ctx.collide_penalty)
        plot.subplots([data1, data2], [title1, title2], title=title,
                shapes_list=[shapes1, shapes2],
                legend_settings=dict(orientation='h', x=0.5, borderwidth=1),
                save_png=save_png)
