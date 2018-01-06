from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..parameters import inf_default
from ..plot import common as plot

from .robot_planner import robot_planner
from .context import HRContext
from scenarios import scenario_starter

def distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def experiment_k(mode, N=10, fixed_beta=1, k1=None, k2=None,
        collide_radius=2, collide_penalty=10, **calc_plans_kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res = calc_plans(ctx=ctx, k=k1, beta_guess=fixed_beta,
            **calc_plans_kwargs)
    plots1 = gen_all_subplots(ctx, plan_res, title=TITLE_K)

    plan_res = calc_plans(
            ctx=ctx, beta_guess=fixed_beta, k=k2, **calc_plans_kwargs)
    plots2 = gen_all_subplots(ctx, plan_res, title=TITLE_K)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}").format(
                    name=mode, t="{t}", c_radius=collide_radius,
                    c_penalty=collide_penalty)
    compare2(ctx, plots1, plots2, title=title)


def experiment_est_vs_fixed(mode, N=10, fixed_beta=1,
        collide_radius=2, collide_penalty=10, **calc_plans_kwargs):
    ctx = HRContext(collide_radius=collide_radius,
            collide_penalty=collide_penalty)
    ctx.cfg_mode(mode, N)

    plan_res = calc_plans(ctx=ctx, beta_guess=fixed_beta,
            **calc_plans_kwargs)
    plots1 = gen_all_subplots(ctx, plan_res)

    plan_res = calc_plans(ctx=ctx, beta_guess=fixed_beta, calc_beta=False,
            **calc_plans_kwargs)
    plots2 = gen_all_subplots(ctx, plan_res)

    title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
            "collide_penalty={c_penalty}")
    compare2(ctx, plots1, plots2, title=title)


def compare2(ctx, plots1, plots2, title):
    for t in range(max(len(plots1), len(plots2))):
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
                save_png=True)


class MetaPlanResult(object):
    def __init__(self, ctx=None, k=None):
        self.ctx = ctx
        self.k = k
        self.collide_count = 0
        self.plans = []
        self.traj_R = []
        self.states_R = []
        self.states_H = []
        self.rewards = []
        self.expected_costs = []
        self.betas = []
        self.ideal_betas = []
        self.collisions = []


# TODO: consider altering max_steps here. There have been cases where the lowest
# cost paths seem to have longer lengths than 25.
def calc_plans(ctx, max_steps=25, inf_mod=inf_default, k=None,
        calc_beta=True, beta_guess=0.8, beta_fallback=1.5, beta_decay=0.9,
        verbose=False):

    n = len(ctx.traj_H)
    traj_H = list(ctx.traj_H)
    traj_H += [(ctx.g_H.transition(*traj_H[-1]), ctx.g_H.Actions.ABSORB)] * \
            (max_steps - n)

    res = MetaPlanResult(ctx=ctx, k=k)

    curr_collisions = []
    state_R = ctx.start_R
    reward_delayed = 0
    for t in range(max_steps):
        if verbose:
            print("step {}".format(t))
        tr_H = traj_H[:t]
        if t == 0:
            state_H = traj_H[0][0]
        else:
            state_H = ctx.g_H.transition(*tr_H[-1])
        x_H, y_H = ctx.g_H.state_to_coor(state_H)
        x_R, y_R = ctx.g_R.state_to_coor(state_R)

        reward = reward_delayed
        if distance(x_R, y_R, x_H, y_H) <= ctx.collide_radius:
            res.collide_count += 1
            reward -= ctx.collide_penalty
            curr_collisions = curr_collisions + [state_R]
        res.collisions.append(curr_collisions)

        res.states_R.append(state_R)
        res.states_H.append(state_H)

        # TODO: Clean this
        plan = None
        ideal_beta = bg = beta_guess
        cb = calc_beta
        while plan is None:
            plan, ex_cost, node, beta = robot_planner(
                    traj_H=tr_H, state_R=state_R,
                    ctx=ctx, k=k, verbose_return=True, calc_beta=cb,
                    max_depth=max_steps, beta_guess=bg)
            if cb:
                ideal_beta = beta
            if verbose:
                print("beta={:.3f}, plan={}").format(beta, plan)
            bg = min(beta * beta_decay, beta_fallback)
            cb=False

        state_R = ctx.g_R.transition(*plan[0])
        reward_delayed = ctx.g_R.rewards[plan[0][0], plan[0][1]]

        res.traj_R.append(plan[0])
        res.plans.append(plan)
        res.rewards.append(reward)
        res.expected_costs.append(ex_cost)
        res.betas.append(beta)
        res.ideal_betas.append(ideal_beta)


        if node.traj[0] == (ctx.g_R.goal, ctx.g_R.Actions.ABSORB):
            break

    res.traj_R.append([ctx.g_R.goal, ctx.g_R.Actions.ABSORB])

    return res


TITLE_K = "k={k}<br>{beta_text}<br>" + \
        "accumulated_reward={reward:.3f}"
TITLE = "{beta_text}<br>" + \
        "accumulated_reward={reward:.3f}"

def gen_all_subplots(ctx, plan_res, title=TITLE, **kwargs):
    subplots = []
    for t in range(len(plan_res.plans)):
        subplots.append(gen_subplot(t, ctx, plan_res, title=title, **kwargs))
    return subplots

def gen_subplot(t, ctx, plan_res, title=None, occ_depth=50,
        inf_mod=inf_default):
    """
    Generate a subplot with human occupancy heat map, human and robot
    trajectories, robot plan, and starred goals.

    Params:
        title [string] (optional) -- Change this to use a custom title.
        beta [float] (optional) -- By default, calculate MLE beta from the
            human's trajectory. Otherwise, provide a value for beta here.
            (Note: if t=0, there is no trajectory to infer from, so we assume
             beta=1).
        occ_depth [int] (optional) -- The depth of the occupancy calculation.
    """
    beta = plan_res.betas[t]
    ideal_beta = plan_res.ideal_betas[t]
    reward = sum(plan_res.rewards[:t+1])
    expected_cost = plan_res.expected_costs[t]
    g_H, g_R = ctx.g_H, ctx.g_R
    k = plan_res.k

    if t == 0:
        tr_H = [(ctx.start_H, g_H.Actions.ABSORB)]
        tr_R = [(ctx.start_R, g_H.Actions.ABSORB)]
    else:
        tr_H = ctx.traj_H[:t]
        tr_R = plan_res.traj_R[:t]
    plan_R = plan_res.plans[t]

    # TODO(?): rename ideal_beta to fixed_beta
    if t == 0:
        occupancies = inf_mod.occupancy.infer_from_start(g_H, ctx.start_H,
                g_H.goal, T=occ_depth, beta_or_betas=beta,
                verbose_return=False)
    else:
        occupancies = inf_mod.occupancy.infer(g_H, tr_H, g_H.goal,
                T=occ_depth, beta_or_betas=beta, verbose_return=False)

    title = title or TITLE
    if ideal_beta != beta:
        beta_text = "mle_beta={ideal_beta:.3f} [comp_beta={beta:.3f}]"
    else:
        beta_text = "mle_beta={beta:.3f}"
    beta_text = beta_text.format(beta=beta, ideal_beta=ideal_beta)
    title = title.format(beta_text=beta_text, value=-expected_cost,
            reward=reward, k=k)

    data = []

    hm = plot.make_heat_map(g_H, occupancies, auto_logarithm=False, zmin=0,
            zmax=1)
    hm.update(showscale=False)
    data.append(hm)

    if k is not None:
        if k == 0:
            tr_H = []
        elif k > 0:
            tr_H = tr_H[-k:]
        else:
            print("warning: invalid k={}".format(k))

    data.append(plot.make_line(g_H, tr_H, name="human traj", color='blue'))
    data.append(plot.make_line(g_R, tr_R, name="robot traj", color='green'))
    data.append(plot.make_line(g_R, plan_R, name="robot plan", color='green',
        dash='dot'))

    data.append(plot.make_stars(g_H, [g_H.goal], name="human goal",
        color='blue'))
    data.append(plot.make_stars(g_R, [g_R.goal], name="robot goal",
        color='green'))

    data.append(plot.make_stars(g_H, [plan_res.states_H[t]], color='blue',
        name="human loc", symbol="diamond"))
    data.append(plot.make_stars(g_R, [plan_res.states_R[t]], color='green',
        name="robot loc", symbol="diamond"))

    data.append(plot.make_stars(g_R, plan_res.collisions[t], color='red',
        name="collision", symbol=4, size=16))

    shapes = []
    shapes.append(plot.make_rect(g_H, g_H.transition(*tr_H[-1]),
        ctx.collide_radius))

    return title, data, shapes
