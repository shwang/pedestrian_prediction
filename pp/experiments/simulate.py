from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..parameters import inf_default
from ..plot import common as plot

from .robot_planner import *
from .context import HRContext
from scenarios import scenario_starter

def distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def simulate_bayes(ctx, betas, priors=None, verbose=False, **kwargs):
    """
    Calculate the actions, plans, and rewards for a Bayes beta Robot interacting
    with Human in the scenario described by `ctx`.
    Returns: An instance of `SimResultBayes`.
    """
    # XXX: Add trajectory forget support via k later.
    #       Probably also requires more changes to this function.
    res = SimResultBayes(ctx=ctx, betas=betas, priors=priors)

    def planning_callback(ctx, res, tr_H, state_R):
        # TODO: Consider enabling k later. By accessing k stored in closure.
        plan, ex_cost, traj, P_beta = robot_planner_bayes(
                ctx=ctx, state_R=state_R, betas=betas, priors=priors,
                traj_H=tr_H)
        assert plan is not None, "A* depth exceeded the preset limit"
        res.P_betas.append(P_beta)
        return plan, ex_cost, traj

    return _sim_shared(ctx, res, planning_callback, verbose=verbose, **kwargs)


def simulate_vanilla(ctx, k=None,
        calc_beta=True, beta_guess=0.8, beta_fallback=1.5, beta_decay=0.9,
        verbose=False, **kwargs):
    """
    calc_beta=True corresponds to fixed_beta planning using beta_guess.
    calc_beta=False corresponse to mle_beta planning using beta_guess as the
        initial guess for "gradient descent".

    Returns: An instance of `SimResult` or `SimResultK`, depending on whether
        the k parameter is set.
    """

    if k is None:
        res = SimResult(ctx=ctx, k=k)
    else:
        res = SimResultK(ctx=ctx, k=k)

    # Dividing vanilla and fixed into two differnt functions looks like a bad
    # idea now...

    if k is None:
        res = SimResult(ctx=ctx, k=k)
    else:
        res = SimResultK(ctx=ctx, k=k)

    if calc_beta:
        planner = robot_planner_vanilla
    else:
        planner = robot_planner_fixed

    def planning_callback(ctx, res, tr_H, state_R, verbose=False):
        # TODO: Clean this
        plan = None
        ideal_beta = bg = beta_guess
        cb = calc_beta
        while plan is None:
            if calc_beta:
                planner = robot_planner_vanilla
            else:
                planner = robot_planner_fixed
            plan, ex_cost, traj, beta = planner(
                    traj_H=tr_H, state_R=state_R,
                    ctx=ctx, k=k, beta_guess=bg)
            if cb:
                ideal_beta = beta
            if verbose:
                print("beta={:.3f}, plan={}").format(beta, plan)
            bg = min(beta * beta_decay, beta_fallback)
            cb=False

        res.betas.append(beta)
        res.ideal_betas.append(ideal_beta)

        return plan, ex_cost, traj

    return _sim_shared(ctx, res, planning_callback, verbose=verbose, **kwargs)


def _sim_shared(ctx, res, planning_callback, max_steps=None, verbose=False):
    if max_steps is None:
        max_steps = ctx.g_R.S
    n = len(ctx.traj_H)
    traj_H = list(ctx.traj_H)
    traj_H += [(ctx.g_H.transition(*traj_H[-1]), ctx.g_H.Actions.ABSORB)] * \
            (max_steps - n)

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

        # Need to delay robot movement reward by one timestep.
        # Otherwise collisions penalties and robot movement rewards are
        # out-of-sync.
        reward = reward_delayed

        # Check for collisions.
        if distance(x_R, y_R, x_H, y_H) <= ctx.collide_radius:
            res.collide_count += 1
            reward -= ctx.collide_penalty
            curr_collisions = curr_collisions + [state_R]
        res.collisions.append(curr_collisions)

        res.states_R.append(state_R)
        res.states_H.append(state_H)

        # Modular call to robot_planner_* function and associated processing.
        plan, ex_cost, traj = planning_callback(ctx, res, tr_H, state_R)

        state_R = ctx.g_R.transition(*plan[0])
        reward_delayed = ctx.g_R.rewards[plan[0][0], plan[0][1]]

        res.traj_R.append(plan[0])
        res.plans.append(plan)
        res.rewards.append(reward)
        res.expected_costs.append(ex_cost)

        if traj[0] == (ctx.g_R.goal, ctx.g_R.Actions.ABSORB):
            break

    res.traj_R.append([ctx.g_R.goal, ctx.g_R.Actions.ABSORB])
    return res


class SimResult(object):
    def __init__(self, ctx=None, k=None, variant="vanilla"):
        self.ctx = ctx
        self.variant=variant
        self.is_bayes = False
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


    TITLE = "{beta_text}<br> accumulated_reward={reward:.3f}"
    def gen_all_subplots(self, title=None, **kwargs):
        title = title or self.TITLE
        subplots = []
        for t in range(len(self.plans)):
            subplots.append(self.gen_subplot(t, title=title,
                **kwargs))
        return subplots


    def gen_subplot(self, t, title=None, occ_depth=50, inf_mod=inf_default):
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
        ctx = self.ctx
        g_H, g_R = ctx.g_H, ctx.g_R
        k = self.k

        if t == 0:
            tr_H = [(ctx.start_H, g_H.Actions.ABSORB)]
            tr_R = [(ctx.start_R, g_H.Actions.ABSORB)]
        else:
            tr_H = ctx.traj_H[:t]
            tr_R = self.traj_R[:t]

        if k is not None:
            if k == 0:
                tr_H = []
            elif k > 0:
                tr_H = tr_H[-k:]
            else:
                print("warning: invalid k={}".format(k))

        title = title or TITLE
        title = self.format(title, t)

        occupancies = self.get_occupancies(t, tr_H, occ_depth=occ_depth)
        data, shapes = self.make_graph_objs(t, tr_H, tr_R, occupancies)
        return title, data, shapes

    def get_occupancies(self, t, tr_H, occ_depth, inf_mod=inf_default):
        ctx = self.ctx
        g_H, g_R = ctx.g_H, ctx.g_R
        beta = self.betas[t]
        if t == 0:
            occupancies = inf_mod.occupancy.infer_from_start(g_H, ctx.start_H,
                    g_H.goal, T=occ_depth, beta_or_betas=beta,
                    verbose_return=False)
        else:
            occupancies = inf_mod.occupancy.infer(g_H, tr_H, g_H.goal,
                    T=occ_depth, beta_or_betas=beta, verbose_return=False)
        return occupancies


    def make_graph_objs(self, t, tr_H, tr_R, occupancies):
        ctx = self.ctx
        g_H, g_R = ctx.g_H, ctx.g_R
        hm = plot.make_heat_map(g_H, occupancies, auto_logarithm=False, zmin=0,
                zmax=1)
        hm.update(showscale=False)

        data = []
        data.append(hm)

        data.append(plot.make_line(g_H, tr_H, name="human traj", color='blue'))
        data.append(plot.make_line(g_R, tr_R, name="robot traj", color='green'))
        data.append(plot.make_line(g_R, self.plans[t], name="robot plan",
            color='green', dash='dot'))

        data.append(plot.make_stars(g_H, [g_H.goal], name="human goal",
            color='blue'))
        data.append(plot.make_stars(g_R, [g_R.goal], name="robot goal",
            color='green'))

        data.append(plot.make_stars(g_H, [self.states_H[t]], color='blue',
            name="human loc", symbol="diamond"))
        data.append(plot.make_stars(g_R, [self.states_R[t]], color='green',
            name="robot loc", symbol="diamond"))

        data.append(plot.make_stars(g_R, self.collisions[t], color='red',
            name="collision", symbol=4, size=16))

        shapes = []
        shapes.append(plot.make_rect(g_H, g_H.transition(*tr_H[-1]),
            ctx.collide_radius))
        return data, shapes

    def format(self, s, t):
        beta = self.betas[t]
        ideal_beta = self.ideal_betas[t]
        reward = sum(self.rewards[:t+1])
        expected_cost = self.expected_costs[t]
        k = self.k

        if ideal_beta != beta:
            beta_text = "beta={ideal_beta:.3f} [comp_beta={beta:.3f}]"
        else:
            beta_text = "beta={beta:.3f}"
        beta_text = beta_text.format(beta=beta, ideal_beta=ideal_beta)
        return s.format(beta_text=beta_text, value=-expected_cost,
                reward=reward, k=k)


class SimResultK(SimResult):
    def __init__(self, *args, **kwargs):
        SimResult.__init__(self, *args, variant="k", **kwargs)

    TITLE = "Traj Forgetting (k={k})<br>{beta_text}<br>" +\
                "accumulated_reward={reward:.3f}"


class SimResultBayes(SimResult):
    def __init__(self, betas=None, priors=None, *args, **kwargs):
        SimResult.__init__(self, *args, variant="bayes", **kwargs)
        self.P_betas = []
        self.betas = betas
        self.priors = priors

    TITLE = "Bayes<br>accumulated_reward={reward:.3f}"

    def get_occupancies(self, t, tr_H, occ_depth, inf_mod=inf_default):
        ctx = self.ctx
        occupancies = inf_mod.occupancy.infer_bayes(ctx.g_H, dest=ctx.start_H,
                T=occ_depth, betas=self.betas, traj=tr_H,
                init_state=ctx.start_H, priors=self.priors)

        return occupancies

    def format(self, s, t):
        reward = sum(self.rewards[:t+1])
        expected_cost = self.expected_costs[t]
        k = self.k

        return s.format(value=-expected_cost, reward=reward, k=k)
