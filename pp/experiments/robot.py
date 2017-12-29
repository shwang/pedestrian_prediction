from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..parameters import inf_default
from ..plot import common as plot
from ..util import build_traj_from_actions, sum_rewards

from .robot_planner import robot_planner

def distance(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))

# TODO: Consider moving this table-like function into its own file
def _scenario_starter(mode, N, reward_human=-1, reward_robot=-1):
    A = GridWorldMDP.Actions
    g_H = GridWorldMDP(N, N, {}, default_reward=reward_human, allow_wait=True)
    g_R = GridWorldMDP(N, N, {}, default_reward=reward_robot,
            allow_wait=False)
    one = g_H.coor_to_state(1, 1)

    c_x = c_y = N//2
    coor = lambda x, y: g_H.coor_to_state(x, y)
    center = coor(c_x, c_y)
    center_left = coor(1, c_y)
    center_right = coor(N-2, c_y)
    center_top = coor(c_x, N-2)
    center_bot = coor(c_x, 1)
    top_right = coor(N-2, N-2)
    top_left = coor(1, N-2)
    bot_left = coor(1, 1)
    bot_right = coor(N-2, 1)


    if mode == "diag only":
        # no difference with collision_penalty=10
        n = (N-3)//2
        actions = [A.UP_RIGHT] * n + [A.UP_LEFT] * n
        start_H, goal_H = bot_left, coor(1, 1+2*n)
        s = int(0.7 * N)
        start_R, goal_R = coor(s, s), bot_left
    elif mode == "invisible wall":
        s = int(0.3 * N)
        start_H, goal_H = coor(N - s, 1), bot_left
        start_R, goal_R = coor(N-1, s), coor(1, s)
        w = N - 5 - s//2
        W = N - 5 - w
        actions = [A.LEFT] * w + [A.UP]*s + [A.LEFT]*2 + \
                [A.DOWN]*s  # + [A.LEFT] * W
    elif mode == "zigzag":
        s = int(0.4 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = top_right, bot_left
        actions = [A.LEFT]*(s-1) + [A.UP_RIGHT]*((N-1)//2) +\
                [A.UP_LEFT]*((N-1)//2)
    elif mode == "hug walls 2":
        start_H, goal_H = bot_right, top_left
        s = int(0.7 * N)
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.LEFT]*(N-3) + [A.UP]*(N-3)
    elif mode == "hug walls 3":
        s = int(0.7 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.LEFT]*(s-1) + [A.UP]*(s-1)
    elif mode == "hug walls 4":
        s = int(0.7 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.DOWN_LEFT] + [A.LEFT]*(s-2) + [A.UP]*(s-2) + \
                [A.UP_RIGHT]
    elif mode == "obtuse":
        start_H, goal_H = bot_left, center_left
        start_R, goal_R = top_right, bot_left
        n = (N-3)//2
        actions = [A.UP_RIGHT] * n + [A.RIGHT] * n
    elif mode == "beta_lag":
        start_H, goal_H = center_left, center_top
        start_R, goal_R = top_right, bot_left
        actions = [A.RIGHT] * (N-3)
    elif mode == "beta_lag 2":
        start_H, goal_H = center_left, coor(N-2, N-1)
        start_R, goal_R = top_right, bot_left
        actions = [A.RIGHT] * (N-3)
    elif mode == "treacherous turn":
        start_H, goal_H = center_left, center_right
        start_R, goal_R = top_right, bot_left
        n = (N-3)//2
        actions = [A.RIGHT] * n + [A.UP_RIGHT] * n
    elif mode == "sync vert":
        start_H, goal_H = coor(c_x + 1, N-2), coor(c_x + 1, 1)
        start_R, goal_R = coor(c_x - 1, N-2), coor(c_x - 1, 1)
        actions = [A.DOWN] * (N-3)
    elif mode == "sync vert antiparallel":
        # Equal reward results: for N=15, radius=1
        start_H, goal_H = coor(c_x + 1, N-2), coor(c_x + 1, 1)
        start_R, goal_R = coor(c_x - 1, 1), coor(c_x - 1, N-2)
        actions = [A.DOWN] * (N-3)
    elif mode == "cross":
        start_H, goal_H = center_right, top_right
        start_R, goal_R = center_top, center_bot
        actions = [A.LEFT] * int(.9 * N)
    elif mode == "shuffle":
        # AKA awkward shuffle
        # good example. (similar to diag shuffle)
        start_H, goal_H = center, center_left
        start_R, goal_R = g_H.coor_to_state(c_x, c_y + 4), center_bot
        actions = [A.RIGHT] * int(.8 * N / 2)
    elif mode == "shuffle 2":
        # good example -- N=17, radius=2
        start_H, goal_H = g_H.coor_to_state(c_x, c_y+1), center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.DOWN] + [A.RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "head-on":
        # N=12, collide_radius=1 ==> better for fixed.
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP] * (int(.8 * N / 2) - 2)
    elif mode == "diag shuffle":
        # good example -- N=17, radius=2
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP_RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "diag shuffle mod":
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP_RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "actually criss-cross":
        start_H, goal_H = one, g_H.coor_to_state(N//2, N-2)
        start_R, goal_R = g_R.coor_to_state(N-2, 1), g_R.coor_to_state(0, N-2)
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "criss-cross" or mode == "diag-cross":
        start_H, goal_H = one, g_H.coor_to_state(N-2, N-2)
        start_R, goal_R = g_R.coor_to_state(N-2, 1), g_R.coor_to_state(0, N-2)
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "criss-cross false":
        # Bad experiment -- when human diverges, human walks perfectly to avoid
        # collision
        start_H, goal_H = one, g_H.coor_to_state(N-2, N-2)
        start_R, goal_R = g_R.coor_to_state(N-2, 1), g_R.coor_to_state(0, N-2)
        w = (N-2) // 2
        W = N - 2 - w
        actions = [A.UP_RIGHT] * w + [A.DOWN_RIGHT] * W
    elif mode == "swap corners":
        start_H, goal_H = one, g_H.coor_to_state(N-2, N-2)
        start_R, goal_R = goal_H, start_H
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "swap corners false":
        start_H, goal_H = one, g_H.coor_to_state(N-2, N-2)
        start_R, goal_R = goal_H, start_H
        w = (N-2) // 2
        W = N - 2 - w
        actions = [A.RIGHT] * w + [A.DOWN_RIGHT] * 1
    else:
        raise Exception("invalid mode: {}".format(mode))

    g_R.set_goal(goal_R)
    g_H.set_goal(goal_H)
    traj_H = build_traj_from_actions(g_H, start_H, actions)

    return g_H, start_H, traj_H, g_R, start_R


def experiment_plot(mode, N=10, fixed_beta=1,
        collide_radius=2, collide_penalty=10):
    g_H, start_H, traj_H, g_R, start_R = _scenario_starter(mode, N)

    plots1 = []
    traj_R, plans, rewards, ex_costs, betas, kwargs = \
            calc_plans(
            mode=mode, N=N, collide_radius=collide_radius,
            collide_penalty=collide_penalty, beta_guess=fixed_beta)
    for t, (plan, ex_cost, reward, beta) in enumerate(
            zip(plans, ex_costs, np.cumsum(rewards), betas)):
        tr_H, tr_R = traj_H[:t], traj_R[:t]
        plots1.append(gen_subplot(t, traj_H=tr_H, traj_R=tr_R, plan_R=plan,
            beta=beta, expected_cost=ex_cost, reward=reward, **kwargs))

    plots2 = []
    traj_R, plans, rewards, ex_costs, betas, kwargs = calc_plans(
            mode, N=N, beta_guess=fixed_beta, calc_beta=False,
            collide_radius=collide_radius, collide_penalty=collide_penalty)
    for t, (plan, ex_cost, reward, beta) in enumerate(
            zip(plans, ex_costs, np.cumsum(rewards), betas)):
        tr_H, tr_R = traj_H[:t], traj_R[:t]
        plots2.append(gen_subplot(t, traj_H=tr_H, traj_R=tr_R, plan_R=plan,
            beta=beta, expected_cost=ex_cost, reward=reward, **kwargs))

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
        title = ("scenario={name}. t={t}<br>collide_radius={c_radius}<br>" +
                "collide_penalty={c_penalty}").format(
                        name=mode, t=t, c_radius=collide_radius,
                        c_penalty=collide_penalty)
        plot.subplots([data1, data2], [title1, title2], title=title,
                shapes_list=[shapes1, shapes2],
                legend_settings=dict(orientation='h',
                    x=0.5,
                    borderwidth=1),
                save_png=True)

def calc_plans(mode, N=10, max_steps=50, inf_mod=inf_default,
        collide_radius=3, collide_penalty=6, calc_beta=True, beta_guess=0.8):

    g_H, start_H, traj_H, g_R, start_R = _scenario_starter(mode, N)

    n = len(traj_H)
    traj_H += [(g_H.transition(*traj_H[-1]), g_H.Actions.ABSORB)] * (max_steps - n)

    collide_count = 0
    plans = []
    traj_R = []
    states_R = []
    states_H = []
    rewards = []
    expected_costs = []
    betas = []
    curr_collisions = []
    collisions = []

    state_R = start_R
    reward_delayed = 0
    for t in range(max_steps):
        print("step {}".format(t))
        tr = traj_H[:t]
        if t == 0:
            state_H = traj_H[0][0]
        else:
            state_H = g_H.transition(*tr[-1])
        x_H, y_H = g_H.state_to_coor(state_H)
        x_R, y_R = g_R.state_to_coor(state_R)

        reward = reward_delayed
        if distance(x_R, y_R, x_H, y_H) <= collide_radius:
            collide_count += 1
            reward -= collide_penalty
            curr_collisions = curr_collisions + [state_R]
        collisions.append(curr_collisions)

        states_R.append(state_R)
        states_H.append(state_H)

        plan, ex_cost, node, beta = robot_planner(
                g_R, state_R, g_H, traj=tr, start_H=start_H,
                collide_radius=collide_radius, collide_penalty=collide_penalty,
                verbose_return=True, calc_beta=calc_beta, beta_guess=beta_guess)

        state_R = g_R.transition(*plan[0])
        reward_delayed = g_R.rewards[plan[0][0], plan[0][1]]

        traj_R.append(plan[0])
        plans.append(plan)
        rewards.append(reward)
        expected_costs.append(ex_cost)
        betas.append(beta)

        if node.traj[0] == (g_R.goal, g_R.Actions.ABSORB):
            break

    traj_R.append([g_R.goal, g_R.Actions.ABSORB])

    return traj_R, plans, rewards, expected_costs, betas, \
        dict(g_H=g_H, g_R=g_R, start_R=start_R, start_H=start_H,
                collide_radius=collide_radius,
                collisions=collisions,
                states_H=states_H, states_R=states_R)


def gen_subplot(t, traj_H, traj_R, plan_R,
        ### set these params using dict ###
        g_H, g_R, start_R, start_H, collide_radius, states_R, states_H,
        collisions,
        ###################################
        expected_cost=0, reward=0, title=None, beta=None, occ_depth=50,
        collide_count=None, inf_mod=inf_default):
    """
    Generate a subplot with human occupancy heat map, human and robot
    trajectories, robot plan, and starred goals.

    Params:
        traj_H, traj_R, plan_R -- Lists of (state, action) describing
            trajectories or plans.
        start_R, start_H -- Used in lieu of trajectories when those are empty,
            as they should be when t=0.
        expected_value [float] -- The expected value of the robot's plan.
        title [string] (optional) -- Change this to use a custom title.
        beta [float] (optional) -- By default, calculate MLE beta from the
            human's trajectory. Otherwise, provide a value for beta here.
            (Note: if t=0, there is no trajectory to infer from, so we assume
             beta=1).
        occ_depth [int] (optional) -- The depth of the occupancy calculation.
    """
    if t == 0:
        occupancies, _, _, beta = inf_mod.occupancy.infer_from_start(g_H,
                start_H, g_H.goal, T=occ_depth, beta_or_betas=beta,
                verbose_return=True)
    else:
        occupancies, _, _, beta = inf_mod.occupancy.infer(g_H,
                traj_H, g_H.goal, T=occ_depth, beta_or_betas=beta,
                verbose_return=True)

    title = title or "beta={beta:.3f} plan_value={value:.3f}<br>" + \
            "accumulated_reward={reward:.3f}"
    title = title.format(beta=beta[0], value=-expected_cost,
            reward=reward)

    data = []

    hm = plot.make_heat_map(g_H, occupancies, auto_logarithm=False, zmin=0,
            zmax=1)
    hm.update(showscale=False)
    data.append(hm)

    # These trajectories are empty. Give them a starting dot.
    if t == 0:
        traj_H = [(start_H, g_H.Actions.ABSORB)]
        traj_R = [(start_R, g_H.Actions.ABSORB)]

    data.append(plot.make_line(g_H, traj_H, name="human traj", color='blue'))
    data.append(plot.make_line(g_R, traj_R, name="robot traj", color='green'))
    data.append(plot.make_line(g_R, plan_R, name="robot plan", color='green',
        dash='dot'))

    data.append(plot.make_stars(g_H, [g_H.goal], name="human goal",
        color='blue'))
    data.append(plot.make_stars(g_R, [g_R.goal], name="robot goal",
        color='green'))

    data.append(plot.make_stars(g_H, [states_H[t]], color='blue',
        name="human loc", symbol="diamond"))
    data.append(plot.make_stars(g_R, [states_R[t]], color='green',
        name="robot loc", symbol="diamond"))

    data.append(plot.make_stars(g_R, collisions[t], color='red',
        name="collision", symbol=4, size=16))

    shapes = []
    shapes.append(plot.make_rect(g_H, g_H.transition(*traj_H[-1]),
        collide_radius))

    return title, data, shapes
