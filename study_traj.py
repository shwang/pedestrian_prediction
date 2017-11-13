from __future__ import absolute_import
from unittest import TestCase

import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from value_iter import *
from inference import *
from beta_inference import *
from beta_inference import _compute_score, _compute_gradient
from plot import *
from itertools import izip

A = GridWorldMDP.Actions

def build_traj_from_actions(g, init_state, actions):
    s = init_state
    traj = []
    for a in actions:
        traj.append((s, a))
        s = g.transition(s, a)
    return traj

def assemble(g, start, actions, goal):
    """
    Produce "study trajectory" information for this trajectory.
    """
    import plotly.offline as py
    import plotly.graph_objs as go

    traj = build_traj_from_actions(g, start, actions)

    min_beta = 0.2
    max_beta = 10
    zmin=-20
    def format_occ(occupancies):
        return occupancies.reshape(g.rows, g.cols)

    beta_hats = []
    beta_vs_score_plots = []
    occupancy_plots = []
    for i in xrange(len(traj) - 1):
        t = traj[:i+1]
        beta_hat = beta_binary_search(g, t, goal, guess=1, verbose=True,
                min_beta=min_beta, max_beta=max_beta)
        beta_hats.append(beta_hat)

        # occupancies = format_occ(infer_occupancies(g, traj, beta=beta_hat,
        #     dest_set={goal}))
        # occupancy_plots.append(output_heat_map(g, occupancies, traj, start, {goal},
        #     beta_hat=beta_hat, zmin=zmin))

        data, layout = plot_traj_log_likelihood(g, t, goal, title="t={}".format(i))
        beta_vs_score_plots.append([data, layout])

def study_traj():
    N = 4
    g = GridWorldMDP(N, N, {}, default_reward=-24)
    start = 0
    # goal = g.S - 1
    goal = g.coor_to_state(N-1, 0)

    traj = [[0, Actions.RIGHT], [4, Actions.RIGHT], [8, Actions.RIGHT]]
    print traj
    print build_traj_from_actions(g, 0, [Actions.RIGHT] * 3)
    # visualize_trajectory(g, 0, goal, traj, heat_maps=(0,))
    # plot_traj_log_likelihood(g, traj, goal,
    #        beta_min=0.2, beta_max=11, beta_step=0.2, add_grad=True, add_beta_hat=True)
    # print _compute_gradient(g, traj, goal, 1)


def shard_study_traj2():
    N = 5
    g = GridWorldMDP(N, N, {}, default_reward=-24)
    actions = [A.UP_RIGHT, A.UP, A.UP_RIGHT, A.UP]
    goal = g.coor_to_state(N//2, N-1)


def compare_ex():
    import matplotlib
    from matplotlib import pyplot as plt
    betas = np.arange(5.6, 6.6, 0.05)
    ex_1s = []
    ex_2s = []
    for beta in betas:
        print "\nbeta={}".format(beta)
        grad, ex_1, ex_2 = _compute_gradient(g, traj, goal, beta=beta, debug=True)
        ex_1s.append(ex_1)
        ex_2s.append(ex_2)
        score = _compute_score(g, traj, goal, beta=beta)
        print "score={}: grad={}".format(score, grad)

    import plotly.offline as py
    import plotly.graph_objs as go
    ex_1s = np.array(ex_1s)
    ex_2s = np.array(ex_2s)
    diff = (ex_1s - ex_2s)
    data = []
    data.append(dict(x=betas, y=ex_1s))
    data.append(dict(x=betas, y=ex_2s))
    data.append(dict(x=betas, y=diff))
    py.plot(data, filename="output/grad_debug.html")

def expected_from_start():
    import plotly.offline as py
    import plotly.graph_objs as go
    actions = [A.ABSORB]
    traj = build_traj_from_actions(g, start, actions)
    # D_start = infer_occupancies_from_start(g, 0, beta=6, dest_set={goal}, gamma=0.9,
    #         ).reshape(g.rows, g.cols)
    D = infer_occupancies(g, traj, dest_set={goal}, gamma=0.9,
           ).reshape(g.rows, g.cols)
    # D = D_start - D
    data = output_heat_map(g, D, traj, start_state=0, dest_set={goal},
            auto_logarithm=False)
    fig = go.Figure(data=data, layout=dict(title="occupancies from wait"))
    py.plot(fig)

def beta_versus(g, start, actions, goal, beta1, beta2, uid=0, title=None):
    traj = build_traj_from_actions(g, start, actions)
    o1 = infer_occupancies(g, traj, beta=beta1,
        dest_set={goal}).reshape(g.rows, g.cols)
    data1 = output_heat_map(g, o1, traj, start, {goal}, beta_hat=beta1, zmin=-10)

    o2 = infer_occupancies(g, traj, beta=beta2,
        dest_set={goal}).reshape(g.rows, g.cols)
    data2 = output_heat_map(g, o2, traj, start, {goal}, beta_hat=beta2, zmin=-10)

    o_diff = np.abs(o1 - o2)
    data3 = output_heat_map(g, o_diff, traj, start, {goal}, beta_hat=beta1, zmin=-10)

    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools
    fig = tools.make_subplots(rows=1, cols=3,
            subplot_titles=(
                "expected occupancies, beta={}".format(beta1),
                "expected occupancies, beta={}".format(beta2),
                "abs difference in ex. occupancies"))
    fig['layout'].update(title=title)

    for t in data1:
        fig.append_trace(t, 1, 1)
    for t in data2:
        fig.append_trace(t, 1, 2)
    for t in data3:
        fig.append_trace(t, 1, 3)
    # py.plot(fig, filename="output/beta_versus.html")
    py.plot(fig, filename="output/beta_versus_{}.html".format(100+uid),
            image='png', image_filename="output/beta_versus_{}.png".format(100+uid),
            image_width=1400, image_height=750)

#def shard_study_traj3():
N = 30
# R = -24
R = -5
g = GridWorldMDP(N, N, {}, default_reward=R)
start = 0
actions = [A.UP_RIGHT, A.UP_RIGHT, A.UP_RIGHT]
traj = build_traj_from_actions(g, start, actions)
goal = g.S - 1
# goal = g.coor_to_state(N//2, N-1)
# plot_traj_log_likelihood(g, traj, goal, title="study_traj_3, problem A.4. Beta stupid search")
# beta_versus(g, 0, actions, goal, 6.32, 5.85)
# print(beta_stupid_search(g, traj, goal, guess=1, verbose=True))
# expected_from_start()


# title="Comparing betas when movement reward={}"
# beta1, beta2 = 0.6, 1
# def compare(R):
#     g = GridWorldMDP(N, N, {}, default_reward=R)
#     beta_versus(g, start, actions, goal, beta1, beta2,
#             uid=R, title=title.format(R))
# 
# compare(-36)
# compare(-24)
# compare(-10)
# compare(-5)
# compare(-4)
# compare(-3)

def compare_beta(betas):
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    x = betas
    y = []
    for b in betas:
        _, steps = backwards_value_iter(g, 0, beta=b, debug_iters=True)
        print(x, steps)
        y.append(steps)
    data = [dict(x=x, y=y)]
    layout = dict(
        title='Softmax Value iteration on 30x30 Gridworld with movement reward=-24',
        xaxis=dict(title='beta'),
        yaxis=dict(title='iterations before convergence'),
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)

#for i in [0.5, 1, 3, 5, 7, 9, 11, 11.1, 11.2, 11.3]
# for i in [11.4, 11.5, 11.6, 11.7]:
#for i in [11.51, 11.52]:
# for i in [11.53, 11.54]:
#     compare_beta(i)
# compare_beta([0.5, 1,2, 3, 5, 7, 9, 11, 11.1, 11.2, 11.3, 11.4, 11.5])
# compare_beta([0.5, 1,2,2.1,2.2,2.3,2.4])
_, steps = backwards_value_iter(g, 0, beta=2.42)
print(steps)

# beta_versus(g, 0, actions, goal, 6.32, 0.3125)
# assemble(g, 0, actions, goal)
