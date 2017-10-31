from __future__ import absolute_import
from unittest import TestCase

import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from value_iter import backwards_value_iter, forwards_value_iter
from inference import sample_action, simulate, _display, infer_destination, \
    infer_occupancies, infer_occupancies_from_start, infer_temporal_occupancies, \
    _sum_rewards
from beta_inference import beta_gradient_ascent, beta_binary_search, _compute_score
from itertools import izip

import plot

Actions = GridWorldMDP.Actions

def test_sample_action(beta=1, trials=10):
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    for i in xrange(trials):
        print Actions(sample_action(g, 25, 0, beta=beta))

def test_simulate_tiny_dest_set():
    g = GridWorldMDP(1, 3, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(0, 2)
    traj = simulate(g, 0, goal)
    traj = traj[:-2]
    dest_set = set([(0,0), (0,2)])
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_big_dest_set():
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(5, 4)
    traj = simulate(g, 0, goal)
    traj = traj[:-3]
    dest_set = set([(5,7), (0,4), (9, 9)])
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_huge_dest_set():
    g = GridWorldMDP(80, 80, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(79, 79)
    traj = simulate(g, 0, goal, path_length=20)
    dest_set = set([(79, 9), (50, 79), (20, 50)])
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_big_beta():
    g = GridWorldMDP(15, 15, {}, default_reward=-40)
    start = 0
    goal = g.coor_to_state(14, 13)
    traj = simulate(g, 0, goal, beta=3, path_length=7)
    traj = traj[:-3]
    dest_set = set([(6,9), (0,4), (14, 13)])
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, beta=1, heat_maps=(0,))
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, beta=2, heat_maps=(0,))
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, beta=3, heat_maps=(0,))
    plot.visualize_trajectory(g, start, goal, traj, dest_set=dest_set, beta=4, heat_maps=(0,))

def test_temporal_occupancy():
    g = GridWorldMDP(15, 15, {}, default_reward=-40)
    start = 0
    goal = g.coor_to_state(14, 13)
    traj = simulate(g, 0, goal, beta=3, path_length=7)
    traj = traj[:-3]
    dest_set = set([(6,9), (0,4), (14, 13)])
    # dest_set={g.coor_to_state(*d) for d in dest_set}
    plot.visualize_trajectory(g, start, goal, traj, T=10, c_0=-40, sigma_0=5, sigma_1=15, beta=2,
            dest_set=dest_set, heat_maps=(0, 1, 2, 5, 10), zmin=-20, zmax=0)

def test_expected_occupancy_start():
    g = GridWorldMDP(1, 8, {}, default_reward=-24)
    start = g.coor_to_state(0,4)
    beta = 2
    dest_set = set([(0,0), (0,6)])
    dest_set=set(g.coor_to_state(*d) for d in dest_set)
    D = infer_occupancies_from_start(g, start,
            beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)

    print u"expected occupancies:"
    print D

    occupancies = D
    data = []

    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    hm = go.Heatmap(z=np.log(occupancies.T))
    data.append(hm)

    if dest_set is not None:
        x, y = izip(*[g.state_to_coor(s) for s in dest_set])
        dest_markers = go.Scatter(x=x, y=y,
            mode=u'markers', marker=dict(size=20, color=u"white", symbol=u"star"))
        data.append(dest_markers)

    py.plot(data, filename=u'expected_occup.html')

def plot_traj_log_likelihood():
    g = GridWorldMDP(10, 10, {}, default_reward=-24)
    start = 0
    goal = g.coor_to_state(9, 9)
    traj = simulate(g, 0, goal, beta=6, path_length=5)
    # plot.visualize_trajectory(g, start, goal, traj, dest_set=[(9,9)], beta=6, heat_maps=(0,))

    x = np.arange(0.5, 11, 0.5)
    scores = [_compute_score(g, traj, goal, beta) for beta in x]
    print scores

    import plotly.offline as py
    import plotly.graph_objs as go
    trace = go.Scatter(x=x, y=scores, mode=u'markers')
    py.plot([trace], filename=u'beta.html')

    beta_hat = beta_binary_search(g, traj, goal, guess=1, verbose=True)
    print u"estimated beta={}".format(beta_hat)

# def value_iter_comparison():
#     for beta in [2,4,6,8]:
#
#     value_k

def andrea_occupancies():
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(5, 4)
    traj = simulate(g, 0, goal)
    traj = traj[:-3]
    dest_set = set([(5,7), (0,4), (9, 9)])
    dest_set=set(g.coor_to_state(*d) for d in dest_set)
    beta=1

    print u"Raw trajectory:"
    print [(g.state_to_coor(s), g.Actions(a)) for s, a in traj]
    print u"With overlay:"
    _display(g, traj, start, goal, overlay=True)
    print u"Trajectory only:"
    _display(g, traj, start, goal)

    P = infer_destination(g, traj, beta=beta, dest_set=dest_set)
    print u"goal probabilities:"
    print P.reshape(g.rows, g.cols)

    D = infer_occupancies(g, traj, beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)
    print u"expected occupancies:"
    print D

def smitha_softmax_diverge():
    g = GridWorldMDP(1, 3, {}, default_reward=-0.1)
    V = backwards_value_iter(g, init_state=0, max_iters=15, verbose=True)

andrea_occupancies()
# smitha_softmax_diverge()

# test_simulate_tiny_dest_set()
# test_simulate_big_dest_set()
# test_sample_action(2)
# test_simulate_big_beta()
# plot_traj_log_likelihood()
# test_expected_occupancy_start()
