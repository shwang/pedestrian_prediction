from __future__ import absolute_import

import numpy as np

from mdp import GridWorldMDP
from mdp.softmax import backwards_value_iter, forwards_value_iter
from inference.softmax.destination import infer_destination
from inference.softmax.occupancy import *
from util import sum_rewards, display, normalize
from util.softmax import simulate, sample_action
from inference.softmax.beta import beta_gradient_ascent, beta_binary_search, compute_score
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

# def value_iter_comparison():
#     for beta in [2,4,6,8]:
#
#     value_k

def andrea_occupancies():
    g = GridWorldMDP(6, 6, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(5, 5)
    beta = 1
    traj = simulate(g, 0, goal, beta=beta)
    traj = traj[:1]
    dest_set = {goal}

    print u"Raw trajectory:"
    print [(g.state_to_coor(s), g.Actions(a)) for s, a in traj]
    print u"With overlay:"
    display(g, traj, start, goal, overlay=True)
    print u"Trajectory only:"
    display(g, traj, start, goal)

    P = infer_destination(g, traj, beta=beta, dest_set=dest_set)
    print u"goal probabilities:"
    print P.reshape(g.rows, g.cols)

    D = infer_occupancies(g, traj, beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)
    print u"expected occupancies:"
    print D

#    plot.output_heat_map(g, D, traj, start, {goal}, plot=True)

def smitha_softmax_diverge():
    g = GridWorldMDP(1, 4, {}, default_reward=-3)
    init_vals = np.array([0, -3, -6, -9], dtype=float)
    init_vals = np.array([0, 0, 0, 0], dtype=float)
    V = backwards_value_iter(g, init_state=0, max_iters=10, verbose=True,
            beta=10, nachum=True, init_vals=init_vals)


# andrea_occupancies()
smitha_softmax_diverge()

# test_simulate_tiny_dest_set()
# test_simulate_big_dest_set()
# test_sample_action(2)
# test_simulate_big_beta()
# test_expected_occupancy_start()
