from unittest import TestCase

import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from value_iter import backwards_value_iter, forwards_value_iter
from inference import sample_action, simulate, _display, infer_destination, \
    infer_occupancies, infer_occupancies_from_start, infer_temporal_occupancies, \
    _sum_rewards

Actions = GridWorldMDP.Actions

ni = float('-inf')
def test_fixed_start():
    g = GridWorldMDP(10, 10, {}, default_reward=-3)
    V = backwards_value_iter(g, 0, None, max_iters=1000, fixed_init=True, verbose=True)

def test_known_goal():
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    V = backwards_value_iter(g, 0, 90, max_iters=1000)

def test_known_goal_no_wait():
    g = GridWorldMDP(4, 1, {}, default_reward=-1)
    # g = GridWorldMDP(10, 10, {}, default_reward=-1)
    V = backwards_value_iter(g, 0, max_iters=1000, fixed_init=True, verbose=True)

def test_known_goal_no_wait_forward():
    g = GridWorldMDP(4, 1, {}, default_reward=-1)
    # g = GridWorldMDP(10, 10, {}, default_reward=-1)
    V = forwards_value_iter(g, 3, max_iters=1000, fixed_goal=True, verbose=True)

def test_sample_action(beta=1, trials=10):
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    for i in range(trials):
        print(Actions(sample_action(g, 25, 0, beta=beta)))

def test_simulate(g, start, goal, traj, beta=1, dest_set=None,
        T=0, c_0=-20, sigma_0=5, sigma_1=5, heat_maps=(), zmin=None, zmax=None):
    print("Task: Start={}, Goal={}".format(g.state_to_coor(start), g.state_to_coor(goal)))
    print("Assumed beta={}".format(beta))
    print("Possible goals:", end=' ')
    if dest_set == None:
        print("<all>")
    else:
        print(dest_set)
    dest_set={g.coor_to_state(*d) for d in dest_set}

    print("Raw trajectory:")
    print([(g.state_to_coor(s), g.Actions(a)) for s, a in traj])
    print("With overlay:")
    _display(g, traj, start, goal, overlay=True)
    print("Trajectory only:")
    _display(g, traj, start, goal)

    P = infer_destination(g, traj, beta=beta, dest_set=dest_set)
    print("goal probabilities:")
    print(P.reshape(g.rows, g.cols))

    D = infer_occupancies(g, traj, beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)
    print("expected occupancies:")
    print(D)

    if T > 0:
        D_t = infer_temporal_occupancies(g, traj, beta=beta, dest_set=dest_set,
                T=T, c_0=c_0, sigma_0=sigma_0, sigma_1=sigma_1)
        D_t = list(x.reshape(g.rows, g.cols) for x in D_t)
        print("calculated T={} expected temporal occupancies.".format(T))
        print("Here is the {}th expected temporal occupancy:".format(T//2))
        print(D_t[T//2])

    if len(heat_maps) > 0:
        # These guys take some time to import, so only do so if necessary.
        import plotly.offline as py
        import plotly.graph_objs as go
        from plotly import offline
        from plotly import tools as tools

        fig = tools.make_subplots(rows=len(heat_maps), cols=1)
        # plot_no = 0 ==> plot D.
        # plot_no > 0 ==> plot D_t[plot_no - 1]
        for row, plot_no in enumerate(heat_maps):
            assert plot_no >= 0, heat_maps
            occupancies = D if plot_no == 0 else D_t[plot_no - 1]
            data = []

            hm = go.Heatmap(z=np.log(occupancies.T), zmin=zmin, zmax=zmax)
            data.append(hm)

            states = [s for s, a in traj]
            if len(traj) > 0:
                states.append(g.transition(*traj[-1]))
            else:
                states.append(start)
            coors = [g.state_to_coor(s) for s in states]
            x, y = zip(*coors)
            traj_line = dict(x=x, y=y, line=dict(color='white', width=3))
            data.append(traj_line)

            if dest_set is not None:
                x, y = zip(*[g.state_to_coor(s) for s in dest_set])
                dest_markers = go.Scatter(x=x, y=y,
                    mode='markers', marker=dict(size=20, color="white", symbol="star"))
                data.append(dest_markers)

            for trace in data:
                fig.append_trace(trace, row + 1, 1)

        py.plot(fig, filename='expected_occup.html')

def test_simulate_small():
    g = GridWorldMDP(1, 5, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(0, 3)
    traj = simulate(g, 0, goal)
    traj = traj[:-1]  # remove potential ABSORB (bad for inference as of now)
    test_simulate(g, start, goal, traj)

def test_simulate_tiny_dest_set():
    g = GridWorldMDP(1, 3, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(0, 2)
    traj = simulate(g, 0, goal)
    traj = traj[:-2]
    dest_set = {(0,0), (0,2)}
    test_simulate(g, start, goal, traj, dest_set=dest_set)

def test_simulate_big_dest_set():
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(5, 4)
    traj = simulate(g, 0, goal)
    traj = traj[:-3]
    dest_set = {(5,7), (0,4), (9, 9)}
    test_simulate(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_huge_dest_set():
    g = GridWorldMDP(80, 80, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(79, 79)
    traj = simulate(g, 0, goal, path_length=20)
    dest_set = {(79, 9), (50, 79), (20, 50)}
    test_simulate(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_big_beta():
    g = GridWorldMDP(15, 15, {}, default_reward=-40)
    start = 0
    goal = g.coor_to_state(14, 13)
    traj = simulate(g, 0, goal, beta=3, path_length=7)
    traj = traj[:-3]
    dest_set = {(6,9), (0,4), (14, 13)}
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=1, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=2, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=3, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=4, heat_maps=(0,))

test_simulate_big_beta()
