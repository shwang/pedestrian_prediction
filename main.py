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

Actions = GridWorldMDP.Actions

def test_sample_action(beta=1, trials=10):
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    for i in xrange(trials):
        print Actions(sample_action(g, 25, 0, beta=beta))

def test_simulate(g, start, goal, traj, beta=1, dest_set=None,
        T=0, c_0=-20, sigma_0=5, sigma_1=5, heat_maps=(), zmin=None, zmax=None):
    print u"Task: Start={}, Goal={}".format(g.state_to_coor(start), g.state_to_coor(goal))
    print u"Assumed beta={}".format(beta)
    print u"Possible goals:"
    if dest_set == None:
        print u"<all>"
    else:
        print dest_set
    dest_set=set(g.coor_to_state(*d) for d in dest_set)

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

    if T > 0:
        D_t = infer_temporal_occupancies(g, traj, beta=beta, dest_set=dest_set,
                T=T, c_0=c_0, sigma_0=sigma_0, sigma_1=sigma_1)
        D_t = list(x.reshape(g.rows, g.cols) for x in D_t)
        print u"calculated T={} expected temporal occupancies.".format(T)
        print u"Here is the {}th expected temporal occupancy:".format(T//2)
        print D_t[T//2]

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
            x, y = izip(*coors)
            traj_line = dict(x=x, y=y, line=dict(color=u'white', width=3))
            data.append(traj_line)

            if dest_set is not None:
                x, y = izip(*[g.state_to_coor(s) for s in dest_set])
                dest_markers = go.Scatter(x=x, y=y,
                    mode=u'markers', marker=dict(size=20, color=u"white", symbol=u"star"))
                data.append(dest_markers)

            for trace in data:
                fig.append_trace(trace, row + 1, 1)

        py.plot(fig, filename=u'expected_occup.html')

def test_simulate_tiny_dest_set():
    g = GridWorldMDP(1, 3, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(0, 2)
    traj = simulate(g, 0, goal)
    traj = traj[:-2]
    dest_set = set([(0,0), (0,2)])
    test_simulate(g, start, goal, traj, dest_set=dest_set)

def test_simulate_big_dest_set():
    g = GridWorldMDP(10, 10, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(5, 4)
    traj = simulate(g, 0, goal)
    traj = traj[:-3]
    dest_set = set([(5,7), (0,4), (9, 9)])
    test_simulate(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_huge_dest_set():
    g = GridWorldMDP(80, 80, {}, default_reward=-9)
    start = 0
    goal = g.coor_to_state(79, 79)
    traj = simulate(g, 0, goal, path_length=20)
    dest_set = set([(79, 9), (50, 79), (20, 50)])
    test_simulate(g, start, goal, traj, dest_set=dest_set, heat_maps=(0,))

def test_simulate_big_beta():
    g = GridWorldMDP(15, 15, {}, default_reward=-40)
    start = 0
    goal = g.coor_to_state(14, 13)
    traj = simulate(g, 0, goal, beta=3, path_length=7)
    traj = traj[:-3]
    dest_set = set([(6,9), (0,4), (14, 13)])
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=1, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=2, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=3, heat_maps=(0,))
    test_simulate(g, start, goal, traj, dest_set=dest_set, beta=4, heat_maps=(0,))

def test_temporal_occupancy():
    g = GridWorldMDP(15, 15, {}, default_reward=-40)
    start = 0
    goal = g.coor_to_state(14, 13)
    traj = simulate(g, 0, goal, beta=3, path_length=7)
    traj = traj[:-3]
    dest_set = set([(6,9), (0,4), (14, 13)])
    # dest_set={g.coor_to_state(*d) for d in dest_set}
    test_simulate(g, start, goal, traj, T=10, c_0=-40, sigma_0=5, sigma_1=15, beta=2,
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
    # test_simulate(g, start, goal, traj, dest_set=[(9,9)], beta=6, heat_maps=(0,))

    x = np.arange(0.5, 11, 0.5)
    scores = [_compute_score(g, traj, goal, beta) for beta in x]
    print scores

    import plotly.offline as py
    import plotly.graph_objs as go
    trace = go.Scatter(x=x, y=scores, mode=u'markers')
    py.plot([trace], filename=u'beta.html')

    beta_hat = beta_binary_search(g, traj, goal, guess=1, verbose=True)
    print u"estimated beta={}".format(beta_hat)

def output_heat_map(g, occupancies, traj, start_state, dest_set, beta_hat=None,
        zmin=None, zmax=None):
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    data = []

    o = np.log(occupancies.T)
    for i in xrange(o.shape[0]):
        for j in xrange(o.shape[1]):
            val = o[i, j]
            if val == -np.inf:
                o[i, j] = -9999

    hm = go.Heatmap(z=o, zmin=zmin, zmax=zmax,
            name=u"log expected occupancy")
    if beta_hat is not None:
        hm.name += u"(beta_hat={})".format(beta_hat)

    data.append(hm)

    states = [s for s, a in traj]
    if len(traj) > 0:
        states.append(g.transition(*traj[-1]))
    else:
        states.append(start)
    coors = [g.state_to_coor(s) for s in states]
    x, y = izip(*coors)
    traj_line = dict(x=x, y=y, line=dict(color=u'white', width=3))
    data.append(traj_line)

    x, y = izip(*[g.state_to_coor(s) for s in dest_set])
    dest_markers = go.Scatter(x=x, y=y,
        mode=u'markers', marker=dict(size=20, color=u"white", symbol=u"star"))
    data.append(dest_markers)

    return data

def plot_all_heat_maps():
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    N = 25
    g = GridWorldMDP(N, N, {}, default_reward=-25)
    start = 0
    # goal = g.S - 1
    # model_goal = g.coor_to_state(N-1,N//2)
    goal = g.coor_to_state(N-1, N//2)
    model_goal = goal
    true_beta = 5
    trajectory = simulate(g, start, goal, beta=true_beta)

    beta_fixed = 1
    beta_hat = 5
    min_beta = 0.83
    max_beta = 11
    def format_occ(occupancies):
        for i, val in enumerate(occupancies):
            if val == -np.inf:
                occupancies[i] = -99999
        return occupancies.reshape(g.rows, g.cols)

    for i in xrange(len(trajectory) - 1):
        traj = trajectory[:i+1]
        beta_hat = beta_binary_search(g, traj, model_goal, guess=beta_hat, verbose=True,
                min_beta=min_beta, max_beta=max_beta)
        print u"{}: beta_hat={}".format(i+1, beta_hat)
        occupancies = format_occ(infer_occupancies(g, traj, beta=beta_hat,
                dest_set=set([model_goal])))
        data1 = output_heat_map(g, occupancies, traj, start, dest_set=set([model_goal]),
                beta_hat=beta_hat, zmin=-8, zmax=0)

        fixed_occupancies = format_occ(infer_occupancies(g, traj, beta=beta_fixed,
                dest_set=set([model_goal])))
        data2 = output_heat_map(g, fixed_occupancies, traj, start, dest_set=set([model_goal]),
                beta_hat=beta_fixed, zmin=-8, zmax=0)

        true_occupancies = format_occ(infer_occupancies(g, traj, beta=true_beta,
                dest_set=set([goal])))
        data3 = output_heat_map(g, true_occupancies, traj, start, dest_set=set([goal]),
                beta_hat=true_beta, zmin=-8, zmax=0)

        first_title = u"wrong goal, beta_hat={} (MLE)".format(beta_hat)
        if abs(beta_hat - 11) < 1e-4:
            first_title += u" (beta at max!)"
        elif abs(beta_hat - 0.8) < 1e-4:
            first_title += u" (beta at min!)"

        fig = tools.make_subplots(rows=1, cols=3,
                subplot_titles=(
                    first_title,
                    u"wrong goal, beta={} (Ziebart beta)".format(beta_fixed),
                    u"correct goal, ground truth beta={}".format(true_beta)))
        fig[u'layout'].update(title=u"t={}".format(i+1))

        for t in data1:
            fig.append_trace(t, 1, 1)
        for t in data2:
            fig.append_trace(t, 1, 2)
        for t in data3:
            fig.append_trace(t, 1, 3)
        py.plot(fig, filename=u"output/{}.html".format(100+i))

def plot_all_heat_maps2():
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    N = 15
    g = GridWorldMDP(N, N, {}, default_reward=-10)
    start = 0
    goal = g.S - 1
    # model_goal = g.coor_to_state(N-1,N//2)
    # goal = g.coor_to_state(N-1, N//2)
    model_goal = goal
    true_beta = 0.2
    trajectory = simulate(g, start, goal, beta=true_beta)

    beta_fixed = 1
    beta_hat = 1
    min_beta = 0.02
    max_beta = 8
    zmin = -20
    def format_occ(occupancies):
        return occupancies.reshape(g.rows, g.cols)

    print trajectory

    for i in xrange(len(trajectory) - 1):
        traj = trajectory[:i+1]
        beta_hat = beta_binary_search(g, traj, model_goal, guess=beta_hat, verbose=True,
                min_beta=min_beta, max_beta=max_beta)
        print u"{}: beta_hat={}".format(i+1, beta_hat)
        occupancies = format_occ(infer_occupancies(g, traj, beta=beta_hat,
                dest_set=set([model_goal])))
        data1 = output_heat_map(g, occupancies, traj, start, dest_set=set([model_goal]),
                zmin=zmin, zmax=0)

        fixed_occupancies = format_occ(infer_occupancies(g, traj, beta=beta_fixed,
                dest_set=set([model_goal])))
        data2 = output_heat_map(g, fixed_occupancies, traj, start, dest_set=set([model_goal]),
                zmin=zmin, zmax=0)

        first_title = u"beta_hat={} (MLE)".format(beta_hat)
        if abs(beta_hat - max_beta) < 1e-4:
            first_title += u" (beta at max!)"

        fig = tools.make_subplots(rows=1, cols=2,
                subplot_titles=(
                    first_title,
                    u"beta={} (Ziebart beta)".format(beta_fixed)))
        fig[u'layout'].update(title=u"t={}".format(i+1))

        for t in data1:
            fig.append_trace(t, 1, 1)
        for t in data2:
            fig.append_trace(t, 1, 2)
        py.plot(fig, filename=u"output/{}.html".format(100+i),
            image=u'png', image_filename=u"output/{}.png".format(100+i),
            image_width=1400, image_height=750)

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
