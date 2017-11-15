from __future__ import absolute_import

import numpy as np

from mdp import GridWorldMDP
from mdp.softmax import backwards_value_iter, forwards_value_iter
from inference.softmax.destination import infer_destination
from inference.softmax.occupancy import *
from util import sum_rewards, display
from util.softmax import simulate, sample_action
from inference.softmax.beta import *
from itertools import izip

Actions = GridWorldMDP.Actions

def build_traj_from_actions(g, init_state, actions):
    s = init_state
    traj = []
    for a in actions:
        traj.append((s, a))
        s = g.transition(s, a)
    return traj

def visualize_trajectory(g, start, goal, traj, beta=1, dest_set=None,
        T=0, c_0=-20, sigma_0=5, sigma_1=5, heat_maps=(), zmin=None, zmax=None,
        uid="tmp"):
    print "Task: Start={}, Goal={}".format(g.state_to_coor(start), g.state_to_coor(goal))
    print "Assumed beta={}".format(beta)
    print "Possible goals:"
    if dest_set == None:
        dest_set = {goal}

    dest_set=set(d if type(d) is int else g.coor_to_state(*d) for d in dest_set)
    print dest_set

    print "Raw trajectory:"
    print [(g.state_to_coor(s), g.Actions(a)) for s, a in traj]
    print "With overlay:"
    _display(g, traj, start, goal, overlay=True)
    print "Trajectory only:"
    _display(g, traj, start, goal)

    P = infer_destination(g, traj, beta=beta, dest_set=dest_set)
    print "goal probabilities:"
    print P.reshape(g.rows, g.cols)

    D = infer_occupancies(g, traj, beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)
    print "expected occupancies:"
    print

    if T > 0:
        D_t = infer_temporal_occupancies(g, traj, beta=beta, dest_set=dest_set,
                T=T, c_0=c_0, sigma_0=sigma_0, sigma_1=sigma_1)
        D_t = list(x.reshape(g.rows, g.cols) for x in D_t)
        print "calculated T={} expected temporal occupancies.".format(T)
        print "Here is the {}th expected temporal occupancy:".format(T//2)
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
            traj_line = dict(x=x, y=y, line=dict(color='white', width=3))
            data.append(traj_line)

            if dest_set is not None:
                x, y = izip(*[g.state_to_coor(s) for s in dest_set])
                dest_markers = go.Scatter(x=x, y=y,
                    mode='markers', marker=dict(size=20, color="white", symbol="star"))
                data.append(dest_markers)

            for trace in data:
                fig.append_trace(trace, row + 1, 1)

        py.plot(fig, filename='output/expected_occup_{}.html'.format(uid))


def output_heat_map(g, occupancies, traj, start_state, dest_set, beta_hat=None,
        zmin=None, zmax=0, auto_logarithm=True, plot=False):
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    data = []

    o = occupancies.T
    if auto_logarithm:
        o = np.log(o)

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
        states.append(start_state)
    coors = [g.state_to_coor(s) for s in states]
    x, y = izip(*coors)
    traj_line = dict(x=x, y=y, line=dict(color=u'white', width=3))
    data.append(traj_line)

    x, y = izip(*[g.state_to_coor(s) for s in dest_set])
    dest_markers = go.Scatter(x=x, y=y,
        mode=u'markers', marker=dict(size=20, color=u"white", symbol=u"star"))
    data.append(dest_markers)

    if plot:
        py.plot(data, filename='output/output_heat_map.html')
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

def plot_all_heat_maps3():
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    N = 6
    g = GridWorldMDP(N, N, {}, default_reward=-10)
    start = 0
    # goal = g.S - 1
    goal = g.coor_to_state(N-1, 0)
    # model_goal = g.coor_to_state(N-1,N//2)
    # goal = g.coor_to_state(N-1, N//2)
    model_goal = goal
    true_beta = 0.2
    trajectory = simulate(g, start, goal, beta=true_beta)

    beta_fixed = 1
    beta_hat = 1
    min_beta = 0.2
    max_beta = 5
    zmin = -20
    def format_occ(occupancies):
        return occupancies.reshape(g.rows, g.cols)

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
        py.plot(fig, filename=u"output/{}.html".format(100+i))
        # py.plot(fig, filename=u"output/{}.html".format(100+i),
        #      image=u'png', image_filename=u"output/{}.png".format(100+i),
        #      image_width=1400, image_height=750)

def plot_traj_log_likelihood(g, traj, goal, title=None, add_grad=True, add_beta_hat=True,
        beta_min=0.2, beta_max=8.0, beta_step=0.05, plot=True, verbose=True):

    x = np.arange(beta_min, beta_max, beta_step)
    scores = [_compute_score(g, traj, goal, beta) for beta in x]

    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import tools
    data = []
    trace1 = go.Scatter(name="log likelihood", x=x, y=scores, mode='markers')
    data.append(trace1)

    if add_beta_hat:
        # beta_hat = beta_binary_search(g, traj, goal, guess=1, verbose=True)
        beta_hat = beta_simple_search(g, traj, goal, guess=1, verbose=True)
        beta_hat_score = _compute_score(g, traj, goal, beta_hat)
        trace2 = go.Scatter(name="beta_hat log likelihood",
                x=[beta_hat], y=[beta_hat_score], mode='markers')
        data.append(trace2)

    if add_grad:
        grads = [_compute_gradient(g, traj, goal, beta) for beta in x]
        trace3 = go.Scatter(name="gradient of log likelihood", x=x, y=grads, mode='markers')
        data.append(trace3)

    layout = go.Layout(title=title, xaxis=dict(title="beta"))

    if plot:
        fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=(
                    "log likelihood of trajectory",
                    "gradient of log likelihood"
                    ))
        fig.append_trace(trace1, 1, 1)
        if add_beta_hat:
            fig.append_trace(trace2, 1, 1)
        if add_grad:
            fig.append_trace(trace3, 2, 1)
        fig['layout'].update(title=title, xaxis=dict(title="beta"))
        py.plot(fig, filename='output/beta.html')


    if verbose and add_beta_hat:
        print "estimated beta={}".format(beta_hat)

    return data, layout

def log_likelihood_wrt_beta():
    N = 5
    g = GridWorldMDP(N, N, {}, default_reward=-24)
    start = 0
    goal = N*N-1
    traj = simulate(g, 0, goal, beta=0.3, path_length=5)[:-1]
    visualize_trajectory(g, 0, goal, traj, heat_maps=(0,))
    plot_traj_log_likelihood(g, traj, goal, add_grad=False)

def shortest_paths_beta_hat():
    """
    Output a heat map showing the beta_hat that would result from taking the
    shortest path to a given state.
    """
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools
    N = 10
    g = GridWorldMDP(N, N, {}, default_reward=-24)
    start = 0
    goal = g.coor_to_state(N//2, N-1)
    # goal = N*N-1
    min_beta = 0.2
    max_beta = 10

    beta_hats = np.zeros(N*N)
    beta_hats[0] = np.nan
    for s in range(1, N*N):
        traj = simulate(g, 0, s, beta=0.1)[:-1]
        beta_hats[s] = beta_simple_search(g, traj, goal, guess=1,
                verbose=False, min_beta=min_beta, max_beta=max_beta,
                min_iters=6, max_iters=12)

    beta_hats = beta_hats.reshape([g.rows, g.cols])
    data = output_heat_map(g, beta_hats, traj=[], start_state=0, dest_set={goal},
            beta_hat=0, zmin=min_beta, zmax=max_beta, auto_logarithm=False)
    fig = go.Figure(data=data,
            layout=dict(title="beta estimate for shortest path to each square"))
    # fig['name'] = 'beta_hat'
    py.plot(fig)
    import pdb; pdb.set_trace()

def plot_all_heat_maps2():
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    N = 40
    R = -8
    g = GridWorldMDP(N, N, {}, default_reward=R)
    start = 0
    # start = g.coor_to_state(N//2, 0)
    goal = g.S - 1
    # goal = g.coor_to_state(N-1, 0)
    # model_goal = g.coor_to_state(N-1,N//2)
    # goal = model_goal = g.coor_to_state(N//2, N-1)
    # goal = g.coor_to_state(N-1, N//2)
    model_goal = goal
    true_beta = 0.01
    trajectory = simulate(g, start, goal, beta=true_beta)
    # trajectory = build_traj_from_actions(g, start, [Actions.UP] * N)
    # trajectory = [[0, 1], [4, 1], [8, 1], [12, 8]]

    beta_fixed = 1
    beta_hat = 1
    min_beta = 0.02
    max_beta = 1.19
    auto_log=True
    zmin = -3.5
    zmax = 0
    # auto_log=False
    # zmin = 0
    # zmax = 0.7
    def format_occ(occupancies):
        return occupancies.reshape(g.rows, g.cols)

    for i in xrange(len(trajectory)):
        if i == 0:
            traj = [(start, Actions.ABSORB)]
        else:
            traj = trajectory[:i]

        fixed_occupancies = format_occ(infer_occupancies(g, traj, beta=beta_fixed,
                dest_set=set([model_goal])))
        data1 = output_heat_map(g, fixed_occupancies, traj, start,
                dest_set=set([model_goal]),
                auto_logarithm=auto_log,
                zmin=zmin, zmax=zmax)

        fig = tools.make_subplots(rows=1, cols=1,
                subplot_titles=(
                    "beta={} (Ziebart beta)".format(beta_fixed),))
        fig['layout'].update(
                title=("Correct Goal, movement reward={R}, " +
                "{min_beta}&lt;beta&lt;{max_beta}<br>t={t}").format(
                    t=i, R=R, min_beta=min_beta, max_beta=max_beta))

        for t in data1:
            fig.append_trace(t, 1, 1)
        # py.plot(fig, filename=u"output/{}.html".format(100+i))
        py.plot(fig, filename=u"output/{}.html".format(100+i),
            image=u'png', image_filename=u"output/{}.png".format(100+i),
            image_width=1400, image_height=750)

def plot_all_heat_maps3():
    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import offline
    from plotly import tools as tools

    N = 20
    R = -2.5
    g = GridWorldMDP(N, N, {}, default_reward=R)
    start = 0
    # start = g.coor_to_state(N//2, 0)
    # goal = g.S - 1
    # goal = g.coor_to_state(N-1, 0)
    # model_goal = g.coor_to_state(N-1,N//2)
    goal = model_goal = g.coor_to_state(N//2, N-1)
    # goal = g.coor_to_state(N-1, N//2)
    model_goal = goal
    true_beta = 1.12
    trajectory = simulate(g, start, goal, beta=true_beta)
    # trajectory = build_traj_from_actions(g, start, [Actions.UP] * N)
    # trajectory = [[0, 1], [4, 1], [8, 1], [12, 8]]

    beta_fixed = 1
    beta_hat = 1
    min_beta = 0.02
    max_beta = 1.19
    auto_log=True
    zmin = -5
    zmax = 0
    # auto_log=False
    # zmin = 0
    # zmax = 0.7
    def format_occ(occupancies):
        return occupancies.reshape(g.rows, g.cols)

    for i in xrange(len(trajectory)):
        if i == 0:
            traj = [(start, Actions.ABSORB)]
        else:
            traj = trajectory[:i]
        #beta_hat = beta_binary_search(g, traj, model_goal, guess=beta_hat, verbose=True,
        #        min_beta=min_beta, max_beta=max_beta)
        if i == 0:
            beta_hat = 1
        else:
            beta_hat = beta_simple_search(g, traj, model_goal, guess=beta_hat,
                    verbose=True, min_beta=min_beta, max_beta=max_beta)
            print "{}: beta_hat={}".format(i+1, beta_hat)

        occupancies = format_occ(infer_occupancies(g, traj, beta=beta_hat,
                dest_set=set([model_goal])))
        data1 = output_heat_map(g, occupancies, traj, start, dest_set=set([model_goal]),
                auto_logarithm=auto_log,
                zmin=zmin, zmax=zmax)

        fixed_occupancies = format_occ(infer_occupancies(g, traj, beta=beta_fixed,
                dest_set=set([model_goal])))
        data2 = output_heat_map(g, fixed_occupancies, traj, start,
                dest_set=set([model_goal]),
                auto_logarithm=auto_log,
                zmin=zmin, zmax=zmax)

        true_occupancies = format_occ(infer_occupancies(g, traj, beta=true_beta,
                dest_set=set([model_goal])))
        data3 = output_heat_map(g, true_occupancies, traj, start,
                dest_set=set([model_goal]),
                auto_logarithm=auto_log,
                zmin=zmin, zmax=zmax)

        first_title = "beta_hat={} (MLE)".format(beta_hat)
        if abs(beta_hat - max_beta) < 1e-4:
            first_title += " (beta at max!)"

        fig = tools.make_subplots(rows=1, cols=3,
                subplot_titles=(
                    first_title,
                    "beta={} (Ziebart beta)".format(beta_fixed),
                    "ground truth beta={}".format(true_beta)))
        fig['layout'].update(
                title=("Correct Goal, movement reward={R}, " +
                "{min_beta}&lt;beta&lt;{max_beta}<br>t={t}").format(
                    t=i, R=R, min_beta=min_beta, max_beta=max_beta))

        for t in data1:
            fig.append_trace(t, 1, 1)
        for t in data2:
            fig.append_trace(t, 1, 2)
        for t in data3:
            fig.append_trace(t, 1, 3)
        # py.plot(fig, filename=u"output/{}.html".format(100+i))
        py.plot(fig, filename=u"output/{}.html".format(100+i),
            image=u'png', image_filename=u"output/{}.png".format(100+i),
            image_width=1400, image_height=750)


if __name__ == '__main__':
    # log_likelihood_wrt_beta()
    # shortest_paths_beta_hat()
    # plot_all_heat_maps3()
    # study_traj()
    plot_all_heat_maps2()
