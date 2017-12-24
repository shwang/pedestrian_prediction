import numpy as np

from ..mdp import GridWorldMDP
from ..inference.softmax.destination import infer_destination

from ..util import build_traj_from_actions
from ..util.hardmax import simulate, sample_action

from ..parameters import inf_default

import plotly.offline as py
from plotly import tools as tools
import plotly.graph_objs as go

Actions = GridWorldMDP.Actions


def make_heat_map(g, occupancies, zmin=-5, zmax=0, auto_logarithm=True):
    o = occupancies.reshape(g.rows, g.cols).T
    if auto_logarithm:
        o = np.log(o)
    o[o == -np.inf] = -9999
    hm = go.Heatmap(z=o, zmin=zmin, zmax=zmax, name="log expected occupancy")
    return hm


def make_line(g, traj, **xtra_line_settings):
    settings = dict(color='white', width=3)
    settings.update(xtra_line_settings)

    states = [s for s, a in traj]
    states.append(g.transition(*traj[-1]))
    coors = [g.state_to_coor(s) for s in states]
    x, y = zip(*coors)
    line = dict(x=x, y=y, line=settings)
    return line


def make_stars(g, stars=[], **xtra_marker_settings):
    settings = dict(size=20, color="white", symbol="star")
    settings.update(xtra_marker_settings)

    x, y = zip(*[g.state_to_coor(s) for s in stars])
    markers = go.Scatter(x=x, y=y, mode='markers', marker=settings)
    return markers


def plot_heat_maps(g, traj_or_trajs, occupancy_list, title_list,
        stars_grid=None, zmin=None, zmax=None, auto_logarithm=True, **kwargs):
    """
    traj_or_trajs: A trajectory (list of state-action pairs), or a list of
        trajectories.
    stars_grid: A list of lists. The ith list is a list of states on which to
        place stars in the ith heat map.
            OR a single list, which is used for every heat map.
    """
    subplot_list = []

    L = len(occupancy_list)

    if stars_grid == None:
        stars_grid = []
    else:
        stars_grid = list(stars_grid)

    try:
        iter(stars_grid[0])
    except TypeError:
        stars_grid = [stars_grid] * L
    assert len(stars_grid) == L

    try:
        # If iterable at two-deep, then this is a list of traj.
        iter(traj_or_trajs[0][0])
        trajs = traj_or_trajs
    except TypeError:
        # Assume this is a single trajectory.
        trajs = [traj_or_trajs] * L
    assert len(trajs) == L, (trajs, L)

    for o, stars, traj in zip(occupancy_list, stars_grid, trajs):
        o = o.reshape(g.rows, g.cols)
        data = []
        data.append(make_heat_map(g, o, zmin=zmin, zmax=zmax,
            auto_logarithm=auto_logarithm))
        data.append(make_line(g, traj))
        data.append(make_stars(g, stars))
        subplot_list.append(data)

    subplots(subplot_list, title_list, **kwargs)


def subplots(subplot_list, title_list, title=None, save_png=False,
        **kwargs):
    assert len(subplot_list) == len(title_list), (subplot_list, title_list)

    fig = tools.make_subplots(rows=1, cols=len(subplot_list),
            subplot_titles=title_list)
    fig['layout'].update(title=title)
    for i, subplot in enumerate(subplot_list):
        for t in subplot:
            fig.append_trace(t, 1, i+1)

    show_plot(fig, save_png, **kwargs)


uid_pointer = [100]
def show_plot(fig, save_png=False, delay=3.2):
    uid_pointer[0] += 1
    uid = uid_pointer[0]
    if not save_png:
        py.plot(fig, filename="output/out{}.html".format(uid))
    else:
        py.plot(fig, filename="output/out{}.html".format(uid),
            image='png', image_filename="output/out{}.png".format(uid),
            image_width=1400, image_height=750)
        if delay is not None:
            import time
            time.sleep(delay)


def _occ_starter(N, R, mode):
    """
    modes: diag, diag-top, vertical, diag-but-top
    """
    g = GridWorldMDP(N, N, {}, default_reward=R)
    one = g.coor_to_state(1,1)
    T = N+N
    if mode == "diag":
        start = one
        goal = model_goal = g.S - 1
    elif mode == "diag-top":
        start = one
        goal = model_goal = g.coor_to_state(N//2, N-1)
    elif mode == "vertical":
        start = g.coor_to_state(N//2, 0)
        goal = model_goal = g.coor_to_state(N//2, N-1)
    elif mode == "diag-but-diag-top":
        start = g.coor_to_state(N//2, 0)
        goal = g.coor_to_state(N//2, N-1)
        model_goal = g.S - 1
    return g, T, start, goal, model_goal


def _traj_starter(N, init_state, mode):
    A = Actions
    g = GridWorldMDP(N, N)
    one = g.coor_to_state(1,1)
    if mode == "diag":
        start = one
        actions = [A.UP_RIGHT] * (N-2)
    elif mode == "horizontal":
        start = g.coor_to_state(0, N//2)
        actions = [A.RIGHT] * (N-2)
    elif mode == "horizontal_origin":
        start = one
        actions = [A.RIGHT] * (N-2)
    elif mode == "vertical":
        start = g.coor_to_state(N//2, 0)
        actions = [A.UP] * (N-2)
    elif mode == "diag-crawl":
        start = one
        actions = [A.RIGHT] * (N-2) + [A.UP] * (N-2)
    elif mode == "diag-fickle":
        start = one
        w = (N-1)//2
        W = N - 1 - w
        actions = [A.RIGHT] * w + [A.UP_RIGHT] * (W-1) + \
                [A.UP] * w
    elif mode == "diag-fickle2":
        start = one
        w = (N-1)//2
        W = N - 1 - w
        actions = [A.UP_RIGHT] * (W-1) + [A.DOWN_RIGHT] * w \
                + [A.DOWN]
    else:
        raise Exception("invalid mode: {}".format(mode))
    return build_traj_from_actions(g, start, actions)


def _traj_beta_inf_loop(on_loop, g, traj, goal, inf_mod=inf_default, guess=1,
        min_beta=0.01, max_beta=100, verbose=True):
    for i in xrange(len(traj) + 1):
        if i == 0:
            start = traj[0][0]
            tr = [(start, Actions.ABSORB)]
            beta_hat = guess
        else:
            tr = traj[:i]
            beta_hat = inf_mod.beta.binary_search(g, tr, goal, guess=beta_hat,
                    verbose=verbose, min_beta=min_beta, max_beta=max_beta)
            if verbose:
                print "{}: beta_hat={}".format(i+1, beta_hat)
        on_loop(tr, round(beta_hat, 3), i)


def simple_ground_truth_inf(mode="diag", N=30, R=-1, true_beta=5,
        zmin=-5, zmax=0, inf_mod=inf_default, title=None,
        **kwargs):
    g, T, start, goal, model_goal = _occ_starter(N, R, mode)

    traj = simulate(g, start, goal, beta_or_betas=true_beta)
    beta_fixed = 1
    beta_hat = 1

    occ = inf_mod.occupancy
    def on_loop(traj, beta_hat, t):
        occupancies = occ.infer(g, traj, beta_or_betas=beta_hat,
                T=T, dest_or_dests=model_goal)
        fixed_occupancies = occ.infer(g, traj, beta_or_betas=beta_fixed,
                T=T, dest_or_dests=model_goal)
        true_occupancies = occ.infer(g, traj, beta_or_betas=true_beta,
                T=T, dest_or_dests=goal)

        occ_list = [occupancies, fixed_occupancies, true_occupancies]
        stars_grid = [[model_goal], [model_goal], [goal]]

        subplot_titles = (
                    "beta_hat={}".format(beta_hat),
                    "beta={}".format(beta_fixed),
                    "ground truth beta={}".format(true_beta),
                    )
        _title = title or "Euclid expected occupancies R={R}" + \
                " (for trajectories of length {T}) <br>t={t}"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, goal)


def simple_traj_inf(traj_or_traj_mode="diag", mode="diag", N=30, R=-1, title=None,
        inf_mod=inf_default, zmin=-5, zmax=0, **kwargs):
    # We don't care about model_goal. The star we show is always `goal`.
    g, T, start, goal, _ = _occ_starter(N, R, mode)
    if type(traj_or_traj_mode) is str:
        traj = _traj_starter(N, start, traj_or_traj_mode)
    else:
        traj = traj_or_traj_mode

    beta_hat = beta_fixed = 1

    occ = inf_mod.occupancy
    def on_loop(traj, beta_hat, t):
        occupancies = occ.infer(g, traj, beta_or_betas=beta_hat,
                T=T, dest_or_dests=goal)
        fixed_occupancies = occ.infer(g, traj, beta_or_betas=beta_fixed,
                T=T, dest_or_dests=goal)

        occ_list = [occupancies, fixed_occupancies]
        stars_grid = [goal]

        subplot_titles = (
                    "beta_hat={}".format(beta_hat),
                    "beta={}".format(beta_fixed))

        _title = title or "hardmax expected occupancies R={R}" + \
            " (for trajectories of length {T}) <br>t={t}"
        _title = _title.format(T=T, t=t, R=R)

        plot_heat_maps(g, traj, occ_list, subplot_titles, title=_title,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax, **kwargs)

    _traj_beta_inf_loop(on_loop, g, traj, goal)
