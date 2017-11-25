import numpy as np

from mdp import GridWorldMDP
from inference.softmax.destination import infer_destination
from inference import hardmax

from util import sum_rewards, display, build_traj_from_actions
from util.hardmax import simulate, sample_action

from parameters import inf_default

Actions = GridWorldMDP.Actions


def make_heat_map(g, occupancies, traj_or_start_state, stars,
        zmin=None, zmax=0, auto_logarithm=True):
    import plotly.graph_objs as go
    data = []

    o = occupancies.T
    if auto_logarithm:
        o = np.log(o)
    o[o == -np.inf] = -9999
    hm = go.Heatmap(z=o, zmin=zmin, zmax=zmax, name="log expected occupancy")
    data.append(hm)

    try:
        states = [s for s, a in traj_or_start_state]
        states.append(g.transition(*traj_or_start_state[-1]))
    except TypeError:
        states = [traj_or_start_state]
    coors = [g.state_to_coor(s) for s in states]
    x, y = zip(*coors)
    traj_line = dict(x=x, y=y, line=dict(color='white', width=3))
    data.append(traj_line)

    if len(stars) > 0:
        x, y = zip(*[g.state_to_coor(s) for s in stars])
        dest_markers = go.Scatter(x=x, y=y,
            mode='markers', marker=dict(size=20, color="white", symbol="star"))
        data.append(dest_markers)

    return data


def plot_heat_maps(g, traj_or_start_state, occupancy_list, title_list,
        stars_grid=None, zmin=None, zmax=None, auto_logarithm=True, **kwargs):
    """
    stars_grid: A list of lists. The ith list is a list of states on which to place
        stars in the ith heat map.
            OR a single list, which is used for every heat map.
    """
    subplot_list = []

    if stars_grid == None:
        stars_grid = []

    stars_grid = list(stars_grid)
    try:
        iter(stars_grid[0])
    except:
        stars_grid = [stars_grid] * len(occupancy_list)

    for o, stars in zip(occupancy_list, stars_grid):
        o = o.reshape(g.rows, g.cols)
        trace = make_heat_map(g, o, traj_or_start_state, stars,
                zmin=zmin, zmax=zmax, auto_logarithm=auto_logarithm)
        subplot_list.append(trace)

    subplots(subplot_list, title_list, **kwargs)


def subplots(subplot_list, title_list, title=None, save_png=False):
    assert len(subplot_list) == len(title_list), (subplot_list, title_list)
    from plotly import tools as tools

    fig = tools.make_subplots(rows=1, cols=len(subplot_list),
            subplot_titles=title_list)
    fig['layout'].update(title=title)
    for i, subplot in enumerate(subplot_list):
        for t in subplot:
            fig.append_trace(t, 1, i+1)

    show_plot(fig, save_png)


def show_plot(fig, save_png=False, uid=100):
    import plotly.offline as py
    uid += 1
    if not save_png:
        py.plot(fig, filename="output/out{}.html".format(uid))
    else:
        py.plot(fig, filename="output/out{}.html".format(uid),
            image='png', image_filename="output/out{}.png".format(uid),
            image_width=1400, image_height=750)


def _occ_starter(N, R, mode):
    """
    modes: diag, diag-top, vertical, diag-but-top
    """
    g = GridWorldMDP(N, N, {}, default_reward=R)
    T = N+N
    if mode == "diag":
        start = 0
        goal = model_goal = g.S - 1
    elif mode == "diag-top":
        start = 0
        goal = model_goal = g.coor_to_state(N//2, N-1)
    elif mode == "vertical":
        start = g.coor_to_state(N//2, 0)
        goal = model_goal = g.coor_to_state(N//2, N-1)
    elif mode == "diag-but-diag-top":
        start = g.coor_to_state(N//2, 0)
        goal = g.coor_to_state(N//2, N-1)
        model_goal = g.S - 1
    return g, T, start, goal, model_goal


def _traj_beta_inf_loop(on_loop, g, traj, goal, inf_mod=inf_default, guess=1,
        min_beta=0.01, max_beta=100, verbose=True):
    for i in xrange(len(traj)):
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
        on_loop(tr, beta_hat, i)


def simple_ground_truth_inf(mode="diag", N=30, R=-3, true_beta=0.001,
        min_beta=0.01, max_beta=100, zmin=-5, zmax=0, inf_mod=inf_default,
        title=""):
    g, T, start, goal, model_goal = _occ_starter(N, R, mode)

    traj = simulate(g, start, goal, beta=true_beta)
    beta_fixed = 1
    beta_hat = 1

    occ = inf_mod.occupancy
    def on_loop(traj, beta_hat, t):
        occupancies = occ.infer(g, traj, beta=beta_hat, T=T, dest=model_goal)
        fixed_occupancies = occ.infer(g, traj, beta=beta_fixed, T=T, dest=model_goal)
        true_occupancies = occ.infer(g, traj, beta=true_beta, T=T, dest=goal)

        occ_list = [occupancies, fixed_occupancies, true_occupancies]
        stars_grid = [[model_goal], [model_goal], [goal]]

        subplot_titles = (
                    "beta_hat={}".format(beta_hat),
                    "beta={}".format(beta_fixed),
                    "ground truth beta={}".format(true_beta),
                    )
        title = ("Hardmax expected occupancies (for trajectories of length {T}) <br>t={t}"
            ).format(T=T, t=t)

        plot_heat_maps(g, traj, occ_list, subplot_titles, stars_grid=stars_grid,
                zmin=zmin, zmax=zmax)

    _traj_beta_inf_loop(on_loop, g, traj, goal)


def simple_traj_inf(traj, stars, N=30, R=-3, title=None, inf_mod=inf_default):
    # We don't care about model_goal. The star we show is always `goal`.
    g, T, start, goal, _ = _occ_starter(N, R, mode)
    beta_hat = beta_fixed = 1

    occ = inf_mod.occupancy
    def on_loop(tr, beta_hat, t):
        occupancies = occ.infer(g, traj, beta=beta_hat, T=T, dest=model_goal)
        fixed_occupancies = occ.infer(g, traj, beta=beta_fixed, T=T, dest=model_goal)

        occ_list = [occupancies, fixed_occupancies]
        stars_grid = [goal]

        subplot_titles = (
                    "beta_hat={}".format(beta_hat),
                    "beta={}".format(beta_fixed),
                    "ground truth beta={}".format(true_beta),
                    )
        title = ("Hardmax expected occupancies (for trajectories of length {T}) <br>t={t}"
            ).format(T=T, t=t)

        plot_heat_maps(g, tr, occ_list, subplot_titles,
                stars_grid=stars_grid, zmin=zmin, zmax=zmax)

    _traj_beta_inf_loop(on_loop, g, traj)
