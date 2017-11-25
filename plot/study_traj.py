import numpy as np

from .common import *
from mdp import GridWorldMDP
from util import sum_rewards, display, build_traj_from_actions
from util.hardmax import simulate, sample_action
from itertools import izip

from parameters import inf_default

A = GridWorldMDP.Actions

def beta_versus(g, start, actions, goal, beta1, beta2, uid=0, title=None,
        inf_mod=inf_default):
    traj = build_traj_from_actions(g, start, actions)
    o1 = infer_occupancies(g, traj, beta=beta1,
        dest_set={goal}).reshape(g.rows, g.cols)
    data1 = output_heat_map(g, o1, traj, start, {goal}, beta_hat=beta1, zmin=-10)

    o2 = infer_occupancies(g, traj, beta=beta2,
        dest_set={goal}).reshape(g.rows, g.cols)
    data2 = output_heat_map(g, o2, traj, start, {goal}, beta_hat=beta2, zmin=-10)

    o_diff = np.abs(o1 - o2)
    data3 = output_heat_map(g, o_diff, traj, start, {goal}, beta_hat=beta1, zmin=-10)

    subplot_titles=(
        "expected occupancies, beta={}".format(beta1),
        "expected occupancies, beta={}".format(beta2),
        "abs difference in ex. occupancies"))


def shortest_paths_beta_hat(N=5, R=-0.3, min_beta=0.1, max_beta=1000,
        inf_mod=inf_default):
    """
    Output a heat map showing the beta_hat that would result from taking the
    shortest path to a given state.
    """
    bt = inf_mod.beta

    g = GridWorldMDP(N, N, default_reward=R)
    start = 0
    goal = g.coor_to_state(N//2, N-1)

    beta_hats = np.zeros(N*N)
    beta_hats[0] = np.nan
    for s in range(1, N*N):
        traj = simulate(g, 0, s, beta=0.1)[:-1]
        beta_hats[s] = bt.binary_search(g, traj, goal, guess=1,
                verbose=False, min_beta=min_beta, max_beta=max_beta)

    title = "beta estimate for shortest path to each square"
    plot_heat_maps(g, start, beta_hats, [title], zmin=min_beta, zmax=max_beta,
            stars_grid=[goal], auto_logarithm=False,
            z_min=min_beta, z_max=max_beta)


def histogram_beta_est(N=30, R=-20, true_beta=10, min_beta=0.01, max_beta=100,
        inf_mod=inf_default):
    """Plot the frequency of beta_hat over every various simulated trajectories"""
    import plotly.graph_objs as go
    g = GridWorldMDP(N, N, {}, default_reward=R)
    start = 0
    goal = model_goal = g.coor_to_state(N//2, N-1)
    model_goal = goal

    bt = inf_mod.beta

    beta_hats=[]
    for i in range(200):
        trajectory = simulate(g, start, goal, beta=true_beta)
        beta = bt.binary_search(g, trajectory, model_goal,
                verbose=True, min_beta=min_beta, max_beta=max_beta)
        beta_hats.append(beta)

    data = [go.Histogram(x=beta_hats, xbins=dict(
        start=-4, end=16, size=0.5),)]
    show_plot(data)

    mean = np.sum(beta_hats)
    mean_sq = np.sum(np.square(beta_hats))
    variance = mean_sq - mean*mean
    print "mean={}, variance={}".format(mean, variance)


def plot_traj_log_likelihood(g, traj, goal, title=None,
        beta_min=0.2, beta_max=8.0, beta_step=0.05, plot=True,
        inf_mod=inf_default, verbose=True):

    bt = inf_mod.beta

    x = np.arange(beta_min, beta_max, beta_step)
    scores = [bt.compute_score(g, traj, goal, beta) for beta in x]

    beta_hat = bt.binary_search(g, traj, goal, guess=1, verbose=True)
    beta_hat_score = bt.compute_score(g, traj, goal, beta_hat)

    import plotly.offline as py
    import plotly.graph_objs as go
    from plotly import tools

    fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=(
                "log likelihood of trajectory",
                "gradient of log likelihood"
                ))
    fig['layout'].update(title=title, xaxis=dict(title="beta"))

    trace1 = go.Scatter(name="log likelihood", x=x, y=scores, mode='markers')
    fig.append_trace(trace1, 1, 1)

    trace2 = go.Scatter(name="beta_hat log likelihood",
            x=[beta_hat], y=[beta_hat_score], mode='markers')
    fig.append_trace(trace2, 1, 1)

    grads = [bt.compute_grad(g, traj, goal, beta) for beta in x]
    trace3 = go.Scatter(name="gradient of log likelihood", x=x, y=grads, mode='markers')
    fig.append_trace(trace3, 2, 1)

    if verbose:
        print "estimated beta={}".format(beta_hat)
    show_plot(fig)
