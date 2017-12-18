from __future__ import division

import numpy as np

from pp.mdp import GridWorldMDP
from pp.inference import hardmax as inf
from pp.plot import plot_heat_maps

A = GridWorldMDP.Actions

def andrea_states():
    """
    Infer `T` hardmax state probability grids, one for each timestep.
    """
    T = 5
    N = 20
    R = -1
    beta = 1
    g = GridWorldMDP(N, N, default_reward=R)

    init_state = g.coor_to_state(0, 0)
    goal = g.coor_to_state(N-1, N//2)

    # A numpy.ndarray with dimensions (T x g.rows x g.cols).
    # `state_prob[t]` holds the exact state probabilities for
    # a beta-irrational, softmax-action-choice-over-hardmax-values
    # agent.
    state_prob = inf.state.infer_from_start(g, init_state, goal,
            T=T, beta=beta, all_steps=True).reshape(T+1, g.rows, g.cols)
    print(state_prob)

    # Plot each of the T heatmaps
    # beware: heat map's color scale changes with each plot
    for t, p in enumerate(state_prob):
        title = "t={}".format(t)
        plot_heat_maps(g, init_state, [p], [title], stars_grid=[goal],
                auto_logarithm=False)

andrea_states()

from pp.plot.common_multi import _occ_starter, _traj_starter
def benchmark(traj_mode="diag", mode="tri", T=2, N=90, R=-1):
    g = GridWorldMDP(N, N, default_reward=R)

    g, _, start, dest_list = _occ_starter(N, R, mode)
    traj = _traj_starter(N, start, traj_mode)[:50]

    def test():
        D = inf.occupancy.infer(g, traj, dest_list, T=T, verbose=False)

    test()
    import cProfile
    cProfile.runctx('test()', globals(), locals())

benchmark()
