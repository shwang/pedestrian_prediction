import numpy as np

from robot_planner import CollideProbs, robot_planner
from scenarios import scenario_starter
from ..plot import common as plot

def inspect_probs():
    N=8
    g_H, start_H, traj_H, g_R, start_R = scenario_starter("criss-cross", N=N)
    T=20

    radius, penalty = 3, 5
    cp = CollideProbs(g_H, T=T, collide_radius=radius, traj=traj_H[:0], beta=0.1,
            start_R=start_R, start_H=start_H)

    P = np.empty([T, g_H.S])
    for t in range(T):
        for s in range(g_H.S):
            P[t, s] = cp.get(t, s)

    # np.set_printoptions(precision=5, suppress=True, linewidth=200)
    # for t in range(T):
    #     print(t)
    #     print(P[t].reshape(N, N))
    #     print ""

    # for t in range(T):
    #     # Consider nonlogatrithm
    #     hm = plot.make_heat_map(g_H, P[t], auto_logarithm=False,
    #             zmax=1, zmin=0)
    #     plot.show_plot([hm])


    plan = robot_planner(g_R, start_R, g_H, start_H=start_H,
            collide_radius=radius, collide_penalty=penalty)
    for s, a in plan:
        print("{}, {!s}".format(g_R.state_to_coor(s), g_R.Actions(a)))

    import pdb; pdb.set_trace()

    for t in range(T):
        hm = plot.make_heat_map(g_H, P[t], auto_logarithm=False, zmax=1, zmin=0)
        line = plot.make_line(g_H, plan[t+1:], color='green', dash='dash')
        plot.show_plot([hm, line])
