from __future__ import division

import numpy as np

from ..mdp import GridWorldMDP
from ..util import build_traj_from_actions


def scenario_starter(mode, N, reward_human=-1, reward_robot=-1):
    A = GridWorldMDP.Actions
    g_H = GridWorldMDP(N, N, default_reward=reward_human, allow_wait=True)
    g_R = GridWorldMDP(N, N, default_reward=reward_robot, allow_wait=False)
    coor = lambda x, y: g_H.coor_to_state(x, y)

    c_x = c_y = N//2
    center = coor(c_x, c_y)
    center_left = coor(1, c_y)
    center_right = coor(N-2, c_y)
    center_top = coor(c_x, N-2)
    center_bot = coor(c_x, 1)
    top_right = coor(N-2, N-2)
    top_right_corner = coor(N-1, N-1)
    top_left = coor(1, N-2)
    bot_left = one = coor(1, 1)
    bot_right = coor(N-2, 1)

    if mode == "diag only":
        # no difference with collision_penalty=10
        n = (N-3)//2
        actions = [A.UP_RIGHT] * n + [A.UP_LEFT] * n
        start_H, goal_H = bot_left, coor(1, 1+2*n)
        s = int(0.7 * N)
        start_R, goal_R = coor(s, s), bot_left
    elif mode == "invisible wall":
        s = int(0.3 * N)
        start_H, goal_H = coor(N - s, 1), bot_left
        start_R, goal_R = coor(N-1, s), coor(1, s)
        w = N - 5 - s//2
        W = N - 5 - w
        actions = [A.LEFT] * w + [A.UP]*s + [A.LEFT]*2 + \
                [A.DOWN]*s  # + [A.LEFT] * W
    elif mode == "zigzag":
        s = int(0.4 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = top_right, bot_left
        actions = [A.LEFT]*(s-1) + [A.UP_RIGHT]*((N-1)//2) +\
                [A.UP_LEFT]*((N-1)//2)
    elif mode == "hug walls 2":
        start_H, goal_H = bot_right, top_left
        s = int(0.7 * N)
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.LEFT]*(N-3) + [A.UP]*(N-3)
    elif mode == "hug walls 3":
        s = int(0.7 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.LEFT]*(s-1) + [A.UP]*(s-1)
    elif mode == "hug walls 4":
        s = int(0.7 * N)
        start_H, goal_H = coor(s, 1), top_left
        start_R, goal_R = coor(s, s), bot_left
        actions = [A.DOWN_LEFT] + [A.LEFT]*(s-2) + [A.UP]*(s-2) + \
                [A.UP_RIGHT]
    elif mode == "obtuse":
        start_H, goal_H = bot_left, center_left
        start_R, goal_R = top_right, bot_left
        n = (N-3)//2
        actions = [A.UP_RIGHT] * n + [A.RIGHT] * n
    elif mode == "beta_lag":
        start_H, goal_H = center_left, center_top
        start_R, goal_R = top_right, bot_left
        actions = [A.RIGHT] * (N-3)
    elif mode == "beta_lag 2":
        start_H, goal_H = center_left, coor(N-2, N-1)
        start_R, goal_R = top_right, bot_left
        actions = [A.RIGHT] * (N-3)
    elif mode == "treacherous turn":
        start_H, goal_H = center_left, center_right
        start_R, goal_R = top_right_corner, bot_left
        n = (N-3)//2
        # TODO: Change back to UP_RIGHT with `u`
        actions = [A.RIGHT] * n + [A.UP] * n
    elif mode == "sync vert":
        start_H, goal_H = coor(c_x + 1, N-2), coor(c_x + 1, 1)
        start_R, goal_R = coor(c_x - 1, N-2), coor(c_x - 1, 1)
        actions = [A.DOWN] * (N-3)
    elif mode == "sync vert antiparallel":
        # Equal reward results: for N=15, radius=1
        start_H, goal_H = coor(c_x + 1, N-2), coor(c_x + 1, 1)
        start_R, goal_R = coor(c_x - 1, 1), coor(c_x - 1, N-2)
        actions = [A.DOWN] * (N-3)
    elif mode == "cross":
        start_H, goal_H = center_right, top_right
        start_R, goal_R = center_top, center_bot
        actions = [A.LEFT] * int(.9 * N)
    elif mode == "shuffle":
        # AKA awkward shuffle
        # good example. (similar to diag shuffle)
        start_H, goal_H = center, center_left
        start_R, goal_R = coor(c_x, c_y + 4), center_bot
        actions = [A.RIGHT] * int(.8 * N / 2)
    elif mode == "shuffle 2":
        # good example -- N=17, radius=2
        start_H, goal_H = coor(c_x, c_y+1), center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.DOWN] + [A.RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "head-on":
        # N=12, collide_radius=1 ==> better for fixed.
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP] * (int(.8 * N / 2) - 2)
    elif mode == "diag shuffle":
        # good example -- N=17, radius=2
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP_RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "diag shuffle mod":
        start_H, goal_H = center, center_left
        start_R, goal_R = center_top, center_bot
        actions = [A.UP_RIGHT] * (int(.8 * N / 2) - 2)
    elif mode == "actually criss-cross":
        start_H, goal_H = one, coor(N//2, N-2)
        start_R, goal_R = coor(N-2, 1), coor(0, N-2)
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "criss-cross" or mode == "diag-cross":
        start_H, goal_H = one, coor(N-2, N-2)
        start_R, goal_R = coor(N-2, 1), coor(0, N-2)
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "criss-cross false":
        # Bad experiment -- when human diverges, human walks perfectly to avoid
        # collision
        start_H, goal_H = one, coor(N-2, N-2)
        start_R, goal_R = coor(N-2, 1), coor(0, N-2)
        w = (N-2) // 2
        W = N - 2 - w
        actions = [A.UP_RIGHT] * w + [A.DOWN_RIGHT] * W
    elif mode == "swap corners":
        start_H, goal_H = one, coor(N-2, N-2)
        start_R, goal_R = goal_H, start_H
        actions = [A.UP_RIGHT] * (N-3)
    elif mode == "swap corners false":
        start_H, goal_H = one, coor(N-2, N-2)
        start_R, goal_R = goal_H, start_H
        w = (N-2) // 2
        W = N - 2 - w
        actions = [A.RIGHT] * w + [A.DOWN_RIGHT] * 1
    else:
        raise Exception("invalid mode: {}".format(mode))

    g_R.set_goal(goal_R)
    g_H.set_goal(goal_H)
    traj_H = build_traj_from_actions(g_H, start_H, actions)

    return g_H, start_H, traj_H, g_R, start_R
