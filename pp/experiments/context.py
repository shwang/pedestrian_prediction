# TODO: merge scenarios and context starter files
from scenarios import scenario_starter

class HRContext(object):
    def __init__(self, g_H=None, g_R=None, traj_H=None, traj_R=None,
            start_H=None, start_R=None, goal_H=None, goal_R=None,
            collide_radius=None, collide_penalty=None, N=None,
            mode=None,
            ):
        self.g_H = g_H
        self.g_R = g_R
        self.traj_H = traj_H
        self.traj_R = traj_R
        self.start_H = start_H
        self.start_R = start_R
        self.goal_H = goal_H
        self.goal_R = goal_R
        self.collide_radius = collide_radius
        self.collide_penalty = collide_penalty
        self.N = N
        self.mode = mode


    def cfg_mode(self, mode, N):
        self.mode = mode
        self.N = N
        self.g_H, self.start_H, self.traj_H, self.g_R, self.start_R = \
                scenario_starter(mode, N)
        self.goal_H, self.goal_R = self.g_H.goal, self.g_R.goal
        return self
