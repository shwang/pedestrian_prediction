from __future__ import division

import numpy as np
from .mdp import MDP
from .classic import GridWorldMDP, transition_helper
import gridless
from enum import IntEnum

OldAct = GridWorldMDP.Actions
granularity = 15
class Actions(IntEnum):
    M345 = 23; M0 = 0; M15 = 1  # RIGHT
    M30 = 2; M45 = 3; M60 = 4;  # UP_RIGHT
    M75 = 5; M90 = 6; M105 = 7  # UP
    M120 = 8; M135 = 9; M150 = 10  # UP_LEFT
    M165 = 11; M180 = 12; M195 = 13  # LEFT
    M210 = 14; M225 = 15; M240 = 16  # DOWN_LEFT
    M255 = 17; M270 = 18; M285 = 19  # DOWN
    M300 = 20; M315 = 21; M330 = 22  # DOWN_RIGHT
    ABSORB = 24  # special

# Map expanded actionspace to reduced actionspace.
def build_action_map():
    m = [0] * len(Actions)
    A = Actions
    m[A.M345] = m[A.M0] = m[A.M15] = OldAct.RIGHT
    m[A.M30] = m[A.M45] = m[A.M60] = OldAct.UP_RIGHT
    m[A.M75] = m[A.M90] = m[A.M105] = OldAct.UP
    m[A.M120] = m[A.M135] = m[A.M150] = OldAct.UP_LEFT
    m[A.M165] = m[A.M180] = m[A.M195] = OldAct.LEFT
    m[A.M210] = m[A.M225] = m[A.M240] = OldAct.DOWN_LEFT
    m[A.M255] = m[A.M270] = m[A.M285] = OldAct.DOWN
    m[A.M300] = m[A.M315] = m[A.M330] = OldAct.DOWN_RIGHT
    m[A.ABSORB] = OldAct.ABSORB
    return m
action_map = build_action_map()

class GridWorldExpanded(MDP):
    Actions = Actions

    def __init__(self, rows, cols, **kwargs):
        MDP.__init__(self, rows=rows, cols=cols, A=len(Actions),
                transition_helper=self._transition_helper,
                default_reward=np.nan, **kwargs)

    def _transition_helper(self, s, a, alert_illegal=False):
        old_a = action_map[a]
        return transition_helper(self, s, old_a, alert_illegal=alert_illegal)

    def q_values(self, goal_state, goal_stuck=False, beta=1):
        """
        Calculate the euclidean Q values for each state action pair.
        """
        if (goal_state, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_state, goal_stuck)])

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in range(self.S):
            s_coor = self.state_to_real_coor(s)
            g_coor = self.state_to_real_coor(goal_state)

            Q[s] = -gridless.circle_dists(center=s_coor, dest=g_coor,
                    W=self.rows, H=self.cols)
            if s == goal_state:
                if goal_stuck:
                    Q[s] = -np.inf
                Q[s, Actions.ABSORB] = 0
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_state, goal_stuck)] = Q
        return np.copy(Q)

    def state_to_real_coor(self, s):
        x, y = self.state_to_coor(s)
        return np.array([x + 0.5, y + 0.5])
