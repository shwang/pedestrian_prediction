from __future__ import division

from enum import IntEnum
import numpy as np
from hardmax import forwards_value_iter as _value_iter
from .mdp import MDP

class Actions(IntEnum):
    # Two humans actions
    action_strings = ["UP", "DOWN", "LEFT", "RIGHT", "UL", "UR", "DL", "DR", "ABSORB"]
    for i1, a1 in enumerate(action_strings):
        for i2, a2 in enumerate(action_strings):
            a = a1 + "_" + a2
            value = i1 * len(action_strings) + i2
            exec(a + " = {}".format(value))

class ActionsSingle(IntEnum):
    # Single human actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    ABSORB = 8   

diagonal_actions = {ActionsSingle.UP_LEFT, ActionsSingle.UP_RIGHT, 
    ActionsSingle.DOWN_LEFT, ActionsSingle.DOWN_RIGHT}

# Classic Gridworld with 2 humans
class GridWorldMDP2H(MDP):
    Actions = Actions
    ActionsSingle = ActionsSingle

    def __init__(self, rows, cols, goal_state=None, euclidean_rewards=True,
            allow_wait=False, **kwargs):
        """
        An agent in a GridWorldMDP can move between adjacent/diagonal cells.

        If the agent chooses an illegal action it receives a float('-inf')
        reward and will stay in place.

        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            goal_state [int]: (optional) The goal state at which ABSORB is legal
                and costs 0.
            euclidean_rewards [bool]: (optional) If True, then scale rewards for
                moving diagonally by sqrt(2).
            allow_wait [bool]: (optional) If False, then the ABSORB action is
                illegal in all states except the goal. If True, then the ABSORB
                action costs default_reward in states other than the goal.
        """
        if goal_state is not None:
            assert isinstance(goal_state, int)

        self.allow_wait = allow_wait

        assert rows > 0
        assert cols > 0
        assert isinstance(rows, int)
        assert isinstance(cols, int)

        self.X = rows
        self.Y = cols
        S = (rows ** 2) * (cols ** 2)
        A = len(Actions)

        MDP.__init__(self, S=S, A=A, transition_helper=transition_helper, **kwargs)
        self.q_cache = {}

        if euclidean_rewards:
            for a in range(A):
                a1 = a // len(ActionsSingle)
                a2 = a % len(ActionsSingle)
                if a1 in diagonal_actions or a2 in diagonal_actions:
                    col = self.rewards[:, a]
                    np.multiply(col, np.sqrt(2), out=col)
                if a1 in diagonal_actions and a2 in diagonal_actions:
                    col = self.rewards[:, a]
                    np.multiply(col, 2, out=col)
                if self.allow_wait and (a1 == ActionsSingle.ABSORB or a2 == ActionsSingle.ABSORB):
                    self.rewards[:, a].fill(self.default_reward)
                else:
                    self.rewards[:, a].fill(-np.inf)

    # XXX: optimize so that we don't need to convert between state and coor.
    def transition_helper(self, s, a, alert_illegal=False):
        r1, c1, r2, c2 = self.state_to_coor(s)
        assert a >= 0 and a < len(Actions), a
        a1 = a // len(ActionsSingle)
        a2 = a % len(ActionsSingle)

        r1_prime, c1_prime, illegal1 = self.transition_one(r1, c1, a1)
        r2_prime, c2_prime, illegal2 = self.transition_one(r2, c2, a2)
        s_prime = self.coor_to_state(r1_prime, c1_prime, r2_prime, c2_prime)

        if alert_illegal:
            return s_prime, (illegal1 or illegal2)
        else:
            return s_prime

    def transition_one(self, r, c, a):
        r_prime, c_prime = r, c
        if a == ActionsSingle.LEFT:
            r_prime = r - 1
        elif a == ActionsSingle.RIGHT:
            r_prime = r + 1
        elif a == ActionsSingle.DOWN:
            c_prime = c - 1
        elif a == ActionsSingle.UP:
            c_prime = c + 1
        elif a == ActionsSingle.UP_LEFT:
            r_prime, c_prime = r - 1, c + 1
        elif a == ActionsSingle.UP_RIGHT:
            r_prime, c_prime = r + 1, c + 1
        elif a == ActionsSingle.DOWN_LEFT:
            r_prime, c_prime = r - 1, c - 1
        elif a == ActionsSingle.DOWN_RIGHT:
            r_prime, c_prime = r + 1, c - 1
        elif a == ActionsSingle.ABSORB:
            pass
        else:
            raise BaseException("undefined action {}".format(a))

        illegal = False
        if r_prime < 0 or r_prime >= self.X or c_prime < 0 or c_prime >= self.Y:
            r_prime, c_prime = r, c
            illegal = True

        return r_prime, c_prime, illegal

    def q_values(self, goal_spec, forwards_value_iter=_value_iter,
            goal_stuck=False):
        """
        Calculate the hardmax Q values for each state action pair.
        For GridWorldMDPs, the goal_spec is simply the goal state.

        Params:
            goal_spec [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If this is True, then all actions other than
                ABSORB are illegal in the goal state.

        Returns:
            Q [np.ndarray]: An SxA array containing the q values
                corresponding to each (s, a) pair.
        """
        if (goal_spec, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_spec, goal_stuck)])

        V = forwards_value_iter(self, goal_spec)

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in range(self.S):
            if s == goal_spec and goal_stuck:
                Q[s, Actions.ABSORB] = 0
                # All other actions will be -np.inf by default.
                continue

            for a in range(self.A):
                if s == goal_spec and a == Actions.ABSORB:
                    Q[s, a] = 0
                else:
                    Q[s,a] = self.rewards[s,a] + V[self.transition(s,a)]
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_spec, goal_stuck)] = Q
        return np.copy(Q)

    def coor_to_state(self, r1, c1, r2, c2):
        """
        Params:
            r1 & r2 [int]: The state's row for human 1 and 2.
            c1 & c2 [int]: The state's column for human 1 and 2.

        Returns:
            s [int]: The state number associated with the given coordinates in
                a standard grid world.
        """
        assert 0 <= r1 < self.X, "invalid (rows, r)={}".format((self.X, r1))
        assert 0 <= c1 < self.Y, "invalid (cols, c)={}".format((self.Y, c1))
        assert 0 <= r2 < self.X, "invalid (rows, r)={}".format((self.X, r2))
        assert 0 <= c2 < self.Y, "invalid (cols, c)={}".format((self.Y, c2))
        return np.ravel_multi_index((r1, c1, r2, c2), (self.X, self.Y, self.X, self.Y))

    def state_to_coor(self, s):
        """
        Params:
            s [int]: The state.

        Returns:
            r1, c1, r2, c2 [int]: The row and column associated with state s for each human.
        """
        assert s < self.X * self.Y * self.X * self.Y
        inds = np.unravel_index(s, (self.X, self.Y, self.X, self.Y))
        return inds[0], inds[1], inds[2], inds[3]