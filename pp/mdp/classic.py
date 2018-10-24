from __future__ import division

from enum import IntEnum
import numpy as np
from hardmax import forwards_value_iter as _value_iter
from .mdp import MDP

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    ABSORB = 8

diagonal_actions = {Actions.UP_LEFT, Actions.UP_RIGHT, Actions.DOWN_LEFT,
        Actions.DOWN_RIGHT}

# XXX: optimize so that we don't need to convert between state and coor.
def transition_helper(g, s, a, alert_illegal=False):
    r, c = g.state_to_coor(s)
    assert a >= 0 and a < len(Actions), a

    r_prime, c_prime = r, c
    if a == Actions.LEFT:
        r_prime = r - 1
    elif a == Actions.RIGHT:
        r_prime = r + 1
    elif a == Actions.DOWN:
        c_prime = c - 1
    elif a == Actions.UP:
        c_prime = c + 1
    elif a == Actions.UP_LEFT:
        r_prime, c_prime = r - 1, c + 1
    elif a == Actions.UP_RIGHT:
        r_prime, c_prime = r + 1, c + 1
    elif a == Actions.DOWN_LEFT:
        r_prime, c_prime = r - 1, c - 1
    elif a == Actions.DOWN_RIGHT:
        r_prime, c_prime = r + 1, c - 1
    elif a == Actions.ABSORB:
        pass
    else:
        raise BaseException("undefined action {}".format(a))

    illegal = False
    if r_prime < 0 or r_prime >= g.rows or \
            c_prime < 0 or c_prime >= g.cols:
        r_prime, c_prime = r, c
        illegal = True

    s_prime = g.coor_to_state(r_prime, c_prime)

    if alert_illegal:
        return s_prime, illegal
    else:
        return s_prime


# TODO: rename this class into GridWorldMDP and rename GridWorldMDP to
# GridWorldMDPClassic.
class MDP2D(MDP):
    def __init__(self, rows, cols, A, reward_dict={}, **kwargs):
        """
        Superclass for GridWorldMDP and GridWorldExpanded.

        Params:
            rows [int]: The number of rows.
            cols [int]: The number of columns.
            A [int]: The number of actions.

        Debug Params (mainly used in unittests):
            reward_dict [dict]: Maps state `s_new` to reward `R`. Passing in a
                nonempty dict for this parameter will make any legal
                state-action pair that transitions to `s_new` yield the reward
                `R`.
        """
        assert rows > 0
        assert cols > 0
        assert isinstance(rows, int)
        assert isinstance(cols, int)

        # TODO: Rename rows=> X and rename cols=> Y. The current naming
        # convention is confusing.
        self.rows = rows
        self.cols = cols
        S = rows * cols

        # Convert from coordinates to state number as required by super-class
        reward_dict = {self.coor_to_state(x, y): R
                for (x, y), R in reward_dict.items()}

        MDP.__init__(self, S=S, A=A, reward_dict=reward_dict, **kwargs)
        self.q_cache = {}


    def coor_to_state(self, r, c):
        """
        Params:
            r [int]: The state's row.
            c [int]: The state's column.

        Returns:
            s [int]: The state number associated with the given coordinates in
                a standard grid world.
        """
        assert 0 <= r < self.rows, "invalid (rows, r)={}".format((self.rows, r))
        assert 0 <= c < self.cols, "invalid (cols, c)={}".format((self.cols, c))
        return r * self.cols + c

    def state_to_coor(self, s):
        """
        Params:
            s [int]: The state.

        Returns:
            r, c [int]: The row and column associated with state s.
        """
        assert s < self.rows * self.cols
        return s // self.cols, s % self.cols


# Classic Gridworld
class GridWorldMDP(MDP2D):
    Actions = Actions

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

        MDP2D.__init__(self, rows=rows, cols=cols, A=len(Actions),
                transition_helper=self._transition_helper, **kwargs)

        if euclidean_rewards:
            for a in diagonal_actions:
                col = self.rewards[:, a]
                np.multiply(col, np.sqrt(2), out=col)

        if self.allow_wait:
            self.rewards[:, Actions.ABSORB].fill(self.default_reward)
        else:
            self.rewards[:, Actions.ABSORB].fill(-np.inf)

    # XXX: optimize so that we don't need to convert between state and coor.
    def _transition_helper(self, s, a, alert_illegal=False):
        return transition_helper(self, s, a, alert_illegal=alert_illegal)

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
