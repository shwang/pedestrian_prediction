from __future__ import division

from enum import IntEnum
import numpy as np
from hardmax import forwards_value_iter as _value_iter
from .mdp import MDP

class Actions(IntEnum):
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

diagonal_actions = {Actions.UP_LEFT, Actions.UP_RIGHT, 
    Actions.DOWN_LEFT, Actions.DOWN_RIGHT}

# XXX: optimize so that we don't need to convert between state and coor.
def transition_helper(g, s, a, alert_illegal=False):
    def transition_single(r, c, a):
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
        if r_prime < 0 or r_prime >= np.sqrt(g.X) or c_prime < 0 or c_prime >= np.sqrt(g.Y):
            r_prime, c_prime = r, c
            illegal = True

        return r_prime, c_prime, illegal

    r, c = g.state_to_coor(s)
    r1, c1, r2, c2 = g.coor2D_to_coor4D(r, c)

    assert a >= 0 and a < len(Actions) ** 2, a
    a1 = a // len(Actions)
    a2 = a % len(Actions)

    r1_prime, c1_prime, illegal1 = transition_single(r1, c1, a1)
    r2_prime, c2_prime, illegal2 = transition_single(r2, c2, a2)
    r_prime, c_prime = g.coor4D_to_coor2D(r1_prime, c1_prime, r2_prime, c2_prime)
    s_prime = g.coor_to_state(r_prime, c_prime)

    if alert_illegal:
        return s_prime, (illegal1 or illegal2)
    else:
        return s_prime

# MDP for 2 humans, so 4D.
class MDP4D(MDP):
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
        self.X = rows ** 2
        self.Y = cols ** 2
        S = self.X * self.Y
        A = A ** 2

        # Convert from coordinates to state number as required by super-class
        reward_dict = {self.coor_to_state(x, y): R
                for (x, y), R in reward_dict.items()}

        MDP.__init__(self, S=S, A=A, reward_dict=reward_dict, **kwargs)
        self.q_cache = {}

    def coor_to_state(self, r, c):
        """
        Params:
            r [int]: The state's row for human 1 and 2 together.
            c [int]: The state's column for human 1 and 2 together.

        Returns:
            s [int]: The state number associated with the given coordinates in
                a standard grid world.
        """
        assert 0 <= r < self.X, "invalid (rows, r)={}".format((self.X, r))
        assert 0 <= c < self.Y, "invalid (cols, c)={}".format((self.Y, c))
        return r * self.Y + c

    def state_to_coor(self, s):
        """
        Params:
            s [int]: The state.

        Returns:
            r, c [int]: The row and column associated with state s for each human.
        """
        assert s < self.X * self.Y
        return s // self.Y, s % self.Y

    def coor2D_to_coor4D(self, r, c):
        """
        Params:
            r, c [int]: The PACKED row and column for both humans.

        Returns:
            r1, c1, r2, c2 [int]: The UNPACKED row and column for both humans.
        """
        r1, r2 = r // np.sqrt(self.X), r % np.sqrt(self.X)
        c1, c2 = c // np.sqrt(self.Y), c % np.sqrt(self.Y)
        return int(r1), int(c1), int(r2), int(c2)

    def coor4D_to_coor2D(self, r1, c1, r2, c2):
        """
        Params:
            r1, c1, r2, c2 [int]: The UNPACKED row and column for both humans.
        Returns:
            r, c [int]: The PACKED row and column for both humans.
        """
        r = r1 * np.sqrt(self.X) + r2
        c = c1 * np.sqrt(self.Y) + c2
        return int(r), int(c)

# Classic Gridworld with 2 humans
class GridWorldMDP2H(MDP4D):
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

        A = len(Actions)

        MDP4D.__init__(self, rows=rows, cols=cols, A=len(Actions),
                transition_helper=self._transition_helper, **kwargs)

        for a in range(self.A):
            a1 = a // len(Actions)
            a2 = a % len(Actions)
            if euclidean_rewards:
                if a1 in diagonal_actions and a2 in diagonal_actions:
                    col = self.rewards[:, a]
                    np.multiply(col, 2*np.sqrt(2), out=col)
                elif a1 in diagonal_actions or a2 in diagonal_actions:
                    col = self.rewards[:, a]
                    np.multiply(col, (np.sqrt(2)+1), out=col)
                else:
                    col = self.rewards[:, a]
                    np.multiply(col, 2, out=col)

            if not (self.allow_wait and (a1 == Actions.ABSORB or a2 == Actions.ABSORB)):
                self.rewards[:, a].fill(-np.inf)

        for s in range(self.S):
            r, c = self.state_to_coor(s)
            r1, c1, r2, c2 = self.coor2D_to_coor4D(r, c)
            dist_H = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

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
                Q[s, Actions.ABSORB**2] = 0
                # All other actions will be -np.inf by default.
                continue

            for a in range(self.A):
                if s == goal_spec and a == Actions.ABSORB**2:
                    Q[s, a] = 0
                else:
                    Q[s,a] = self.rewards[s,a] + V[self.transition(s,a)]
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_spec, goal_stuck)] = Q
        return np.copy(Q)