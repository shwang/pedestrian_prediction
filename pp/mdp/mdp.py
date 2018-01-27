from __future__ import division

from enum import IntEnum
import numpy as np

class MDP(object):
    def __init__(self, rows, cols, A, transition_helper, reward_dict={},
            default_reward=-1):
        """
        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            S [int]: The number of states.
            A [int]: The number of actions.
            transition_helper [function]: The state transition function for the
                deterministic MDP. transition(s, a) returns the state that
                results from taking action a at state s.
                Actually, also has more requirements used for caching, no time
                to write that documentation right now.
            reward_dict [dict]: Maps (r, c) to _reward. In the GridWorldMDP,
                transitioning to (r, c) will grant the reward _reward.
        """
        assert isinstance(A, int), A
        assert callable(transition_helper), transition_helper
        assert rows > 0
        assert cols > 0
        assert isinstance(rows, int)
        assert isinstance(cols, int)

        self.S = S = rows * cols
        self.rows = self.width = rows
        self.cols = self.height = cols
        self.A = A

        self.default_reward = default_reward
        self.rewards = np.zeros([S, A])
        self.rewards.fill(default_reward)

        self.act_prob_cache = {}
        self.trans_prob_cache = {}
        # Right now this is used by `val_mod`s. But since I want q_values
        # to become tied to the MDP itself, it should be a private thing.
        self.q_cache = {}

        # neighbor[s] is a set of tuples (a, s_prime)
        self.neighbors = [[] for _ in xrange(S)]
        # reverse_neighbors is a set of tuples (a, s)
        self.reverse_neighbors = [[] for _ in xrange(S)]

        self.transition_cached = np.empty([S, A], dtype=int)
        self.transition_cached_l = [0] * (S*A)

        for s in xrange(S):
            for a in xrange(A):
                s_prime, illegal = transition_helper(s, a, alert_illegal=True)
                self.transition_cached[s, a] = s_prime
                self.transition_cached_l[a + s*A] = s_prime
                coor = self.state_to_coor(s_prime)
                if not illegal:
                    if coor in reward_dict:
                        self.rewards[s, a] = reward_dict[coor]
                    self.neighbors[s].append((a, s_prime))
                    self.reverse_neighbors[s_prime].append((a, s))
                else:
                    self.rewards[s, a] = -np.inf

        # For speedier computation in which transition is a bottleneck.
        self.transition_cached_t = tuple(self.transition_cached_l)

    def transition(self, s, a):
        return self.transition_cached[s, a]

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
class GridWorldMDP(MDP):
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
        MDP.__init__(self, rows=rows, cols=cols, A=len(self.Actions),
                transition_helper=self._transition_helper, **kwargs)

        S, A = self.S, self.A

        if euclidean_rewards:
            for a in self.diagonal_actions:
                col = self.rewards[:, a]
                np.multiply(col, np.sqrt(2), out=col)

        self.set_goal(goal_state)

    # XXX: optimize so that we don't need to convert between state and coor.
    def _transition_helper(self, s, a, alert_illegal=False):
        r, c = self.state_to_coor(s)
        assert a >= 0 and a < len(self.Actions), a

        r_prime, c_prime = r, c
        if a == self.Actions.LEFT:
            r_prime = r - 1
        elif a == self.Actions.RIGHT:
            r_prime = r + 1
        elif a == self.Actions.DOWN:
            c_prime = c - 1
        elif a == self.Actions.UP:
            c_prime = c + 1
        elif a == self.Actions.UP_LEFT:
            r_prime, c_prime = r - 1, c + 1
        elif a == self.Actions.UP_RIGHT:
            r_prime, c_prime = r + 1, c + 1
        elif a == self.Actions.DOWN_LEFT:
            r_prime, c_prime = r - 1, c - 1
        elif a == self.Actions.DOWN_RIGHT:
            r_prime, c_prime = r + 1, c - 1
        elif a == self.Actions.ABSORB:
            pass
        else:
            raise BaseException("undefined action {}".format(a))

        illegal = False
        if r_prime < 0 or r_prime >= self.rows or \
                c_prime < 0 or c_prime >= self.cols:
            r_prime, c_prime = r, c
            illegal = True

        s_prime = self.coor_to_state(r_prime, c_prime)

        if alert_illegal:
            return s_prime, illegal
        else:
            return s_prime

    def set_goal(self, goal_state):
        """
        Reconfigure the goal state in this GridWorldMDP by allowing an agent at
        the goal state to use the ABSORB action at no cost.

        If self.allow_wait is True, then at nongoal states, ABSORB has
        half the `default_reward` cost.
        If self.allow_wait is False, then at nongoal states,
        ABSORB will be illegal (i.e., incur inf cost).

        Params:
            goal_state: The new goal. Overrides previous goals.
        """
        self.goal = goal_state
        if self.allow_wait:
            self.rewards[:, self.Actions.ABSORB].fill(self.default_reward)
        else:
            self.rewards[:, self.Actions.ABSORB].fill(-np.inf)
        if goal_state != None:
            self.rewards[goal_state, self.Actions.ABSORB] = 0

GridWorldMDP.diagonal_actions = {
        GridWorldMDP.Actions.UP_LEFT,
        GridWorldMDP.Actions.UP_RIGHT,
        GridWorldMDP.Actions.DOWN_LEFT,
        GridWorldMDP.Actions.DOWN_RIGHT,
}
