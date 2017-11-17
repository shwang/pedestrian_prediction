from __future__ import absolute_import
from enum import IntEnum
import numpy as np

class MDP(object):
    def __init__(self, S, A, rewards, transition):
        """
        Params:
            S [int]: The number of states.
            A [int]: The number of actions.
            rewards [np.ndarray]: a SxA array where rewards[s, a] is the reward
                received from taking action a at state s.
            transition [function]: The state transition function for the deterministic
                MDP. transition(s, a) returns the state that results from taking action
                a at state s.
        """
        assert isinstance(S, int), S
        assert isinstance(A, int), A
        assert rewards.shape == (S, A), rewards
        assert callable(transition), transition

        self.S = S
        self.A = A
        self.rewards = rewards
        self.transition = transition

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

    # TODO: investigate how goal state works nowadays
    def __init__(self, rows, cols, reward_dict={}, goal_state=None, default_reward=0,
            euclidean_rewards=False):
        """
        An agent in a GridWorldMDP can move between adjacent/diagonal cells.

        If the agent chooses an illegal action it receives a float('-inf') reward
        and will stay in place.

        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            reward_dict [dict]: Maps (r, c) to _reward. In the GridWorldMDP, transitioning
                to (r, c) will grant the reward _reward.
            goal_state [int]: (optional) The goal state at which ABSORB is legal and costs 0.
                If not provided, then goal_state is 0 everywhere.
            default_reward [float]: (optional) Every reward not set by reward_dict
                will receive this default reward instead.
            euclidean_rewards [bool]: (optional) If True, then scale rewards for moving
                diagonally by sqrt(2).
        """
        assert rows > 0
        assert cols > 0
        assert isinstance(rows, int)
        assert isinstance(cols, int)
        self.rows = rows
        self.cols = cols

        S = rows * cols
        A = len(self.Actions)

        rewards = np.zeros([S, A])
        rewards.fill(default_reward)

        # neighbor[s] is a set of tuples (a, s_prime)
        self.neighbors = [[] for _ in xrange(S)]
        # reverse_neighbors is a set of tuples (a, s)
        self.reverse_neighbors = [[] for _ in xrange(S)]

        self.transition_cached = np.empty([S, A], dtype=int)

        for s in xrange(S):
            for a in xrange(A):
                s_prime, illegal = self._transition_helper(s, a, alert_illegal=True)
                self.transition_cached[s, a] = s_prime
                coor = self.state_to_coor(s_prime)
                if not illegal:
                    if coor in reward_dict:
                        rewards[s, a] = reward_dict[coor]
                    self.neighbors[s].append((a, s_prime))
                    self.reverse_neighbors[s_prime].append((a, s))
                else:
                    rewards[s, a] = -np.inf

        if euclidean_rewards:
            for a in self.diagonal_actions:
                col = rewards[:, a]
                np.multiply(col, np.sqrt(2), out=col)

        super(GridWorldMDP, self).__init__(S, A, rewards, self._transition)

        self.state_rewards = np.full([S], default_reward)
        for (r, c), reward in reward_dict.items():
            self.state_rewards[self.coor_to_state(r,c)] = reward
        self.set_goal(goal_state)

    def copy(self):
        cp = GridWorldMDP(self.rows, self.cols, {})
        cp.rewards = np.copy(self.rewards)
        return cp

    def _transition(self, s, a):
        # import pdb; pdb.set_trace()
        return self.transition_cached[s, a]

    # TODO: optimize so that we don't need to convert between state and coor.
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
            raise BaseException(u"undefined action {}".format(a))

        illegal = False
        if r_prime < 0 or r_prime >= self.rows or c_prime < 0 or c_prime >= self.cols:
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
        the goal state to use the ABSORB action at no cost. At all other states,
        ABSORB will be illegal (i.e., incur inf cost).

        Params:
            goal_state: The new goal. Overrides previous goals.
        """
        self.rewards[:, self.Actions.ABSORB].fill(float(u'-inf'))
        if goal_state != None:
            self.rewards[goal_state, self.Actions.ABSORB] = 0

    def set_all_goals(self):
        """
        (Experimental)
        Allow ABSORB at every state.
        """
        self.rewards[:, self.Actions.ABSORB].fill(0)

    def coor_to_state(self, r, c):
        """
        Params:
            r [int]: The state's row.
            c [int]: The state's column.

        Returns:
            s [int]: The state number associated with the given coordinates in a standard
                grid world.
        """
        assert 0 <= r < self.rows, u"invalid (rows, r)={}".format((self.rows, r))
        assert 0 <= c < self.cols, u"invalid (cols, c)={}".format((self.cols, c))
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

GridWorldMDP.diagonal_actions = {
        GridWorldMDP.Actions.UP_LEFT,
        GridWorldMDP.Actions.UP_RIGHT,
        GridWorldMDP.Actions.DOWN_LEFT,
        GridWorldMDP.Actions.DOWN_RIGHT,
}
