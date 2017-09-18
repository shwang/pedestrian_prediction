from enum import IntEnum
import numpy as np

class MDP:
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
        WAIT = 8

    def __init__(self, rows, cols, reward_dict, default_reward=0, wait_reward=0):
        """
        An agent in a GridWorldMDP can move between adjacent/diagonal cells.
        Illegal actions (those that would move off the grid) always result in
        a float('-inf') reward.

        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            reward_dict [dict]: Maps (r, c) to _reward. In the GridWorldMDP, transitioning
                to (r, c) with a non-WAIT action will grant the reward _reward.
            default_reward [float]: (optional) Every reward not set by reward_dict
                or wait_reward will receive this default reward instead.
            wait_reward [float]: (optional) The reward granted for taking the WAIT action
                in any state.

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
        rewards[:, self.Actions.WAIT].fill(wait_reward)

        for s in range(S):
            for a in range(A):
                if a == self.Actions.WAIT:
                    continue
                s_prime, illegal = self._transition(s, a, alert_illegal=True)
                coor = self.state_to_coor(s_prime)
                if not illegal:
                    if coor in reward_dict:
                        rewards[s, a] = reward_dict[coor]
                else:
                    rewards[s, a] = float('-inf')

        super().__init__(S, A, rewards, self._transition)

    def _transition(self, s, a, alert_illegal=False):
        r, c = self.state_to_coor(s)
        assert a >= 0 and a < len(self.Actions), a

        r_prime, c_prime = r, c
        if a == self.Actions.UP:
            r_prime = r - 1
        elif a == self.Actions.DOWN:
            r_prime = r + 1
        elif a == self.Actions.LEFT:
            c_prime = c - 1
        elif a == self.Actions.RIGHT:
            c_prime = c + 1
        elif a == self.Actions.UP_LEFT:
            r_prime, c_prime = r - 1, c - 1
        elif a == self.Actions.UP_RIGHT:
            r_prime, c_prime = r - 1, c + 1
        elif a == self.Actions.DOWN_LEFT:
            r_prime, c_prime = r + 1, c - 1
        elif a == self.Actions.DOWN_RIGHT:
            r_prime, c_prime = r + 1, c + 1
        elif a == self.Actions.WAIT:
            pass
        else:
            raise BaseException("undefined action {}".format(a))

        illegal = False
        if r_prime < 0 or r_prime >= self.rows or c_prime < 0 or c_prime >= self.cols:
            r_prime, c_prime = r, c
            illegal = True

        s_prime = self.coor_to_state(r_prime, c_prime)

        if alert_illegal:
            return s_prime, illegal
        else:
            return s_prime

    def coor_to_state(self, r, c):
        """
        Params:
            r [int]: The state's row.
            c [int]: The state's column.

        Returns:
            s [int]: The state number associated with the given coordinates in a standard
                grid world.
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
