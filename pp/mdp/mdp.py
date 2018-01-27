from __future__ import division

from enum import IntEnum
import numpy as np
from hardmax import forwards_value_iter as _value_iter
from sklearn.preprocessing import normalize

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

    def q_values(mdp, goal_state, goal_stuck=False, **kwargs):
        raise Exception("Abstract method")

    def action_probabilities(self, goal_state, beta=1, q_cached=None,
            goal_stuck=False, **kwargs):
        """
        At each state, calculate the softmax probability of each action
        using hardmax Q values.

        Params:
            goal_state [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
        Returns:
        """
        assert beta > 0, beta

        key = (goal_state, beta, goal_stuck)

        if q_cached is not None:
            Q = np.copy(q_cached)
        else:
            if key in self.act_prob_cache:
                return np.copy(self.act_prob_cache[key])
            Q = self.q_values(goal_state, goal_stuck=goal_stuck, **kwargs)

        np.divide(Q, beta, out=Q)
        # Use amax to mitigate numerical errors
        amax = np.amax(Q, axis=1, keepdims=1)
        np.subtract(Q, amax, out=Q)

        np.exp(Q, out=Q)
        normalize(Q, norm='l1', copy=False)
        self.act_prob_cache[key] = np.copy(Q)
        return Q

    def transition_probabilities(self, beta=1, dest=None, goal_stuck=False,
            act_probs_cached=None):
        """
        Calculate the SxS state probability transition matrix `T` for a
        beta-irrational agent.
        Params:
        dest [int]: If provided, switch the MDP's goal to this state first.
            Otherwise, use the MDP's most recent goal.
        """
        assert beta > 0, beta
        if dest is not None:
            self.set_goal(dest)
        key = (self.goal, beta, goal_stuck)

        if act_probs_cached is None:
            if key in self.trans_prob_cache:
                return self.trans_prob_cache[key]
            P = self.action_probabilities(self.goal, beta=beta, q_cached=None,
                    goal_stuck=goal_stuck)
        else:
            P = act_probs_cached

        tup_ref = self.transition_cached_t

        T = np.zeros([self.S, self.S])
        for s in range(self.S):
            for a in range(self.A):
                s_prime = tup_ref[s*self.A + a]
                T[s_prime, s] += P[s, a]

        self.trans_prob_cache[key] = np.copy(T)
        return T

    def trajectory_probability(self, goal_state, traj, beta=1,
            cached_act_probs=None):
        """
        Calculate the product of the probabilities of each
        state-action pair in this trajectory given an mdp,
        a goal_state, and beta.

        Params:
            goal_state [int]: The goal state. At the goal state, the agent
                always chooses the ABSORB action at no cost.
            traj [list of (int, int)]: A list of state-action pairs. If this
                is an empty list, return traj_prob=1.
            beta [float] (optional): Irrationality constant.
            cached_act_probs [ndarray] (optional): Cached results of
                action_probabilities. Mainly for testing purposes.
        Return:
            traj_prob [float].
        """
        if len(traj) == 0:
            return 1

        if cached_act_probs is None:
            P = self.action_probabilities(goal_state, beta=beta)
        else:
            P = cached_act_probs

        traj_prob = 1
        for s, a in traj:
            traj_prob *= P[s, a]
        return traj_prob

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
    diagonal_actions = {4, 5, 6, 7}

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

    def q_values(self, goal_state, forwards_value_iter=_value_iter,
            goal_stuck=False):
        """
        Calculate the hardmax Q values for each state action pair.

        Params:
            goal_state [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If this is True, then all actions other than
                ABSORB are illegal in the goal_state.

        Returns:
            Q [np.ndarray]: An SxA array containing the q values
                corresponding to each (s, a) pair.
        """
        if (goal_state, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_state, goal_stuck)])

        self.set_goal(goal_state)
        V = forwards_value_iter(self, goal_state)

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in range(self.S):
            if s == goal_state:
                Q[s, self.Actions.ABSORB] = 0
                if goal_stuck:
                    continue
            # TODO:
            # For the purposes of Jaime/Andrea's demo, I am allowing non-ABSORB
            # actions at the goal.
            #
            # My simulations might break if human moves off this square.
            # This is something worth thinking about. XXX
            for a in range(self.A):
                Q[s,a] = self.rewards[s,a] + V[self.transition(s,a)]
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_state, goal_stuck)] = Q
        return np.copy(Q)
