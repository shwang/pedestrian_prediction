from __future__ import division

import numpy as np
from sklearn.preprocessing import normalize

# Abstract Class -- implement Q values
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

    # TODO: cache all these values, because `self.*` is costly.
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

    # TODO: cache all these values, because `self.*` is costly.
    def state_to_coor(self, s):
        """
        Params:
            s [int]: The state.

        Returns:
            r, c [int]: The row and column associated with state s.
        """
        assert s < self.rows * self.cols
        return s // self.cols, s % self.cols

    def set_goal(self, goal):
        self.goal = goal

    def q_values(self, goal_state, goal_stuck=False, **kwargs):
        raise Exception("Abstract method")

    def action_probabilities(self, goal, beta=1, q_cached=None,
            goal_stuck=False, **kwargs):
        """
        At each state, calculate the softmax probability of each action
        using hardmax Q values.

        Params:
            goal_state [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
        Returns:
            P [np.ndarray]: An S x A matrix where the (s, a) entry is the
                probability that a beta-irrational agent will choose action `a`
                given state `s`.
        """
        assert beta > 0, beta

        key = (goal, beta, goal_stuck)

        if q_cached is not None:
            Q = np.copy(q_cached)
        else:
            if key in self.act_prob_cache:
                return np.copy(self.act_prob_cache[key])
            Q = self.q_values(goal, goal_stuck=goal_stuck, **kwargs)

        np.divide(Q, beta, out=Q)
        # Use amax to mitigate numerical errors
        amax = np.amax(Q, axis=1, keepdims=1)
        np.subtract(Q, amax, out=Q)

        np.exp(Q, out=Q)
        normalize(Q, norm='l1', copy=False)
        self.act_prob_cache[key] = np.copy(Q)
        return Q

    def transition_probabilities(self, beta=1, goal=None, goal_stuck=False,
            act_probs_cached=None):
        """
        Calculate the SxS state probability transition matrix `T` for a
        beta-irrational agent.
        Params:
        goal [int]: If provided, switch the MDP's goal to this state first.
            Otherwise, use the MDP's most recent goal.
        """
        assert beta > 0, beta
        if goal is not None:
            self.set_goal(goal)
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

    def trajectory_probability(self, goal, traj, beta=1,
            cached_act_probs=None):
        """
        Calculate the product of the probabilities of each
        state-action pair in this trajectory given an mdp,
        a goal_state, and beta.

        Params:
            goal [int]: The goal state. At the goal state, the agent
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
            P = self.action_probabilities(goal, beta=beta)
        else:
            P = cached_act_probs

        traj_prob = 1
        for s, a in traj:
            traj_prob *= P[s, a]
        return traj_prob
