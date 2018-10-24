from __future__ import division

import numpy as np
from sklearn.preprocessing import normalize

# Abstract Class -- implement Q values
class MDP(object):
    def __init__(self, S, A, transition_helper, reward_dict={},
            default_reward=-1):
        """
        A generic MDP with a discrete and finite state and action spaces.

        Params:
            S [int]: The number of states.
            A [int]: The number of actions.
            transition_helper [function]: The state transition function for the
                deterministic MDP. transition_helper(s, a, alert_illegal=True)
                returns the state `s_new`
                that results from taking action a at state s, and whether this
                state-action pair represents an illegal move.
            default_reward [int]: The default reward of any state-action pair
                (s, a). This is reward yielded by any legal state-action pair
                that is unaffected by reward_dict.

        Debug Params (mainly used in unittests):
            reward_dict [dict]: Maps state `s_new` to reward `R`. Passing in a
                nonempty dict for this parameter will make any legal
                state-action pair that transitions to `s_new` yield the reward
                `R`.
        """
        assert isinstance(S, int), S
        assert isinstance(A, int), A
        assert callable(transition_helper), transition_helper

        self.S = S
        self.A = A

        self.default_reward = default_reward
        self.rewards = np.zeros([S, A])
        self.rewards.fill(default_reward)

        self.act_prob_cache = {}
        self.trans_prob_cache = {}

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
                if not illegal:
                    if s_prime in reward_dict:
                        self.rewards[s, a] = reward_dict[s_prime]
                    self.neighbors[s].append((a, s_prime))
                    self.reverse_neighbors[s_prime].append((a, s))
                else:
                    self.rewards[s, a] = -np.inf

        # For speedier computation in which transition is a bottleneck.
        self.transition_cached_t = tuple(self.transition_cached_l)

    def transition(self, s, a):
        return self.transition_cached[s, a]

    def q_values(self, goal_spec, goal_stuck=False, **kwargs):
        """
        Calculate the hardmax Q values for each state action pair.

        Params:
            goal_spec: A hashable parameter that indicates which states are goal
                states. The format is subclass-dependent (see q_values).
                At goal states, the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If True, then force the agent to take the ABSORB
                action if it is at a goal state.
        """
        raise Exception("Abstract method")

    def action_probabilities(self, goal_spec, beta=1, q_cached=None,
            goal_stuck=False, **kwargs):
        """
        At each state, calculate the softmax probability of each action
        using hardmax Q values.

        Params:
            goal_spec: A hashable parameter that indicates which states are goal
                states. The format is subclass-dependent (see q_values).
                At goal states, the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If True, then force the agent to take the ABSORB
                action if it is at a goal state.
        Returns:
            P [np.ndarray]: An S x A matrix where the (s, a) entry is the
                probability that a beta-irrational agent will choose action `a`
                given state `s`.
        """
        assert beta > 0, beta

        key = (goal_spec, beta, goal_stuck)

        if q_cached is not None:
            Q = np.copy(q_cached)
        else:
            if key in self.act_prob_cache:
                return np.copy(self.act_prob_cache[key])
            Q = self.q_values(goal_spec, goal_stuck=goal_stuck, **kwargs)

        np.divide(Q, beta, out=Q)
        # Use amax to mitigate numerical errors
        amax = np.amax(Q, axis=1, keepdims=1)
        np.subtract(Q, amax, out=Q)

        np.exp(Q, out=Q)
        normalize(Q, norm='l1', copy=False)
        self.act_prob_cache[key] = np.copy(Q)
        return Q


    def transition_probabilities(self, goal_spec, beta=1, goal_stuck=False,
            act_probs_cached=None):
        """
        Calculate the SxS state probability transition matrix `T` for a
        beta-irrational agent.

        Params:
            goal_spec: A hashable parameter that indicates which states are goal
                states. The format is subclass-dependent (see q_values).
                At goal states, the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If True, then force the agent to take the ABSORB
                action if it is at a goal state.
        """
        assert beta > 0, beta
        key = (goal_spec, beta, goal_stuck)

        if act_probs_cached is None:
            if key in self.trans_prob_cache:
                return self.trans_prob_cache[key]
            P = self.action_probabilities(goal_spec, beta=beta, q_cached=None,
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

    def trajectory_probability(self, goal_spec, traj, beta=1,
            cached_act_probs=None):
        """
        Calculate the product of the probabilities of each
        state-action pair in this trajectory given an mdp,
        a goal_spec, and beta.

        Params:
            goal_spec: A hashable parameter that indicates which states are goal
                states. The format is subclass-dependent (see q_values).
                At goal states, the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
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
            P = self.action_probabilities(goal_spec, beta=beta)
        else:
            P = cached_act_probs

        traj_prob = 1
        for s, a in traj:
            traj_prob *= P[s, a]
        return traj_prob
