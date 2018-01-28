from unittest import TestCase
import numpy as np
from numpy import testing as t

from .. import GridWorldMDP
from .hardmax import *

Actions = GridWorldMDP.Actions
ni = -np.inf

class TestBackwardsValueIter(TestCase):
    def test_simple(self):
        g = GridWorldMDP(3, 1, reward_dict={(2,0): -9}, default_reward=-1)
        t.assert_allclose([0, -1, -10], backwards_value_iter(g, 0))
        t.assert_allclose([-1, 0, -9], backwards_value_iter(g, 1))
        t.assert_allclose([-2, -1, 0], backwards_value_iter(g, 2))
    def test_2d(self):
        g = GridWorldMDP(3, 3, reward_dict={(2,0): -3, (1,1): -4},
                default_reward=-1, euclidean_rewards=False)
        expected = [-3, -2, -2,
                    -2, -4, -1,
                    -4, -1, 0]
        t.assert_allclose(expected, backwards_value_iter(g, g.coor_to_state(2,2)))

class TestForwardsValueIter(TestCase):
    def test_simple(self):
        g = GridWorldMDP(3, 1, reward_dict={(2,0): -9}, default_reward=-1,
                euclidean_rewards=False)
        t.assert_allclose([0, -1, -2], forwards_value_iter(g, 0))
        t.assert_allclose([-1, 0, -1], forwards_value_iter(g, 1))
        t.assert_allclose([-10, -9, 0], forwards_value_iter(g, 2))
    def test_2d(self):
        g = GridWorldMDP(3, 3, reward_dict={(2,0): -3, (1,1): -4},
                default_reward=-1, euclidean_rewards=False)
        expected = [-3, -2, -2,
                    -2, -1, -1,
                    -2, -1, 0]

        t.assert_allclose(expected, forwards_value_iter(g, g.coor_to_state(2,2)))

class TestQValues(TestCase):
    def test_simple(self):
        g = GridWorldMDP(3, 1, reward_dict={(2,0): -9}, default_reward=-1)
        Q = np.empty([g.S, g.A], dtype=float)

        # V = [0, -1, -2]
        Q.fill(-np.inf)
        Q[0][Actions.ABSORB] = 0
        Q[1][Actions.LEFT] = -1
        Q[1][Actions.RIGHT] = -11
        Q[2][Actions.LEFT] = -2
        t.assert_allclose(Q, g.q_values(0, goal_stuck=True))

        # V = [-1, 0, -1]
        Q.fill(-np.inf)
        Q[0][Actions.RIGHT] = -1
        Q[1][Actions.ABSORB] = 0
        Q[2][Actions.LEFT] = -1
        t.assert_allclose(Q, g.q_values(1, goal_stuck=True))

        # V = [-10, -9, 0]
        Q.fill(-np.inf)
        Q[0][Actions.RIGHT] = -10
        Q[1][Actions.LEFT] = -11
        Q[1][Actions.RIGHT] = -9
        Q[2][Actions.LEFT] = -10
        Q[2][Actions.ABSORB] = 0
        t.assert_allclose(Q, g.q_values(2, goal_stuck=False))

class TestActionProbabilities(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GridWorldMDP(5,5)

    def test_one_choice(self):
        q_cached = np.array([[1, ni, ni], [ni, 1, ni], [ni, ni, 1]])
        P = self.g.action_probabilities(3, q_cached=q_cached)
        t.assert_allclose(P, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_uniform(self):
        q_cached = np.zeros([3,3])
        P = self.g.action_probabilities(3, q_cached=q_cached)
        t.assert_allclose(P, np.ones([3,3])/3)

    def test_goal_state_forces_absorb(self):
        g = self.g
        s = g.coor_to_state(3, 3)
        P = self.g.action_probabilities(goal=s, goal_stuck=True)

        a = g.Actions.UP

        assert P[s, a] == 0

    def test_goal_state_allows_nonabsorb(self):
        g = self.g
        s = g.coor_to_state(3, 3)
        P = self.g.action_probabilities(goal=s)

        a = g.Actions.UP

        assert P[s, a] > 0

class TestTrajProb(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GridWorldMDP(5,5)

    def test_empty(self):
        t.assert_equal(self.g.trajectory_probability(2, [], beta=1), 1)
        t.assert_equal(self.g.trajectory_probability(2, [], beta=2), 1)
        t.assert_equal(self.g.trajectory_probability(2, [], beta=3), 1)

    def test_one_traj(self):
        fake_prob = np.zeros([self.g.S, self.g.A])
        traj = [(0,0), (1,1), (2, 2)]
        fake_prob[0, 0] = 0.5
        fake_prob[1, 1] = 0.5
        fake_prob[2, 2] = 0.5
        t.assert_equal(self.g.trajectory_probability(2, traj=traj,
                    cached_act_probs=fake_prob), 0.125)


class TestTransitionProbabilities(TestCase):

    def test_empty(self):
        g = GridWorldMDP(5,5)
        g.set_goal(4)  # meaningless, a vestigal necessity for now. XXX

        act_probs = np.zeros([g.S, g.A])
        res = g.transition_probabilities(act_probs_cached=act_probs)
        expect = np.zeros([g.S, g.S])

        t.assert_equal(res, expect)

    def test_simple(self):
        g = GridWorldMDP(5,5)
        g.set_goal(4)

        s = g.coor_to_state(0, 0)
        s_right = g.coor_to_state(1, 0)
        s_up = g.coor_to_state(0, 1)

        act_probs = np.zeros([g.S, g.A])
        act_probs[s, g.Actions.RIGHT] = 0.5
        act_probs[s, g.Actions.UP] = 0.5

        res = g.transition_probabilities(act_probs_cached=act_probs)
        expect = np.zeros([g.S, g.S])

        expect[s_right, s] = 0.5
        expect[s_up, s] = 0.5

        t.assert_allclose(res, expect)

    def test_clockwise(self):
        g = GridWorldMDP(2, 2)
        A = g.coor_to_state(0, 1)
        B = g.coor_to_state(1, 1)
        C = g.coor_to_state(1, 0)
        D = g.coor_to_state(0, 0)

        P = np.zeros([g.S, g.A])
        P[A, g.Actions.RIGHT] = 1
        P[B, g.Actions.DOWN] = 1
        P[C, g.Actions.LEFT] = 1
        P[D, g.Actions.UP] = 1

        expect = np.zeros([g.S, g.S])
        M = expect.T
        M[A, B] = 1
        M[B, C] = 1
        M[C, D] = 1
        M[D, A] = 1

        res = g.transition_probabilities(act_probs_cached=P)
        t.assert_allclose(res, expect)
