from __future__ import division

from unittest import TestCase

import numpy as np
from numpy import testing as t

from .robot_planner import *
from ..mdp import GridWorldMDP

class TestCollideProbs(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traj = [(0, 1), (1, 2), (3, 4)]

    def test_radius_0(self):
        g = GridWorldMDP(5, 5)
        state_probs = np.ones([1, g.S])/g.S
        cp = CollideProbs(g, T=0, collide_radius=0, traj = self.traj,
                state_probs_cached=np.copy(state_probs))
        collide_probs = np.empty([1, g.S])

        for s in range(g.S):
            collide_probs[0, s] = cp.get(0, s)
        rows, cols = g.rows, g.cols
        t.assert_allclose(
                state_probs.reshape(1, rows, cols),
                collide_probs.reshape(1, rows, cols))

    def test_radius_1(self):
        g = GridWorldMDP(3, 3)
        state_probs = np.array(
                [[1/3, 0, 0],
                [0, 0, 0],
                [1/3, 0, 1/3]])
        expected_probs = np.array(
                [[1/3, 1/3, 0],
                [2/3, 1, 1/3],
                [1/3, 2/3, 1/3]])

        cp = CollideProbs(g, T=0, collide_radius=1, traj=self.traj,
                state_probs_cached=np.copy(state_probs))
        collide_probs = np.empty([1, g.S])
        for s in range(g.S):
            collide_probs[0, s] = cp.get(0, s)
        rows, cols = g.rows, g.cols

        t.assert_allclose(
                collide_probs.reshape(1, rows, cols),
                expected_probs.reshape(1, rows, cols))

class TestAStarNode(TestCase):
    def test_init(self):
        g = GridWorldMDP(3, 3)
        s = 2
        node = AStarNode(g, s)
        self.assertEqual(node.s, s)
        self.assertEqual(node.backward_cost, 0)
        self.assertEqual(node.traj, tuple())
        self.assertEqual(node.t, 0)

    def test_child(self):
        g = GridWorldMDP(3, 3)
        s = 0
        node = AStarNode(g, s)

        a = g.Actions.UP
        s_prime = g.transition(s, a)
        new_cost = 5
        child = node.make_child(a, new_cost)
        self.assertEqual(child.t, 1)
        self.assertEqual(child.backward_cost, new_cost)
        self.assertEqual(child.s, s_prime)
        self.assertEqual(child.traj, ((s, a),))

class TestRobotPlanner(TestCase):

    # XXX: This test is broken. For now behavior is verifiable with plots,
            # but it certainly would be useful to have this unit test.
    # def test_trivial_at_goal(self):
    #     s = 1
    #     g_R = GridWorldMDP(3, 3)
    #     g_R.set_goal(s)
    #     g_H = GridWorldMDP(3, 3)
    #     g_H.set_goal(2)
    #     plan = robot_planner(g_R, s, g_H, traj=[], start_H=2)
    #     t.assert_equal(plan, [])

    def test_long_no_crash(self):
        g_R = GridWorldMDP(6, 6)
        g_R.set_goal(35)
        g_H = GridWorldMDP(6, 6)
        g_H.set_goal(2)
        # The following calls should not crash.
        robot_planner(g_R, 0, g_H, collide_penalty=10, collide_radius=1,
                traj=[(1, 2)])
        robot_planner(g_R, 0, g_H, collide_penalty=10, collide_radius=1,
                start_H=0)
