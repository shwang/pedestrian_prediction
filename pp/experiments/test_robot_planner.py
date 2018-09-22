from __future__ import division
from unittest import TestCase

import numpy as np
from numpy import testing as t

from .robot_planner import *
from .context import HRContext
from ..mdp import GridWorldMDP

class TestCollideProbs(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traj = [(0, 1), (1, 2), (3, 4)]

    def test_radius_0(self):
        g_R = GridWorldMDP(5, 5)
        g_H = GridWorldMDP(5, 5)
        ctx = HRContext(g_R=g_R, goal_R=20, g_H=g_H, goal_H=2,
                collide_radius=0, collide_penalty=10, traj_H=[(1, 2)],
                start_H=0)

        state_probs = np.ones([1, g_H.S])/g_H.S
        cp = CollideProbs(ctx=ctx, T=0, traj = self.traj,
                state_probs_cached=np.copy(state_probs))
        collide_probs = np.empty([1, g_H.S])

        for s in range(g_H.S):
            collide_probs[0, s] = cp.get(0, s)
        rows, cols = g_H.rows, g_H.cols
        t.assert_allclose(
                state_probs.reshape(1, rows, cols), collide_probs.reshape(1, rows, cols)) 
    def test_radius_1(self):
        g_R = GridWorldMDP(3, 3)
        g_H = GridWorldMDP(3, 3)
        ctx = HRContext(g_R=g_R, goal_R=1, g_H=g_H, goal_H=2,
                collide_radius=1, collide_penalty=10, traj_H=[(1, 2)],
                start_H=0)

        state_probs = np.array(
                [[1/3, 0, 0],
                [0, 0, 0],
                [1/3, 0, 1/3]])
        expected_probs = np.array(
                [[1/3, 1/3, 0],
                [2/3, 1, 1/3],
                [1/3, 2/3, 1/3]])

        cp = CollideProbs(ctx, T=0, traj=self.traj,
                state_probs_cached=np.copy(state_probs))
        collide_probs = np.empty([1, g_H.S])
        for s in range(g_H.S):
            collide_probs[0, s] = cp.get(0, s)
        rows, cols = g_H.rows, g_H.cols

        t.assert_allclose(
                collide_probs.reshape(1, rows, cols),
                expected_probs.reshape(1, rows, cols))

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

    @classmethod
    def setUpClass(cls):
        g_R = cls.g_R = GridWorldMDP(6, 6)
        g_R.set_goal(35)
        g_H = cls.g_H = GridWorldMDP(6, 6)
        g_H.set_goal(2)
        cls.ctx = HRContext(g_R=g_R, goal_R=35, g_H=g_H, goal_H=2,
                collide_radius=1, collide_penalty=10, traj_H=[(1, 2)],
                start_H=0)
        cls.traj = [(0, 1), (1, 2), (3, 4)]
        cls.safe_traj_H = [(g_H.coor_to_state(3, 3), g_H.Actions.UP)]

    def test_long_no_crash(self):
        # The following calls should not crash.
        robot_planner_vanilla(ctx=self.ctx, state_R=0, traj_H=[])
        robot_planner_vanilla(ctx=self.ctx, state_R=0, traj_H=[(1, 2)])

    def test_long_no_crash_fixed(self):
        robot_planner_fixed(ctx=self.ctx, state_R=0, traj_H=[])
        robot_planner_fixed(ctx=self.ctx, state_R=0, traj_H=[(1, 2)])

    def test_long_no_crash_bayes(self):
        betas = [0.1, 1, 2]
        priors = [0.3, 0.4, 0.3]
        robot_planner_bayes(ctx=self.ctx, betas=betas, priors=priors, state_R=0,
                traj_H=[])
        # TODO: recall that bayes gives all nan in collide probabilities if
        #       traj_H contains illegal moves. Think about fixing this?
        #
        #       Example: change traj_H back to (1, 2)
        robot_planner_bayes(ctx=self.ctx, betas=betas, priors=priors, state_R=0,
                traj_H=self.safe_traj_H)
