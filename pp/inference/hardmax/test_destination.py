from __future__ import division

from unittest import TestCase

import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP, GridWorldExpanded
from .destination import *

class TestInfer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GridWorldMDP(5, 5)
        cls.traj = [(0, 1), (1, 2), (3, 4)]

    def test_simple(self):
        bs = [5, 5, 1]
        def bin_search(*kargs, **kwargs):
            return bs.pop(0)

        tp = [3, 3, 4]
        def traj_prob(*kargs, **kwargs):
            return tp.pop(0)

        res_d_probs, res_betas = infer(self.g, self.traj, [1, 2, 3],
                mk_bin_search=bin_search, mk_traj_prob=traj_prob)

        t.assert_allclose(res_d_probs, [0.3, 0.3, 0.4])
        t.assert_equal(res_betas, [5, 5, 1])

class TestHMMInfer(TestCase):

    def test_one_dest(self):
        g = GridWorldMDP(5, 5)
        traj = []
        dests = [g.coor_to_state(4, 4)]
        P_d, _ = hmm_infer(g, traj, dests)
        t.assert_allclose(P_d, [1])

    def test_two_dests_empty_traj(self):
        g = GridWorldMDP(5, 5)
        traj = []
        dests = [g.coor_to_state(4, 4), g.coor_to_state(1, 3)]

        P_d, _ = hmm_infer(g, traj, dests)
        t.assert_allclose(P_d, [0.5, 0.5])

    def test_two_dests(self):
        g = GridWorldMDP(5, 5)
        traj = [(0, g.Actions.UP), (1, g.Actions.UP)]
        dests = [g.coor_to_state(4, 4), g.coor_to_state(1, 3)]

        def act_probs(*kargs, **kwargs):
            return [[1/g.A] * g.A] * g.S

        P_d, _ = hmm_infer(g, traj, dests, mk_act_probs=act_probs, epsilon=0.05)
        t.assert_allclose(P_d, [0.5, 0.5])

    def test_two_dests_verbose(self):
        g = GridWorldMDP(5, 5)
        traj = [(0, g.Actions.UP), (1, g.Actions.UP)]
        dests = [g.coor_to_state(4, 4), g.coor_to_state(1, 3)]

        bs = [5, 5]
        def bin_search(*kargs, **kwargs):
            return bs.pop(0)

        def act_probs(*kargs, **kwargs):
            return [[1/g.A] * g.A] * g.S

        P_d_all, betas = hmm_infer(g, traj, dests, epsilon=0.05,
                mk_bin_search=bin_search,
                mk_act_probs=act_probs, verbose_return=True)
        t.assert_allclose(P_d_all, [[0.5, 0.5]] * 2)
        t.assert_allclose(betas, [5, 5])

class TestInferJoint(TestCase):
    def test_noop(self):
        g = GridWorldMDP(5, 5)
        t.assert_allclose(infer_joint(g, dests=[4], betas=[0.7]), [[1]])
        t.assert_allclose(infer_joint(g, dests=[1, 4], betas=[0.7, 2]),
                np.ones([2,2])/4)
        t.assert_allclose(infer_joint(g, dests=[1, 2], betas=[2, 5], traj=[],
            priors=[[0, 0], [0, 1]]),
            [[0, 0], [0, 1]])
        t.assert_allclose(infer_joint(g, dests=[1, 2], betas=[2, 5],
            priors=[[0, 2], [1, 1]]),
            [[0, 1/2], [1/4, 1/4]])

    def test_beta_rise(self):
        """H takes a path that is irrational given the only goal. We expect
        P(beta=0.5) to fall, and P(beta=2) to rise."""
        g = GridWorldMDP(10, 10)
        A = g.Actions
        coor = g.coor_to_state
        traj = [(coor(3,1), A.DOWN), (coor(2,1), A.DOWN)]
        dests = [coor(9,9)]
        betas = [0.5, 2]

        P_joint_DB, P_joint_DB_all = infer_joint(g, dests=dests, betas=betas,
                traj=traj, verbose_return=True)
        t.assert_equal(len(P_joint_DB_all), 3)
        t.assert_array_equal(P_joint_DB, P_joint_DB_all[2])

        AXIS_D = 1
        P_B_all = np.sum(P_joint_DB_all, axis=AXIS_D)
        # P(beta=2) rises
        self.assertLess(P_B_all[0, 1], P_B_all[1, 1])
        self.assertLess(P_B_all[1, 1], P_B_all[2, 1])
        # P(beta=0.5) falls
        self.assertGreater(P_B_all[0, 0], P_B_all[1, 0])
        self.assertGreater(P_B_all[1, 0], P_B_all[2, 0])

    def test_dest_rise(self):
        """H takes a path that favors goal 0. We expect P(goal_0) to rise, and
        P(goal_1) to fall."""
        g = GridWorldMDP(10, 10)
        A = g.Actions
        coor = g.coor_to_state
        traj = [(coor(9,9), A.LEFT), (coor(8,9), A.LEFT)]
        dests = [coor(1,4), coor(9,1)]
        betas = [0.5, 1, 2, 3]

        P_joint_DB, P_joint_DB_all = infer_joint(g, dests=dests, betas=betas,
                traj=traj, verbose_return=True)
        t.assert_equal(len(P_joint_DB_all), 3)
        t.assert_array_equal(P_joint_DB, P_joint_DB_all[2])

        AXIS_B = 2
        P_D_all = np.sum(P_joint_DB_all, axis=AXIS_B)
        # P(goal_0) rises
        self.assertLess(P_D_all[0, 0], P_D_all[1, 0])
        self.assertLess(P_D_all[1, 0], P_D_all[2, 0])
        # P(goal_1) falls
        self.assertGreater(P_D_all[0, 1], P_D_all[1, 1])
        self.assertGreater(P_D_all[1, 1], P_D_all[2, 1])

class TestInferJointGridless(TestCase):
    def test_noop(self):
        g = GridWorldExpanded(5, 5)
        t.assert_allclose(infer_joint(g, dests=[4], betas=[0.7],
            use_gridless=True),
            [[1]])
        t.assert_allclose(infer_joint(g, dests=[1, 4], betas=[0.7, 2],
            use_gridless=True),
            np.ones([2,2])/4)
        t.assert_allclose(infer_joint(g, dests=[1, 2], betas=[2, 5], traj=[],
            use_gridless=True, priors=[[0, 0], [0, 1]]),
            [[0, 0], [0, 1]])
        t.assert_allclose(infer_joint(g, dests=[1, 2], betas=[2, 5],
            use_gridless=True, priors=[[0, 2], [1, 1]]),
            [[0, 1/2], [1/4, 1/4]])

    def test_dest_rise(self):
        """H takes a path that favors goal 0. We expect P(goal_0) to rise, and
        P(goal_1) to fall."""
        g = GridWorldExpanded(10, 10)
        A = g.Actions
        coor = g.coor_to_state
        traj = [(0, 0), (0.1, 0.4), (0.2, 0.8)]
        dests = [coor(1,4), coor(9,1)]
        betas = [0.5, 1, 2, 3]

        P_joint_DB, P_joint_DB_all = infer_joint(g, dests=dests, betas=betas,
                traj=traj, verbose_return=True, use_gridless=True)
        t.assert_equal(len(P_joint_DB_all), 3)
        t.assert_array_equal(P_joint_DB, P_joint_DB_all[2])

        AXIS_B = 2
        P_D_all = np.sum(P_joint_DB_all, axis=AXIS_B)
        # P(goal_0) rises
        self.assertLess(P_D_all[0, 0], P_D_all[1, 0])
        self.assertLess(P_D_all[1, 0], P_D_all[2, 0])
        # P(goal_1) falls
        self.assertGreater(P_D_all[0, 1], P_D_all[1, 1])
        self.assertGreater(P_D_all[1, 1], P_D_all[2, 1])

    def test_beta_rise(self):
        """H takes a path that is irrational given the only goal. We expect
        P(beta=0.5) to fall, and P(beta=2) to rise."""
        g = GridWorldExpanded(10, 10)
        A = g.Actions
        coor = g.coor_to_state
        traj = [(3,1), (2,1), (1,1)]
        dests = [coor(9,9)]
        betas = [0.5, 2]

        P_joint_DB, P_joint_DB_all = infer_joint(g, dests=dests, betas=betas,
                traj=traj, use_gridless=True, verbose_return=True)
        t.assert_equal(len(P_joint_DB_all), 3)
        t.assert_array_equal(P_joint_DB, P_joint_DB_all[2])

        AXIS_D = 1
        P_B_all = np.sum(P_joint_DB_all, axis=AXIS_D)
        # P(beta=2) rises
        self.assertLess(P_B_all[0, 1], P_B_all[1, 1])
        self.assertLess(P_B_all[1, 1], P_B_all[2, 1])
        # P(beta=0.5) falls
        self.assertGreater(P_B_all[0, 0], P_B_all[1, 0])
        self.assertGreater(P_B_all[1, 0], P_B_all[2, 0])
