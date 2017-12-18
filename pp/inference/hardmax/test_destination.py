from __future__ import division

from unittest import TestCase

import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP
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

