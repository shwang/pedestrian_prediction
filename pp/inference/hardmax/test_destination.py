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
