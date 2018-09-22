import math
import numpy as np

from unittest import TestCase
from numpy import testing as t

from .. import GridWorldMDP
from .euclid import forwards_value_iter, backwards_value_iter

class TestEuclidValues(TestCase):

    @staticmethod
    def check_both(g, s, expected):
        t.assert_allclose(forwards_value_iter(g, s), expected)
        t.assert_allclose(backwards_value_iter(g, s), expected)

    def test_min(self):
        g = GridWorldMDP(1, 1)
        s = 0
        expected = [0]
        self.check_both(g, s, expected)

    def test_tiny(self):
        g = GridWorldMDP(2, 2)

        s = 0
        expected = np.zeros(g.S)
        def conf(r, c, v):
            expected[g.coor_to_state(r, c)] = v

        conf(0, 0, 0)
        conf(0, 1, 1)
        conf(1, 0, 1)
        conf(1, 1, math.sqrt(2))

        self.check_both(g, s, -expected)

    def test_mid(self):
        g = GridWorldMDP(3, 3)

        s = g.coor_to_state(2, 2)
        expected = np.zeros(g.S)
        def conf(r, c, v):
            expected[g.coor_to_state(r, c)] = v

        conf(2, 2, 0)
        conf(1, 2, 1)
        conf(2, 1, 1)
        conf(1, 1, math.sqrt(2))
        conf(2, 0, 2)
        conf(0, 2, 2)
        conf(1, 0, math.sqrt(5))
        conf(0, 1, math.sqrt(5))
        conf(0, 0, math.sqrt(8))

        self.check_both(g, s, -expected)

