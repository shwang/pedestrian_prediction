from __future__ import absolute_import
from unittest import TestCase
import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from .hardmax import dijkstra

class TestDijkstra(TestCase):
    def test_simple(self):
        g = GridWorldMDP(3, 1, {(2,0): -9}, default_reward=-1)
        t.assert_allclose([0, -1, -10], dijkstra(g, 0))
        t.assert_allclose([-1, 0, -9], dijkstra(g, 1))
        t.assert_allclose([-2, -1, 0], dijkstra(g, 2))
    def test_2d(self):
        g = GridWorldMDP(3, 3, {(2,0): -3, (1,1): -4}, default_reward=-1)
        expected = [-3, -2, -2,
                    -2, -4, -1,
                    -4, -1, 0]
        t.assert_allclose(expected, dijkstra(g, g.coor_to_state(2,2)))
