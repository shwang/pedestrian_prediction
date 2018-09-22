from __future__ import division
import numpy as np
import math
from unittest import TestCase
from numpy import testing as t
from .gridless import *

class TestDist(TestCase):
    def test_dist_basic(self):
        a = np.array([1,0])
        b = np.array([1,5])
        self.assertAlmostEqual(dist(a, b), 5)
        self.assertAlmostEqual(dist(b, a), 5)

class TestCircle(TestCase):
    def test_circle_basic(self):
        center = np.array([0, 0])
        up = [0, 1]
        down = [0, -1]
        left = [-1, 0]
        right = [1, 0]
        absorb = [0, 0]

        points = circle(center)
        t.assert_allclose(points[0//15], right, atol=1e-7)
        t.assert_allclose(points[90//15], up, atol=1e-7)
        t.assert_allclose(points[180//15], left, atol=1e-7)
        t.assert_allclose(points[270//15], down, atol=1e-7)
        t.assert_allclose(points[-1], absorb, atol=1e-7)

    def test_circle_tripled(self):
        center = np.array([1, 1])
        up = [1, 4]
        down = [1, -2]
        left = [-2, 1]
        right = [4, 1]
        absorb = [1, 1]

        points = circle(center, R=3)
        t.assert_allclose(points[0//15], right, atol=1e-7)
        t.assert_allclose(points[90//15], up, atol=1e-7)
        t.assert_allclose(points[180//15], left, atol=1e-7)
        t.assert_allclose(points[270//15], down, atol=1e-7)
        t.assert_allclose(points[-1], absorb, atol=1e-7)

class TestCircleDists(TestCase):
    def test_circle_dists(self):
        center = np.array([0, 0])
        dest = np.array([5, 0])
        dists = circle_dists(center=center, dest=dest, R=2)
        self.assertAlmostEqual(dists[0//15], 3)
        self.assertAlmostEqual(dists[180//15], 7)
        self.assertAlmostEqual(dists[-1], 5)

    def test_out_of_bounds(self):
        center = np.array([1, 1])
        dest = np.array([5, 1])

        dists = circle_dists(center=center, dest=dest, R=2, W=10, H=10)
        self.assertAlmostEqual(dists[0//15], 2)
        self.assertEqual(dists[180//15], np.inf)
        self.assertAlmostEqual(dists[-1], 4)

class TestActionProbabilities(TestCase):
    def test_basic(self):
        center = np.array([10, 10])
        dest = np.array([15, 10])

        P = action_probabilities(start=center, dest=dest, W=100, H=100, R=1)
        pright, pabsorb, pleft = P[0//15], P[-1], P[180//15]
        pup, pdown = P[90//15], P[270//15]
        self.assertGreater(pright, pabsorb)
        self.assertGreater(pabsorb, pleft)
        self.assertAlmostEqual(pup, pdown)

    def test_out_of_bounds(self):
        center = np.array([10, 10])
        dest = np.array([50, 10])

        P = action_probabilities(start=center, dest=dest, W=100, H=100, R=15)
        pright, pabsorb, pleft = P[0//15], P[-1], P[180//15]
        pup, pdown = P[90//15], P[270//15]
        self.assertGreater(pright, pabsorb)
        self.assertGreater(pabsorb, pleft)
        self.assertAlmostEqual(pup, pdown)
        self.assertEqual(pleft, 0)

class TestActionProbability(TestCase):
    def test_basic(self):
        center = np.array([10, 10])
        right = np.array([11, 10])
        left = np.array([9, 10])
        up = np.array([10, 11])
        down = np.array([10, 9])

        dest = np.array([15, 10])  # dest is far to the right

        act_prob = lambda end: action_probability(start=center, end=end,
                dest=dest, W=100, H=100)

        pright, pleft = act_prob(right), act_prob(left)
        pup, pdown = act_prob(up), act_prob(down)

        self.assertGreater(pright, pleft)
        self.assertAlmostEqual(pup, pdown)
