from unittest import TestCase
import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from value_iter import backwards_value_iter, _calc_max_update

ni = float('-inf')

class TestUtilities(TestCase):
    def test_calc_max_update(self):
        t.assert_allclose(0, _calc_max_update(np.array([ni, ni]), np.array([ni, ni])))
        t.assert_allclose(1, _calc_max_update(np.array([0, ni]), np.array([-1, ni])))
        t.assert_allclose(float('inf'), _calc_max_update(np.array([0, 5]), np.array([-1, ni])))
        t.assert_allclose(2,
                _calc_max_update(np.array([1, 0, ni, 7]), np.array([1.1, 2, ni, 6.5])))

class TestBackwardsValueIter(TestCase):
    def test_initial(self):
        g = GridWorldMDP(3, 1, {(0, 0): -1, (0, 1): 1, (1, 1): 2, (1, 0): 3})
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 1, max_iters=0), [ni, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 2, max_iters=0), [ni, ni, 0])

    def test_easy_no_rewards(self):
        g = GridWorldMDP(3, 1, {})
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 1, max_iters=0), [ni, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 2, max_iters=0), [ni, ni, 0])

        t.assert_allclose(backwards_value_iter(g, 0, max_iters=1), [0, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 1, max_iters=1), [0, 0, 0])
        t.assert_allclose(backwards_value_iter(g, 2, max_iters=1), [ni, 0, 0])

        t.assert_allclose(backwards_value_iter(g, 0, max_iters=2),
                [np.log(2), np.log(2), 0])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=3),
                [np.log(4), np.log(5), np.log(3)])

    def test_easy_neg_one_reward(self):
        g = GridWorldMDP(3, 1, {}, default_reward=-1)
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=1), [0, -1, ni])

        V_2 = [np.log(1 + np.exp(-2)), np.log(2) - 1, -2]
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=2), V_2)

        V_3 = [np.log(np.exp(np.log(1 + np.exp(-2))) + np.exp(np.log(2) - 2)),
                np.log(np.exp(np.log(1 + np.exp(-2)) - 1) + np.exp(np.log(2) - 1)
                    + np.exp(-3)),
                np.log(np.exp(np.log(2) - 2) + np.exp(-2))]
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=3), V_3)

    def test_easy_neg_one_reward_plus_special(self):
        g = GridWorldMDP(3, 1, {(2,0): -10}, default_reward=-1)
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=1), [0, -1, ni])
        V_2 = [np.log(1 + np.exp(-2)), np.log(2) - 1, -11]
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=2), V_2)

    def test_hardmax_value_iter(self):
        g = GridWorldMDP(5, 1, {(4,0): -10}, default_reward=-1)
        value = [0, -1, -2, -3, -13]
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=0, softmax=False),
                [0, ni, ni, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=1, softmax=False),
                [0, -1, ni, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=2, softmax=False),
                [0, -1, -2, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=3, softmax=False),
                [0, -1, -2, -3, ni])
        t.assert_allclose(backwards_value_iter(g, 0, max_iters=4, softmax=False),
                [0, -1, -2, -3, -13])
        t.assert_allclose(backwards_value_iter(g, 0, softmax=False),
                [0, -1, -2, -3, -13])
