from unittest import TestCase
import numpy as np
from numpy import testing as t

from mdp import GridWorldMDP
from value_iter import backwards_value_iter

ni = float('-inf')

class TestBackwardsValueIter(TestCase):
    def test_initial(self):
        g = GridWorldMDP(3, 1, {(0, 0): -1, (0, 1): 1, (1, 1): 2, (1, 0): 3})
        t.assert_allclose(backwards_value_iter(g, 0, iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 1, iters=0), [ni, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 2, iters=0), [ni, ni, 0])

    def test_easy_no_rewards(self):
        g = GridWorldMDP(3, 1, {})
        t.assert_allclose(backwards_value_iter(g, 0, iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 1, iters=0), [ni, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 2, iters=0), [ni, ni, 0])

        t.assert_allclose(backwards_value_iter(g, 0, iters=1), [0, 0, ni])
        t.assert_allclose(backwards_value_iter(g, 1, iters=1), [0, 0, 0])
        t.assert_allclose(backwards_value_iter(g, 2, iters=1), [ni, 0, 0])

        t.assert_allclose(backwards_value_iter(g, 0, iters=2),
                [np.log(2), np.log(2), 0])
        t.assert_allclose(backwards_value_iter(g, 0, iters=3),
                [np.log(4), np.log(5), np.log(3)])
        t.assert_allclose(backwards_value_iter(g, 0), [np.log(4), np.log(5), np.log(3)])

    def test_easy_neg_one_reward(self):
        g = GridWorldMDP(3, 1, {}, default_reward=-1)
        t.assert_allclose(backwards_value_iter(g, 0, iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, iters=1), [0, -1, ni])

        V_2 = [np.log(1 + np.exp(-2)), np.log(2) - 1, -2]
        t.assert_allclose(backwards_value_iter(g, 0, iters=2), V_2)

        V_3 = [np.log(np.exp(np.log(1 + np.exp(-2))) + np.exp(np.log(2) - 2)),
                np.log(np.exp(np.log(1 + np.exp(-2)) - 1) + np.exp(np.log(2) - 1)
                    + np.exp(-3)),
                np.log(np.exp(np.log(2) - 2) + np.exp(-2))]
        t.assert_allclose(backwards_value_iter(g, 0, iters=3), V_3)
        t.assert_allclose(backwards_value_iter(g, 0), V_3)

    def test_easy_neg_one_reward_plus_special(self):
        g = GridWorldMDP(3, 1, {(2,0): -10}, default_reward=-1)
        t.assert_allclose(backwards_value_iter(g, 0, iters=0), [0, ni, ni])
        t.assert_allclose(backwards_value_iter(g, 0, iters=1), [0, -1, ni])
        V_2 = [np.log(1 + np.exp(-2)), np.log(2) - 1, -11]
        t.assert_allclose(backwards_value_iter(g, 0, iters=2), V_2)
