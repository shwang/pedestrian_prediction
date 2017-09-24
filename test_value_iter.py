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

    def check_value_iter(self, g, expected, max_iters=None, softmax=False, init_state=0):
        t.assert_allclose(backwards_value_iter(g, init_state, max_iters=max_iters,
            softmax=softmax), expected)

    def test_hardmax_value_iter(self):
        g = GridWorldMDP(5, 1, {(4,0): -10}, default_reward=-1)
        self.check_value_iter(g, [0, ni, ni, ni, ni], max_iters=0)
        self.check_value_iter(g, [0, -1, ni, ni, ni], max_iters=1)
        self.check_value_iter(g, [0, -1, -2, ni, ni], max_iters=2)
        self.check_value_iter(g, [0, -1, -2, -3, ni], max_iters=3)
        self.check_value_iter(g, [0, -1, -2, -3, -13], max_iters=4)
        self.check_value_iter(g, [0, -1, -2, -3, -13], max_iters=5)
        self.check_value_iter(g, [0, -1, -2, -3, -13])

    def test_hardmax_value_iter_complex(self):
        reward_grid = [[-1, -1, -1, -2, -2],
                       [-1, -2, -2, -2, -2],
                       [-1, -2, -5, -4, -3],
                       [-1, -2, -2, -1, -3],
                       [-1, -1, -1, -3, -3]]
        reward_dict = {}
        for x in range(5):
            for y in range(5):
                reward_dict[(x,y)] = reward_grid[x][y]

        g = GridWorldMDP(5, 5, reward_dict, default_reward=-1)

        V_0 = np.empty([5,5])
        V_0.fill(ni)
        V_0[0, 0] = 0
        self.check_value_iter(g, np.ravel(V_0), max_iters=0)

        V_1 = [[ 0, -1, ni, ni, ni],
               [-1, -2, ni, ni, ni],
               [ni] * 5,
               [ni] * 5,
               [ni] * 5]
        self.check_value_iter(g, np.ravel(V_1), max_iters=1)

        V_2 = [[ 0, -1, -2, ni, ni],
               [-1, -2, -3, ni, ni],
               [-2, -3, -7, ni, ni],
               [ni] * 5,
               [ni] * 5]
        self.check_value_iter(g, np.ravel(V_2), max_iters=2)

        V_3 = [[ 0, -1, -2, -4, ni],
               [-1, -2, -3, -4, ni],
               [-2, -3, -7, -7, ni],
               [-3, -4, -5, -8, ni],
               [ni] * 5]
        self.check_value_iter(g, np.ravel(V_3), max_iters=3)

        V_4 = [[ 0, -1, -2, -4, -6],
               [-1, -2, -3, -4, -6],
               [-2, -3, -7, -7, -7],
               [-3, -4, -5, -6, -10],
               [-4, -4, -5, -8, -11]]
        self.check_value_iter(g, np.ravel(V_4), max_iters=4)

        V_5 = [[ 0, -1, -2, -4, -6],
               [-1, -2, -3, -4, -6],
               [-2, -3, -7, -7, -7],
               [-3, -4, -5, -6, -9],
               [-4, -4, -5, -8, -9]]
        self.check_value_iter(g, np.ravel(V_5), max_iters=5)
        self.check_value_iter(g, np.ravel(V_5), max_iters=6)
        self.check_value_iter(g, np.ravel(V_5), max_iters=None)
