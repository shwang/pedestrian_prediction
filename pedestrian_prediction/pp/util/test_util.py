from unittest import TestCase
import numpy as np
from numpy import testing as t

from .util import normalize, display, sum_rewards
from ..mdp import GridWorldMDP

Actions = GridWorldMDP.Actions

class TestUtilities(TestCase):
    def test_sum_rewards(self):
        g = GridWorldMDP(3, 1, reward_dict={(2,0): -10}, default_reward=-1)
        traj_1 = [(0, Actions.ABSORB)] * 10
        t.assert_allclose(-np.inf, sum_rewards(g, traj_1))

        traj_2 = [(0, Actions.RIGHT), (1, Actions.RIGHT)]
        t.assert_allclose(-11, sum_rewards(g, traj_2))

        traj_3 = [(0, Actions.RIGHT), (1, Actions.RIGHT), (2, Actions.RIGHT)]
        t.assert_allclose(-np.inf, sum_rewards(g, traj_3))

    def test_normalize(self):
        t.assert_allclose([1], normalize([1]))
        t.assert_allclose([1], normalize([0.12351]))
        t.assert_allclose([1/3, 1/3, 1/3], normalize([1, 1, 1]))
        t.assert_allclose([0, 2/3, 1/3], normalize([0, 2, 1]))

