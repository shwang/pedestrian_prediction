from __future__ import division
from unittest import TestCase
import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP
from ...mdp.softmax import backwards_value_iter, forwards_value_iter
from ...util import normalize
from ...util.softmax import simulate
from .destination import infer_destination

Actions = GridWorldMDP.Actions

class TestInferDestination(TestCase):
    def make_mock_backwards_value_iter(self, V_A, V_B):
        class Closure(object):
            i = 0
        def backwards_value_iter_mock(*args, **kwargs):
            if Closure.i == 0:
                Closure.i += 1
                return V_A
            elif Closure.i == 1:
                Closure.i += 1
                return V_B
            else:
                raise AssertionError(u"Expected only two calls to backwards_value_iter")
        return backwards_value_iter_mock

    def test_easy_no_prior(self):
        g = GridWorldMDP(3, 1, {}, default_reward=-1)

        V_A = [0.340753, 0.184995, -0.90138771]
        V_B = [0.184995, 0.59443766, 0.184995]

        traj = [(0, Actions.RIGHT)]
        traj_reward = -1

        P_dest = np.zeros(3)
        P_dest[0] = np.exp(traj_reward + V_B[0] - V_A[0])
        P_dest[1] = np.exp(traj_reward + V_B[1] - V_A[1])
        P_dest[2] = np.exp(traj_reward + V_B[2] - V_A[2])
        expected_prob = normalize(P_dest)
        t.assert_allclose(expected_prob, infer_destination(g, traj,
            backwards_value_iter_fn=self.make_mock_backwards_value_iter(V_A, V_B)))

    def test_easy_with_prior(self):
        g = GridWorldMDP(3, 1, {}, default_reward=-1)

        prior = [0, 0.5, 0.5]
        V_A = [0.340753, 0.184995, -0.90138771]
        V_B = [0.184995, 0.59443766, 0.184995]
        traj = [(0, Actions.RIGHT)]
        traj_reward = -1

        P_dest = np.zeros(3)
        P_dest[0] = np.exp(traj_reward + V_B[0] - V_A[0]) * prior[0]
        P_dest[1] = np.exp(traj_reward + V_B[1] - V_A[1]) * prior[1]
        P_dest[2] = np.exp(traj_reward + V_B[2] - V_A[2]) * prior[2]
        expected_prob = normalize(P_dest)
        t.assert_allclose(expected_prob, infer_destination(g, traj, prior=prior,
            backwards_value_iter_fn=self.make_mock_backwards_value_iter(V_A, V_B)))

    def test_limited_dest_set(self):
        g = GridWorldMDP(10, 10, {}, default_reward=-9)
        start = 0
        goal = g.coor_to_state(6, 6)
        traj = simulate(g, 0, goal)
        traj = traj[:-1]  # remove potential ABSORB (bad for inference as of now)

        P_full = infer_destination(g, traj)

        P_start_only = infer_destination(g, traj, dest_set=set([0]))
        t.assert_allclose(P_start_only[0], 1)

        P_should_be_zeros = np.copy(P_start_only)
        P_should_be_zeros[0] = 0
        expected = np.zeros(P_should_be_zeros.shape)
        t.assert_allclose(expected, P_should_be_zeros)

        p_a = P_full[0]
        p_b = P_full[10]
        P_two = infer_destination(g, traj, dest_set=set([0, 10]))
        expected = np.zeros(P_two.shape)
        expected[0] = p_a / (p_a + p_b)
        expected[10] = p_b / (p_a + p_b)
        t.assert_allclose(expected, P_two)
