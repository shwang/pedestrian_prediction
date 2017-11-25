from __future__ import division
from unittest import TestCase
import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP
from .state import *

class TestInferFromStart(TestCase):

    def test_base_case(self):
        mdp = GridWorldMDP(3, 3, euclidean_rewards=True)
        D = np.zeros(9)
        D[0] = 1
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=0,
            all_steps=False))

    def test_uniform_easy(self):
        mdp = GridWorldMDP(3, 3)
        p = uniform = np.ones([mdp.S, mdp.A]) / mdp.A
        # p = {s: uniform for s in range(mdp.S)}

        D = D0 = np.zeros(9)
        D[0] = 1
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=0,
            all_steps=False, cached_action_prob=p))

        D = D1 = np.zeros(9)
        D[0] = 1
        D[mdp.coor_to_state(0,0)] = 1 / mdp.A
        D[mdp.coor_to_state(0,1)] = 1 / mdp.A
        D[mdp.coor_to_state(1,0)] = 1 / mdp.A
        D[mdp.coor_to_state(1,1)] = 1 / mdp.A
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=1,
            all_steps=False, cached_action_prob=p))

        D = D2 = np.zeros(9)
        D[0] = 1
        q = 1 / mdp.A / mdp.A
        D[mdp.coor_to_state(0,0)] = 4*q
        D[mdp.coor_to_state(0,1)] = 4*q
        D[mdp.coor_to_state(1,0)] = 4*q
        D[mdp.coor_to_state(1,1)] = 4*q
        D[mdp.coor_to_state(2,2)] = 1*q
        D[mdp.coor_to_state(0,2)] = 2*q
        D[mdp.coor_to_state(1,2)] = 2*q
        D[mdp.coor_to_state(2,0)] = 2*q
        D[mdp.coor_to_state(2,1)] = 2*q
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=2,
            all_steps=False, cached_action_prob=p))

        t.assert_allclose([D0, D1, D2], infer_from_start(mdp, 0, 3, T=2,
            cached_action_prob=p, all_steps=True))


   #  def test_easy(self):
   #      mdp = GridWorldMDP(3, 3)
   #      uniform = np.ones([3, 3]) / 3
   #      action_prob = {s: uniform for s in range(mdp.S)}

   #      D = np.zeros(9)
   #      D[0] = 1
   #      t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=0))

   #      D = np.zeros(9)
   #      D[0] = 1
   #      D[mdp.coor_to_state(0,0)] = (mdp.A - 3) / mdp.A
   #      D[mdp.coor_to_state(0,1)] = 1 / mdp.A
   #      D[mdp.coor_to_state(1,0)] = 1 / mdp.A
   #      D[mdp.coor_to_state(1,1)] = 1 / mdp.A
   #      t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=1))
