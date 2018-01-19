from __future__ import division

from unittest import TestCase 
import numpy as np
from numpy import testing as t

from .simulate import *
from .context import HRContext
from ..mdp import GridWorldMDP

# Just a bunch of no-crash tests.

class TestSimulate(TestCase):
    @classmethod
    def setUpClass(cls):
        g_R = GridWorldMDP(6, 6)
        g_R.set_goal(35)
        g_H = GridWorldMDP(6, 6)
        g_H.set_goal(2)
        cls.safe_traj_H = [(g_H.coor_to_state(3, 3), g_H.Actions.UP)]
        cls.ctx = HRContext(g_R=g_R, goal_R=35, g_H=g_H, goal_H=2,
                collide_radius=1, collide_penalty=10, traj_H=cls.safe_traj_H,
                start_H=0, start_R=3)
        cls.traj = [(0, 1), (1, 2), (3, 4)]

    def test_no_crash_vanilla(self):
        simulate_vanilla(self.ctx)
        simulate_vanilla(self.ctx, k=2)
        simulate_vanilla(self.ctx, calc_beta=False, k=2)
        simulate_vanilla(self.ctx, calc_beta=False)

    # IGNORE THESE TESTS FOR NOW
    # def test_no_crash_bayes(self):
    #     simulate_bayes(self.ctx, betas=[1])
    #     simulate_bayes(self.ctx, betas=[1, 0.1])
    #     simulate_bayes(self.ctx, betas=[1, 0.1], priors=[0.3, 0.7])

    def test_no_crash_vanilla_subplots(self):
        sim_res = simulate_vanilla(self.ctx, k=2)
        sim_res.gen_all_subplots()

        sim_res = simulate_vanilla(self.ctx, calc_beta=False)
        sim_res.gen_all_subplots()

    # def test_no_crash_bayes_subplots(self):
    #     sim_res = simulate_bayes(self.ctx, betas=[1, 0.1])
    #     sim_res.gen_all_subplots()
