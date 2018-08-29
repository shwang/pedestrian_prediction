from __future__ import division
from unittest import TestCase
import numpy as np
from numpy import testing as t

from ...mdp import GridWorldMDP
from .state import *

class TestInferJoint(TestCase):
    def test_classic_nocrash(self):
        g = GridWorldMDP(10, 10)
        coor = g.coor_to_state
        dests = [coor(7, 7), coor(2, 2), coor(5, 0)]
        betas = [1, 2, 3, 4]
        T = 5
        traj = [(coor(1, 1), g.Actions.RIGHT), (coor(2, 1), g.Actions.UP)]
        infer_joint(g=g, dests=dests, betas=betas, T=T, use_gridless=False,
                traj=traj)

    def test_gridless_nocrash(self):
        g = GridWorldExpanded(10, 10)
        coor = g.coor_to_state
        dests = [coor(7, 7), coor(2, 2), coor(5, 0)]
        betas = [1, 2, 3, 4]
        T = 5
        traj = [(1, 1), (2, 1), (3, 1), (4.5, 6)]
        infer_joint(g=g, dests=dests, betas=betas, T=T, use_gridless=True,
                traj=traj)

class TestInferBayesBeta(TestCase):

    def test_trivial(self):
        g = GridWorldMDP(3, 3)
        betas = [0.1, 1, 2]
        init_state = 2
        dest = 4
        T = 0

        expect = np.zeros([1, g.S])
        expect[0, init_state] = 1

        occ_res, occ_all, P_beta = infer_bayes(g, init_state=init_state,
                dest=dest, T=T, betas=betas, verbose_return=True)

        t.assert_allclose(P_beta, np.ones(len(betas))/len(betas))
        for occ_beta in occ_all:
            t.assert_allclose(occ_beta, expect)
        t.assert_allclose(occ_res, expect)


    def test_suboptimal(self):
        g = GridWorldMDP(4, 4)
        betas = [0.1, 1]
        init_state = g.coor_to_state(1, 1)
        traj = [(init_state, g.Actions.DOWN_RIGHT)]
        s_prime = g.transition(init_state, g.Actions.DOWN_RIGHT)
        dest = g.coor_to_state(3, 3)
        T = 0

        occ_res, occ_all, P_beta = infer_bayes(g, traj=traj, dest=dest,
                T=T, betas=betas, verbose_return=True)

        # Confirm that P(beta=0.1) < P(beta=1).
        assert P_beta[0] < P_beta[1], P_beta

        expect = np.zeros([1, g.S])
        expect[0, s_prime] = 1

        for occ_beta in occ_all:
            t.assert_allclose(occ_beta, expect)
        t.assert_allclose(occ_res, expect)


    def test_optimal(self):
        g = GridWorldMDP(4, 4)
        betas = [0.1, 1, 10]
        init_state = g.coor_to_state(1, 1)
        traj = [(init_state, g.Actions.UP_RIGHT)]
        s_prime = g.transition(init_state, g.Actions.UP_RIGHT)
        dest = g.coor_to_state(3, 3)
        T = 0

        occ_res, occ_all, P_beta = infer_bayes(g, traj=traj, dest=dest,
                T=T, betas=betas, verbose_return=True)

        # Confirm that P(beta=0.1) > P(beta=1) > P(beta=10)
        assert P_beta[0] > P_beta[1] > P_beta[2], P_beta

        expect = np.zeros([1, g.S])
        expect[0, s_prime] = 1

        for occ_beta in occ_all:
            t.assert_allclose(occ_beta, expect)
        t.assert_allclose(occ_res, expect)


class TestInferFromStart(TestCase):
    def test_base_case(self):
        mdp = GridWorldMDP(3, 3, euclidean_rewards=True)
        mdp.set_goal(4)
        D = np.zeros(9)
        D[0] = 1
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=0,
            verbose_return=False))

    def test_uniform_easy(self):
        mdp = GridWorldMDP(3, 3)
        p = uniform = np.ones([mdp.S, mdp.A]) / mdp.A
        # p = {s: uniform for s in range(mdp.S)}

        D = D0 = np.zeros(9)
        D[0] = 1
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=0,
            verbose_return=False, cached_action_probs=p))

        D = D1 = np.zeros(9)
        D[0] = 1
        D[mdp.coor_to_state(0,0)] = (mdp.A - 3) / mdp.A
        D[mdp.coor_to_state(0,1)] = 1 / mdp.A
        D[mdp.coor_to_state(1,0)] = 1 / mdp.A
        D[mdp.coor_to_state(1,1)] = 1 / mdp.A
        t.assert_allclose(D, infer_from_start(mdp, 0, 3, T=1,
            verbose_return=False, cached_action_probs=p))
        t.assert_allclose([D0, D1], infer_from_start(mdp, 0, 3, T=1,
            cached_action_probs=p, verbose_return=True)[0])

    # Closely related to `test_clockwise` in test_hardmax.py
    def test_clockwise_multistep(self):
        g = GridWorldMDP(2, 2)
        A = g.coor_to_state(0, 1)
        B = g.coor_to_state(1, 1)
        C = g.coor_to_state(1, 0)
        D = g.coor_to_state(0, 0)

        P = np.zeros([g.S, g.A])
        P[A, g.Actions.RIGHT] = 1
        P[B, g.Actions.DOWN] = 1
        P[C, g.Actions.LEFT] = 1
        P[D, g.Actions.UP] = 1

        res = infer_simple(g, A, 0, T=4, action_prob=P)

        expect = np.zeros([5, 4])
        expect[0, A] = 1
        expect[1, B] = 1
        expect[2, C] = 1
        expect[3, D] = 1
        expect[4, A] = 1

        t.assert_allclose(res, expect)

    def test_infer_multidest_no_crash(self):
        mdp = GridWorldMDP(3, 3)
        p = uniform = np.ones([mdp.S, mdp.A]) / mdp.A

        D = D1 = np.zeros(9)
        D[0] = 1
        D[mdp.coor_to_state(0,0)] = 1 / mdp.A
        D[mdp.coor_to_state(0,1)] = 1 / mdp.A
        D[mdp.coor_to_state(1,0)] = 1 / mdp.A
        D[mdp.coor_to_state(1,1)] = 1 / mdp.A
        traj = [(4,4)]
        infer(mdp, traj, [0, 1], T=1, verbose_return=False,
                cached_action_probs=p)

class TestInferMultiDest(TestCase):
    def test_different_beta(self):
        g = GridWorldMDP(24, 24, euclidean_rewards=True)
        D = np.zeros(9)
        D[0] = 1
        dest_list = [g.coor_to_state(23, 10), g.coor_to_state(10, 23)]
        # traj = [(g.coor_to_state(0, 0), g.Actions.UP_RIGHT)]

        from ...util.hardmax.simulate import simulate
        traj = simulate(g, 0, g.coor_to_state(20, 0), beta=0.1)

        P, betas, dest_probs = infer(g, traj=traj[:10], dest_or_dests=dest_list,
                T=10)

        assert betas[0] != betas[1]
