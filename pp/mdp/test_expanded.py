from __future__ import division
import numpy as np
from .expanded import GridWorldExpanded
from unittest import TestCase

A = GridWorldExpanded.Actions
RIGHT = A.M0
UP_RIGHT = A.M45
UP = A.M90
UP_LEFT = A.M135
LEFT = A.M180
DOWN_LEFT = A.M225
DOWN = A.M270
DOWN_RIGHT = A.M315
ABSORB = A.ABSORB

rights = [A.M345, A.M0, A.M15]
up_rights = [A.M30, A.M45, A.M60]
ups = [A.M75, A.M90, A.M105]
up_lefts = [A.M120, A.M135, A.M150]
lefts = [A.M165, A.M180, A.M195]
down_lefts = [A.M210, A.M225, A.M240]
downs = [A.M255, A.M270, A.M285]
down_rights = [A.M330, A.M315, A.M330]

class TestTransitions(TestCase):
    def test_transitions_all_legal(self):
        g = GridWorldExpanded(3, 3)
        s = g.coor_to_state(1, 1)
        for a in rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 1))
        for a in up_rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        for a in ups:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(1, 2))
        for a in up_lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(0, 2))
        for a in lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(0, 1))
        for a in down_lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(0, 0))
        for a in downs:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(1, 0))
        for a in down_rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 0))
        s_prime = g.transition(s, ABSORB)

    def test_transitions_corner(self):
        g = GridWorldExpanded(3, 3)
        s = g.coor_to_state(2, 2)
        for a in rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        for a in up_rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        for a in ups:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        for a in up_lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        for a in lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(1, 2))
        for a in down_lefts:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(1, 1))
        for a in downs:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 1))
        for a in down_rights:
            s_prime = g.transition(s, a)
            self.assertEquals(s_prime, g.coor_to_state(2, 2))
        s_prime = g.transition(s, ABSORB)
        self.assertEquals(s_prime, s)


class TestQValues(TestCase):
    def test_q_values_easy(self):
        g = GridWorldExpanded(3, 1)
        Q = g.q_values(g.coor_to_state(2, 0))

        s = g.coor_to_state(0, 0)
        self.assertAlmostEqual(Q[s, RIGHT], -1)
        self.assertAlmostEqual(Q[s, ABSORB], -2)
        self.assertEqual(Q[s, LEFT], -np.inf)
        self.assertEqual(Q[s, UP], -np.inf)

        s = g.coor_to_state(1, 0)
        self.assertAlmostEqual(Q[s, LEFT], -2)
        self.assertAlmostEqual(Q[s, ABSORB], -1)
        self.assertAlmostEqual(Q[s, RIGHT], 0)

    def test_q_values_goal_easy(self):
        g = GridWorldExpanded(3, 1)
        Q = g.q_values(g.coor_to_state(1, 0))

        s = g.coor_to_state(1, 0)
        self.assertEqual(Q[s, ABSORB], 0)

    def test_q_values_direct_vs_indirect(self):
        g = GridWorldExpanded(3, 1)
        Q = g.q_values(g.coor_to_state(2, 0))

        s = g.coor_to_state(0, 0)
        self.assertLess(Q[s, A.M15], Q[s, A.M0])

class TestTransitionProb(TestCase):

    def test_rise_beta(self):
        g = GridWorldExpanded(5, 1)
        coor = g.coor_to_state
        goal = coor(4, 0)
        s_bad, s, s_prime = coor(1, 0), coor(2, 0), coor(3, 0)

        M_1 = g.transition_probabilities(goal=goal, beta=1)
        self.assertEqual(M_1.shape, (g.S, g.S))
        M_2 = g.transition_probabilities(goal=goal, beta=2)
        self.assertEqual(M_2.shape, (g.S, g.S))

        self.assertGreater(M_1[s_prime, s], M_2[s_prime, s])
        self.assertLess(M_1[s_bad, s], M_2[s_bad, s])

    def test_change_goal(self):
        g = GridWorldExpanded(5, 1)
        coor = g.coor_to_state
        s_left, s, s_right = coor(1, 0), coor(2, 0), coor(3, 0)

        M_left = g.transition_probabilities(goal=coor(1,0))
        self.assertEqual(M_left.shape, (g.S, g.S))
        M_right = g.transition_probabilities(goal=coor(4,0))
        self.assertEqual(M_right.shape, (g.S, g.S))

        self.assertGreater(M_left[s_left, s], M_right[s_left, s])
        self.assertLess(M_left[s_right, s], M_right[s_right, s])
