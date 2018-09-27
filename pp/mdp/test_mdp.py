from __future__ import absolute_import
from unittest import TestCase
import numpy as np

from .classic import GridWorldMDP

ni = -np.inf
Actions = GridWorldMDP.Actions

class TestGridWorldMDP(TestCase):
    def test_state_coor_switching(self):
        self.state_coor_helper(1, 1, 0, 0, 0)

        self.state_coor_helper(2, 3, 0, 0, 0)
        self.state_coor_helper(2, 3, 1, 0, 1)
        self.state_coor_helper(2, 3, 2, 0, 2)
        self.state_coor_helper(2, 3, 3, 1, 0)
        self.state_coor_helper(2, 3, 4, 1, 1)
        self.state_coor_helper(2, 3, 5, 1, 2)

        self.state_coor_helper(1, 1, 0, 0, 0)
        self.state_coor_helper(1, 3, 0, 0, 0)
        self.state_coor_helper(1, 3, 1, 0, 1)
        self.state_coor_helper(1, 3, 2, 0, 2)

    class DummyMDP(GridWorldMDP):
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols

    def state_coor_helper(self, rows, cols, s, r, c):
        mdp = self.DummyMDP(rows, cols)
        self.assertEqual((r, c), GridWorldMDP.state_to_coor(mdp, s))
        self.assertEqual(s, GridWorldMDP.coor_to_state(mdp, r, c))

    def test_num_states(self):
        self.assertEqual(25, GridWorldMDP(5, 5).S)
        self.assertEqual(1, GridWorldMDP(1, 1).S)
        self.assertEqual(3, GridWorldMDP(1, 3).S)
        self.assertEqual(10, GridWorldMDP(5, 2).S)

    def test_num_actions(self):
        self.assertEqual(len(GridWorldMDP.Actions), GridWorldMDP(5, 5).A)
        self.assertEqual(len(GridWorldMDP.Actions), GridWorldMDP(1, 1).A)
        self.assertEqual(len(GridWorldMDP.Actions), GridWorldMDP(1, 3).A)
        self.assertEqual(len(GridWorldMDP.Actions), GridWorldMDP(5, 2).A)

    def assert_illegality(self, mdp, r, c, illegal_list):
        s = mdp.coor_to_state(r, c)
        for a in list(Actions):
            should_be_illegal = a in illegal_list
            s_prime, illegal = mdp._transition_helper(s, a, alert_illegal=True)
            self.assertEqual(should_be_illegal, illegal, a)
            if should_be_illegal:
                self.assertEqual(s, s_prime)

    def test_transitions_illegal(self):
        g = GridWorldMDP(1, 1)
        self.assert_illegality(g, 0, 0, set(Actions) - set([Actions.ABSORB]))

        g = GridWorldMDP(3, 3)
        illegal = [Actions.DOWN, Actions.LEFT, Actions.UP_LEFT, Actions.DOWN_LEFT,
                Actions.DOWN_RIGHT]
        self.assert_illegality(g, 0, 0, illegal)
        self.assert_illegality(g, 1, 1, {})
        self.assert_illegality(g, 1, 2,
                [Actions.UP, Actions.UP_LEFT, Actions.UP_RIGHT])
        self.assert_illegality(g, 2, 2,
                set(Actions) - set([Actions.LEFT, Actions.DOWN, Actions.DOWN_LEFT,
                    Actions.ABSORB]))

    def assert_transition(self, mdp, r, c, a, r_prime, c_prime):
        s = mdp.coor_to_state(r, c)
        s_prime = mdp.transition(s, a)
        self.assertEqual((r_prime, c_prime), mdp.state_to_coor(s_prime))

    def test_transitions(self):
        g = GridWorldMDP(3, 3)
        self.assert_transition(g, 0, 0, Actions.RIGHT, 1, 0)
        self.assert_transition(g, 0, 0, Actions.UP, 0, 1)
        self.assert_transition(g, 0, 0, Actions.UP_RIGHT, 1, 1)
        self.assert_transition(g, 0, 0, Actions.ABSORB, 0, 0)

        self.assert_transition(g, 1, 1, Actions.LEFT, 0, 1)
        self.assert_transition(g, 1, 1, Actions.RIGHT, 2, 1)
        self.assert_transition(g, 1, 1, Actions.DOWN, 1, 0)
        self.assert_transition(g, 1, 1, Actions.UP, 1, 2)
        self.assert_transition(g, 1, 1, Actions.DOWN_LEFT, 0, 0)
        self.assert_transition(g, 1, 1, Actions.UP_LEFT, 0, 2)
        self.assert_transition(g, 1, 1, Actions.DOWN_RIGHT, 2, 0)
        self.assert_transition(g, 1, 1, Actions.UP_RIGHT, 2, 2)
        self.assert_transition(g, 1, 1, Actions.ABSORB, 1, 1)

    def assert_reward(self, mdp, r, c, a, reward):
        s = mdp.coor_to_state(r, c)
        self.assertEqual(reward, mdp.rewards[s, a])

    def test_rewards_illegal(self):
        g = GridWorldMDP(3, 3)
        self.assert_reward(g, 0, 0, Actions.LEFT, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN_LEFT, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN_RIGHT, ni)
        self.assert_reward(g, 0, 0, Actions.UP_LEFT, ni)

        g = GridWorldMDP(3, 3, reward_dict={(0, 0): 1})
        self.assert_reward(g, 0, 0, Actions.LEFT, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN_LEFT, ni)
        self.assert_reward(g, 0, 0, Actions.DOWN_RIGHT, ni)
        self.assert_reward(g, 0, 0, Actions.UP_LEFT, ni)

    def test_rewards(self):
        g = GridWorldMDP(3, 3,
                reward_dict={(0, 0): -1, (0, 1): 1, (1, 1): 2, (1, 0): 3},
                euclidean_rewards=False)
        self.assert_reward(g, 0, 0, Actions.ABSORB, ni)
        g.set_goal(0)
        self.assert_reward(g, 0, 0, Actions.ABSORB, 0)
        self.assert_reward(g, 0, 0, Actions.UP, 1)
        self.assert_reward(g, 0, 0, Actions.UP_RIGHT, 2)
        self.assert_reward(g, 0, 0, Actions.RIGHT, 3)
        self.assert_reward(g, 2, 2, Actions.DOWN_LEFT, 2)
        self.assert_reward(g, 1, 1, Actions.DOWN_LEFT, -1)
        self.assert_reward(g, 1, 0, Actions.LEFT, -1)

    def test_rewards_defaults(self):
        g = GridWorldMDP(3, 3,
                reward_dict={(0, 0): -1, (0, 1): 1, (1, 1): 2, (1, 0): 3},
                default_reward=1.5, euclidean_rewards=False)
        self.assert_reward(g, 0, 0, Actions.ABSORB, ni)
        self.assert_reward(g, 0, 0, Actions.UP, 1)
        self.assert_reward(g, 0, 0, Actions.UP_RIGHT, 2)
        self.assert_reward(g, 0, 0, Actions.RIGHT, 3)
        self.assert_reward(g, 2, 2, Actions.DOWN_LEFT, 2)
        self.assert_reward(g, 1, 1, Actions.DOWN_LEFT, -1)
        self.assert_reward(g, 1, 0, Actions.LEFT, -1)
        self.assert_reward(g, 1, 1, Actions.UP, 1.5)
