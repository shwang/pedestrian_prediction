from __future__ import division

from enum import IntEnum
import numpy as np
from hardmax import forwards_value_iter as _value_iter
from .mdp import MDP

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    ABSORB = 8

diagonal_actions = {Actions.UP_LEFT, Actions.UP_RIGHT, Actions.DOWN_LEFT,
        Actions.DOWN_RIGHT}

# XXX: optimize so that we don't need to convert between state and coor.
def transition_helper(g, s, a, alert_illegal=False):
    r, c = g.state_to_coor(s)
    assert a >= 0 and a < len(Actions), a

    r_prime, c_prime = r, c
    if a == Actions.LEFT:
        r_prime = r - 1
    elif a == Actions.RIGHT:
        r_prime = r + 1
    elif a == Actions.DOWN:
        c_prime = c - 1
    elif a == Actions.UP:
        c_prime = c + 1
    elif a == Actions.UP_LEFT:
        r_prime, c_prime = r - 1, c + 1
    elif a == Actions.UP_RIGHT:
        r_prime, c_prime = r + 1, c + 1
    elif a == Actions.DOWN_LEFT:
        r_prime, c_prime = r - 1, c - 1
    elif a == Actions.DOWN_RIGHT:
        r_prime, c_prime = r + 1, c - 1
    elif a == Actions.ABSORB:
        pass
    else:
        raise BaseException("undefined action {}".format(a))

    illegal = False
    if r_prime < 0 or r_prime >= g.rows or \
            c_prime < 0 or c_prime >= g.cols:
        r_prime, c_prime = r, c
        illegal = True

    s_prime = g.coor_to_state(r_prime, c_prime)

    if alert_illegal:
        return s_prime, illegal
    else:
        return s_prime

# Classic Gridworld
class GridWorldMDP(MDP):
    Actions = Actions

    def __init__(self, rows, cols, goal_state=None, euclidean_rewards=True,
            allow_wait=False, **kwargs):
        """
        An agent in a GridWorldMDP can move between adjacent/diagonal cells.

        If the agent chooses an illegal action it receives a float('-inf')
        reward and will stay in place.

        Params:
            rows [int]: The number of rows in the grid world.
            cols [int]: The number of columns in the grid world.
            goal_state [int]: (optional) The goal state at which ABSORB is legal
                and costs 0.
            euclidean_rewards [bool]: (optional) If True, then scale rewards for
                moving diagonally by sqrt(2).
            allow_wait [bool]: (optional) If False, then the ABSORB action is
                illegal in all states except the goal. If True, then the ABSORB
                action costs default_reward in states other than the goal.
        """
        if goal_state is not None:
            assert isinstance(goal_state, int)

        self.allow_wait = allow_wait
        MDP.__init__(self, rows=rows, cols=cols, A=len(Actions),
                transition_helper=self._transition_helper, **kwargs)

        S, A = self.S, self.A

        if euclidean_rewards:
            for a in diagonal_actions:
                col = self.rewards[:, a]
                np.multiply(col, np.sqrt(2), out=col)

        self.set_goal(goal_state)

    # XXX: optimize so that we don't need to convert between state and coor.
    def _transition_helper(self, s, a, alert_illegal=False):
        return transition_helper(self, s, a, alert_illegal=alert_illegal)

    def set_goal(self, goal_state):
        """
        Reconfigure the goal state in this GridWorldMDP by allowing an agent at
        the goal state to use the ABSORB action at no cost.

        If self.allow_wait is True, then at nongoal states, ABSORB has
        half the `default_reward` cost.
        If self.allow_wait is False, then at nongoal states,
        ABSORB will be illegal (i.e., incur inf cost).

        Params:
            goal_state: The new goal. Overrides previous goals.
        """
        self.goal = goal_state
        if self.allow_wait:
            self.rewards[:, Actions.ABSORB].fill(self.default_reward)
        else:
            self.rewards[:, Actions.ABSORB].fill(-np.inf)
        if goal_state != None:
            self.rewards[goal_state, Actions.ABSORB] = 0

    def q_values(self, goal_state, forwards_value_iter=_value_iter,
            goal_stuck=False):
        """
        Calculate the hardmax Q values for each state action pair.

        Params:
            goal_state [int]: The goal state, where the agent is allowed to
                choose a 0-cost ABSORB action. The goal state's value is 0.
            goal_stuck [bool]: If this is True, then all actions other than
                ABSORB are illegal in the goal_state.

        Returns:
            Q [np.ndarray]: An SxA array containing the q values
                corresponding to each (s, a) pair.
        """
        if (goal_state, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_state, goal_stuck)])

        self.set_goal(goal_state)
        V = forwards_value_iter(self, goal_state)

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in range(self.S):
            if s == goal_state:
                Q[s, Actions.ABSORB] = 0
                if goal_stuck:
                    continue
            # TODO:
            # For the purposes of Jaime/Andrea's demo, I am allowing non-ABSORB
            # actions at the goal.
            #
            # My simulations might break if human moves off this square.
            # This is something worth thinking about. XXX
            for a in range(self.A):
                Q[s,a] = self.rewards[s,a] + V[self.transition(s,a)]
        assert Q.shape == (self.S, self.A)

        self.q_cache[(goal_state, goal_stuck)] = Q
        return np.copy(Q)
