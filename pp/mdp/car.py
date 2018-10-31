from __future__ import division

import numpy as np
from .mdp import MDP
from enum import IntEnum

class Actions(IntEnum):
    FORWARD = 0
    FORWARD_CCW = 1
    FORWARD_CW = 2
    ABSORB = 3

class CarMDP(MDP):
    Actions = Actions

    def __init__(self, X, Y, T, vel=1.0, allow_wait=True, **kwargs):
        """
        Params:
            X [int] -- The width of this MDP.
            Y [int] -- The height of this MDP.
            T [int] -- The number of different angles, uniformly spaced,
                allowed in this MDP. It's recommended for T to be a power of 2
                (avoid asymmetries) and to be at least 4.
            vel [float] -- The arc length traveled, provided that the action
                involves changing position (x, y). `vel` is in units of
                gridsquares per timestep. It's recommended that `vel` is at
                least 1.
            allow_wait [bool] -- If this is True, then ABSORB is allowed on
                at every state. If this is False, then ABSORB is illegal except
                on goal states.
        """

        # TODO: Future param suggestion:
        """
        Right now the we can only increment the angle `t` coordinate by +/- 1
        every timestep. (With FORWARD_CW and FORWARD_CCW actions; this
        corresponds to increasing or decreasing the angle by 2*pi/T). Maybe we
        want to increase the number of turning actions.

        Future Param:
        max_angle_change [int] -- An integer between 1 and T//2 inclusive.
            This MDP will have 2*max_angle_change turning actions.
        """

        if not allow_wait:
            print("warning: if allow_wait is False, then some states might"
                    + " have no legal actions!")

        if 'reward_dict' in kwargs:
            raise NotImplementedError

        assert isinstance(X, int), X
        assert isinstance(Y, int), Y
        assert isinstance(T, int), T
        assert X > 0
        assert Y > 0
        assert T > 0
        assert vel > 0

        self.X = X
        self.Y = Y
        self.T = T
        self.vel = vel
        self.allow_wait = allow_wait
        S = X * Y * T

        self.q_cache = {}

        # Intentionally set default_reward to nan here. The idea is that we
        # shouldn't be looking at the self.reward method and instead rely
        # entirely on self.score for determining q_values.
        MDP.__init__(self, S=S, A=len(Actions),
                transition_helper=self._transition_helper,
                default_reward=np.nan, **kwargs)


    def _transition_helper_real(self, x, y, t, a):
        """
        Given coordinates in continuous space and a initial angle (in radians),
        calculate the new position and angle after performing action `a`.

        This method is independent of the Gridworld.

        Params:
        x, y [float] -- x and y continuous coordinates representing the position.
            We establish the convention of mapping the discrete coordinates
            [3, 5] to continuous coordinates [3.5, 5.5].
        t [float] -- The initial angle, in radians. 0 radians points East, with
            positive radians turning counterclockwise.
        a [int] -- The action. (Selected from the Actions enum defined in this
            file.)

        Return:
        x_new, y_new, t_new -- The new coordinates and angle in continuous space.
                `t_new` is guaranteed to be in the range [0, 2*pi).
        """
        assert 0.0 <= x < self.X + 1
        assert 0.0 <= y < self.Y + 1
        assert 0.0 <= t < 2*np.pi, t
        assert 0 <= a < self.A

        if a == Actions.ABSORB:
            return x, y, t
        elif a == Actions.FORWARD:
            x_new = x + self.vel * np.cos(t)
            y_new = y + self.vel * np.sin(t)
            t_new = t
        else:
            if a == Actions.FORWARD_CCW:
                ang_vel = 2*np.pi/self.T
            if a == Actions.FORWARD_CW:
                ang_vel = -2*np.pi/self.T
            x_new = x + self.vel/ang_vel * (np.sin(t + ang_vel) - np.sin(t))
            y_new = y - self.vel/ang_vel * (np.cos(t + ang_vel) - np.cos(t))
            t_new = (t + ang_vel) % (2*np.pi)
        return x_new, y_new, t_new

    def _transition_helper(self, s, a, alert_illegal=False):
        if not alert_illegal:
            raise NotImplementedError

        x, y, t = self.state_to_real(s)
        illegal = False

        try:
            x_new, y_new, t_new = self._transition_helper_real(x, y, t, a)
            s = self.real_to_state(x_new, y_new, t_new)
        except ValueError:  # Assumption: ValueError iff action is illegal.
            illegal = True
        return s, illegal


    def score(self, gx, gy, gt, sx, sy, st):
        """
        This is the function used to calculate the Q-value of state-action
        pairs that don't have the special status of BLOCKED or GOAL.

        Right now this function just returns the negative squared Euclidean
        distance between the positions of the two states. Their angles are not
        considered.

        gx, gy [float] -- The continuous position coordinates of the goal.
        gt [float] -- The continuous angle of the goal, in radians. CURRENTLY
            UNUSED.
        sx, sy [float] -- The position coordinates of the transition state
            The transition state is the state that results from the state-action
            pair whose Q value we are calculating.
        st [float] -- The continuous angle of the transition state.
            The transition state is the state that results from the state-action
            pair whose Q value we are calculating. CURRENTLY UNUSED.
        """
        # TODO: We could imagine incorporating the angle into this score
        # function. Also note that MDPs with blocking tiles will have nontrivial
        # Q_values for each state, which probably require some planning to
        # calculate.
        return -((gx - sx)**2 + (gy - sy)**2)


    def is_goal(self, s, goal_spec):
        """
        Return whether s is a goal state.

        Params:
        goal_spec [tuple(int, int)] -- The (x, y) coordinates of the goal.

        Return:
        True iff the position coordinate of goal_spec matches the position of s.
        """
        goal_x, goal_y = goal_spec
        x, y, t = self.state_to_coor(s)

        assert 0 <= goal_x < self.X
        assert 0 <= goal_y < self.Y
        assert isinstance(goal_x, int)
        assert isinstance(goal_y, int)

        return (x, y) == (goal_x, goal_y)


    def is_blocked(self, s):
        """
        Returns True if s is blocked.

        By default, state-action pairs that lead to blocked states are illegal.
        Rewards are undefined for state-action pairs starting from blocked
        states.
        """
        # TODO
        return False


    def q_values(self, goal_spec, goal_stuck=True):
        """
        Calculate the Q values for each state action pair using `self.score()`.

        Any action moving into a state that `is_blocked()` will be have value
        -infinity. Any illegal action will have value -infinity.

        Behavior is undefined if any goal state is blocked.

        Param:
        goal_spec [tuple(int, int)] -- The (x, y) coordinates of the goal. An
            AssertionError will fire if these coordinates are invalid.
            At goal states, the agent is allowed to
            choose a 0-cost ABSORB action.
        goal_stuck [bool] -- If True, then force ABSORB action at the goal by
            making all other actions worth -infinity.

        Return:
        Q [np.ndarray([S, A]) -- The Q values of each state action pair.
        """

        # TODO:
        # Use value iteration instead, like in mdp/classic.py:q_values()
        # Note that value iteration depends on self.rewards, an array
        # containing the immediate reward of each state-action pair.
        #
        # You can find the definition of rewards in mdp/mdp.py:39.
        #
        # I think we can't directly use mdp/hardmax/hardmax.py:forwards_value_iter()
        # for the following reason. Our new problem formulation might expect
        # multiple goal states. However, forwards_value_iter expects a single
        # goal state.
        # -- Therefore, we need to edit this line of _value_iter() so that we
        #   can initialize multiple state values to 0:
        # ```
        # pq.put((0, s))
        # ```
        #

        # TODO:
        # Update goal_spec so that we can specify angles. Right now, goal_spec
        # only specifies (x, y). Modifying goal_spec would require a change to
        # self.is_goal(goal_spec).

        # INFO:
        # To perform occupancy inference from the trajectory-so-far, we call
        # `infer_joint(mdp, dests, betas, T, use_gridless=False,
        # verbose_return=True, priors=dest_beta_prob...)` as usual.
        #
        # Note that, as described in the `infer_joint` documentation, `dests`
        # here will no longer be a list of state numbers, but instead a list of
        # `goal_spec` tuples.
        #
        # Note that since our input trajectories now take the form of
        # state-action pairs
        # instead of (x, y) coordinates, we set `use_gridless=False`.
        #
        # I saw that in crazyflie_human/src/human_pred.py, #OPTION2 (recursive
        # update) inputs traj=traj[-2:]. I believe that if we want to use a
        # recursive update for `infer_joint`, we will use `traj=traj[-1:]`.
        # Since a single state-action pair [(s, a)] describes a pair of
        # positions [(x1, y1), (x2, y2)].

        if (goal_spec, goal_stuck) in self.q_cache:
            return np.copy(self.q_cache[(goal_spec, goal_stuck)])

        Q = np.empty([self.S, self.A])
        Q.fill(-np.inf)
        for s in range(self.S):
            # Easy case: If already at goal, then we just force ABSORB action.
            at_goal = self.is_goal(s, goal_spec)

            if at_goal:
                Q[s, Actions.ABSORB] = 0
                if goal_stuck:
                    # All other actions will be by default -inf.
                    continue

            # Convert goal spec to real coordinates for use by score function.
            gx, gy = goal_spec
            gx += 0.5
            gy += 0.5
            gt = 0

            for a in range(self.A):
                sx, sy, st = self.state_to_real(s)
                illegal = False
                try:
                    sx_new, sy_new, st_new = self._transition_helper_real(
                            sx, sy, st, a)
                    s_new = self.real_to_state(sx_new, sy_new, st_new)
                except ValueError:
                    # Illegal transition
                    illegal = True

                if illegal or self.is_blocked(s_new):
                    Q[s, a] = -np.inf
                elif (a == Actions.ABSORB and not at_goal
                        and not self.allow_wait):
                    Q[s, a] = -np.inf
                elif at_goal and a == Actions.ABSORB:
                    Q[s, a] = 0
                else:
                    Q[s, a] = self.score(gx, gy, gt, sx, sy, st)

        self.q_cache[(goal_spec, goal_stuck)] = Q
        return np.copy(Q)


    #################################
    # Conversion functions
    #################################
    # These helper functions convert between state number ("state"), discrete
    # coordinates ("coor"), and continuous coordinates ("real").
    #
    # State number ("state"):
    # A state `s` is an integer such that 0 <= s < self.S.
    #
    # Discrete coordinates ("coor"):
    # `x` is an integer such that 0 <= x < self.X. Increasing `x` corresponds to
    #       moving east.
    # `y` is an integer such that 0 <= y < self.Y. Increasing `y` corresponds to
    #       moving north.
    # `t`, the angle coordinate, is an integer such that 0 <= t < self.T.
    #       An angle coordinate of `t` corresponds to a continuous angle
    #       (2*pi / self.T) * t. At `t = 0`, the agent is facing east, and
    #       increasing `t` corresponds to counter-clockwise turning.
    #
    # Continuous coordinates ("real"):
    # `x` is a float between 0 <= x < self.X + 1.
    # `y` is a float between 0 <= y < self.Y + 1.
    # `t`, the angle, is a float between 0 <= t < 2*pi.
    ###

    def state_to_coor(self, s):
        """
        Params:
            s [int] -- The state.
        Returns:
            x, y, t -- The discrete coordinates corresponding to s.
        """
        assert isinstance(s, int)
        assert 0 <= s < self.S
        t = s % self.T
        y = (s % (self.Y * self.T)) // self.T
        x = s // (self.Y * self.T)
        return x, y, t

    def real_to_state(self, x, y, t):
        x, y, t = self.real_to_coor(x, y, t)
        return self.coor_to_state(x, y, t)

    def coor_to_state(self, x, y, t):
        """
        Convert discrete coordinates into a state, if that state exists.
        If no such state exists, raise a ValueError.

        Params:
        x, y [int or float] -- The discrete x, y coordinates of the state.
        t [int] -- The discrete angle coordinate.

        Returns:
        s [int] -- The state.
        """

        x, y = int(x), int(y)
        if not(0 <= x < self.X):
            raise ValueError(x, self.X)
        if not (0 <= y < self.Y):
            raise ValueError(y, self.Y)

        assert isinstance(t, int)
        assert 0 <= t < self.T

        return (x * self.Y * self.T) + (y * self.T) + (t)

    def state_to_real(self, s):
        """
        Params:
            s -- The state.
        Returns:
            x, y, t -- The continuous coordinates corresponding to s.
        """
        x, y, t = self.state_to_coor(s)
        return self.coor_to_real(x, y, t)

    def real_to_state(self, x, y, t):
        """
        Project real coordinates onto a state if that state exists. If no such
        state exists, raise a ValueError.

        Params:
        x, y (float) -- Continuous coordinates.
        t (float) -- The continuous angle.
        Returns:
            s -- The state corresponding to the arguments.
        """
        x_dis, y_dis, t_dis = self.real_to_coor(x, y, t)
        return self.coor_to_state(x_dis, y_dis, t_dis)

    def real_to_coor(self, x, y, t, tol=1e-5):
        """
        Project real coordinates onto discrete state space.

        Return:
        x_dis, y_dis, t_dis [int]
        """

        t %= 2*np.pi
        increment = 2*np.pi / self.T

        if tol is not None:
            rem = t % increment
            if min(rem, increment - rem) > tol:
                raise ValueError("Invalid angle: t={}".format(t))
                int(round(4.9))

        x_dis, y_dis = int(x), int(y)
        t_dis = int(round(t / increment))

        assert 0 <= t_dis < self.T, t_dis
        if not (0 <= x_dis < self.X):
            raise ValueError(x_dis, self.X)
        if not (0 <= y_dis < self.Y):
            raise ValueError(y_dis, self.Y)

        return int(x), int(y), t_dis

    def coor_to_real(self, x, y, t):
        """
        Convert discrete coordinates into continuous coordinates.
        """
        assert 0 <= x < self.X
        assert 0 <= y < self.Y
        assert 0 <= t < self.T

        x_real, y_real = x+0.5, y+0.5
        theta_increment = 2*np.pi / self.T
        theta = theta_increment * t
        return x_real, y_real, theta
