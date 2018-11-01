from __future__ import division

import numpy as np
from pp.mdp.mdp import MDP
from enum import IntEnum
from pp.mdp.hardmax import hardmax

class Actions(IntEnum):
    FORWARD = 0
    FORWARD_CCW1 = 1
    FORWARD_CCW2 = 2
    FORWARD_CCW3 = 3
    FORWARD_CW1 = 4
    FORWARD_CW2 = 5
    FORWARD_CW3 = 6
    ABSORB = 7

# An integer between 1 and T//2 inclusive.
MAX_ANGLE_CHANGE = 3

class CarMDP(MDP):
    Actions = Actions

    def __init__(self, X, Y, T, goals, dt=0.1, vel=1.0, allow_wait=True, obstacle_list=None, **kwargs):
        """
        Params:
            X [int] -- The width of this MDP.
            Y [int] -- The height of this MDP.
            T [int] -- The number of different angles, uniformly spaced,
                allowed in this MDP. It's recommended for T to be a power of 2
                (avoid asymmetries) and to be at least 4.
            goals [list] -- A list of real-world goals (x,y,t) where t is theta. 
            dt [float] -- Timestep for discrete-time car dynamics. 
            vel [float] -- The arc length traveled, provided that the action
                involves changing position (x, y). `vel` is in units of
                gridsquares per timestep. It's recommended that `vel` is at
                least 1.
            allow_wait [bool] -- If this is True, then ABSORB is allowed on
                at every state. If this is False, then ABSORB is illegal except
                on goal states.
            obstacle_list [list] -- List of axis-aligned 2D boxes that represent
                obstacles in the envrionment. Specified in real coords:
                [((lower_x, lower_y), (upper_x, upper_y)), (...)]
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
        self.dt = dt
        self.vel = vel
        self.allow_wait = allow_wait
        S = X * Y * T

        # Compute maximum angular velocity based on the max angle change
        # and timestep.
        # self.max_angle_vel = MAX_ANGLE_CHANGE*(2*np.pi/self.T)/self.dt

        self.obstacle_list = obstacle_list
        self.goal_list = goals

        # Intentionally set default_reward to nan here. The idea is that we
        # shouldn't be looking at the self.reward method and instead rely
        # entirely on self.score for determining q_values.
        MDP.__init__(self, S=S, A=len(Actions),
                transition_helper=self._transition_helper,
                default_reward=np.nan, **kwargs)

        # Overwrite the default reward with the custom reward based on 
        # obstacles in environment. 
        for state in range(S):
            if self.is_blocked(state):
                self.rewards[state,:] = -np.inf

        # Compute values at each state for each goal.
        self.value_dict = {}
        for (x,y,t) in self.goal_list:
            goal_state = self.real_to_state(x,y,t)
            self.value_dict[goal_state] = hardmax._value_iter(self, goal_state, True)

        # Map from goal to Q-values.
        self.q_cache_dict = {}

        # Compute the Q-values for all the goals upon construction. 
        for (x,y,t) in self.goal_list:
            goal_coor = self.real_to_coor(x,y,t)
            self.q_cache_dict[goal_coor] = self.q_values(goal_coor, goal_stuck=True)

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
            x_new = x + self.dt * self.vel * np.cos(t)
            y_new = y + self.dt * self.vel * np.sin(t)
            t_new = t*self.dt
        else:
            if a == Actions.FORWARD_CCW1:
                ang_vel = (2*np.pi/self.T)/self.dt
            if a == Actions.FORWARD_CCW2:
                ang_vel = 2*(2*np.pi/self.T)/self.dt
            if a == Actions.FORWARD_CCW3:
                ang_vel = 3*(2*np.pi/self.T)/self.dt
            if a == Actions.FORWARD_CW1:
                ang_vel = -(2*np.pi/self.T)/self.dt
            if a == Actions.FORWARD_CW2:
                ang_vel = -2*(2*np.pi/self.T)/self.dt
            if a == Actions.FORWARD_CW3:
                ang_vel = -3*(2*np.pi/self.T)/self.dt

            x_new = x + self.dt * self.vel/ang_vel * (np.sin(t + ang_vel) - np.sin(t))
            y_new = y - self.dt * self.vel/ang_vel * (np.cos(t + ang_vel) - np.cos(t))
            t_new = (t + ang_vel * self.dt) % (2*np.pi)
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

    def is_blocked(self, s):
        """
        Returns True if s is blocked.

        By default, state-action pairs that lead to blocked states are illegal.
        Rewards are undefined for state-action pairs starting from blocked
        states.
        """
        if self.obstacle_list is None:
            return False

        # Check against internal representation of boxes. 
        sx, sy, _ = self.state_to_real(s)
        for box in self.obstacle_list:
            if sx >= box[0][0] and sx <= box[1][0] and \
               sy >= box[0][1] and sy <= box[1][1]:
                return True

        return False

    def q_values(self, goal_spec, goal_stuck=True):
        """
        Calculate the Q values for each state action pair using `self.score()`.

        Any action moving into a state that `is_blocked()` will be have value
        -infinity. Any illegal action will have value -infinity.

        Behavior is undefined if any goal state is blocked.

        Param:
        goal_spec [tuple(int, int, int)] -- A list of (x, y, t) coordinates of the goals. An
            AssertionError will fire if these coordinates are invalid.
            At goal states, the agent is allowed to
            choose a 0-cost ABSORB action.
        goal_stuck [bool] -- If True, then force ABSORB action at the goal by
            making all other actions worth -infinity.

        Return:
        Q [np.ndarray([S, A])] -- The Q values of each state action pair.
        """

        # Check if you have already computed the Q-values for this goal.
        if goal_spec not in self.q_cache_dict:
            self.q_cache_dict[goal_spec] = np.array([self.S, self.A])
            values = self.value_dict[goal_spec]

            for s in range(self.S):
                # Get the (x,y,t) tuple for this state.
                coor = self.state_to_coor(s)
                for a in range(self.A):
                    if coor == goal_spec:
                        reward = 0
                    else:
                        reward = self.rewards[s][a]
                    next_s, illegal = self._transition_helper(s, a, alert_illegal=True)
                    if illegal:
                        self.q_cache_dict[goal_spec][s,a] = -np.inf
                    else:
                        self.q_cache_dict[goal_spec][s,a] = reward + values[next_s]

                # If True, then force ABSORB action at the goal by
                # making all other actions worth -infinity.
                if goal_stuck and coor == goal_spec:
                    q_stay = self.q_cache_dict[goal_spec][s,actions.ABSORB]
                    self.q_cache_dict[goal_spec][s,:] = -np.inf
                    self.q_cache_dict[goal_spec][s,actions.ABSORB] = q_stay

        return self.q_cache_dict[goal_spec]

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
        x, y, t [int or float] -- The discrete x, y, t coordinates of the state.

        Returns:
        s [int] -- The state.
        """

        x, y, t = int(x), int(y), int(t)
        if not(0 <= x < self.X):
            raise ValueError(x, self.X)
        if not (0 <= y < self.Y):
            raise ValueError(y, self.Y)
        if not (0 <= t < self.T):
            raise ValueError(t, self.T)

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
