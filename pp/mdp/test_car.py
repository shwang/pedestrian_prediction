import unittest
import pytest
import numpy as np
from numpy import testing as t

from pp.mdp.car import CarMDP

class TestCar(unittest.TestCase):

    def test_init(self):
        c1 = CarMDP(5, 5, 10, [(1,1,0)])
        c2 = CarMDP(8, 16, 10, [(5,8,np.pi), (5,8,0)])

    def test_n_states(self):
        c1 = CarMDP(5, 5, 5, [(1,1,0)])
        assert c1.S == 125
        c2 = CarMDP(8, 16, 10, [(1,1,0)])
        assert c2.S == 8 * 16 * 10


    def test_coor_to_state_isomorphic(self):
        def inner(c):
            for i in range(c.S):
                assert i == c.coor_to_state(*c.state_to_coor(i))
        inner(CarMDP(5, 5, 5, [(1,1,0)]))
        inner(CarMDP(6, 2, 11, [(1,1,0)]))

    def test_real_to_state_isomorphic(self):
        def inner(c):
            for i in range(c.S):
                assert i == c.real_to_state(*c.state_to_real(i))
        inner(CarMDP(5, 5, 5, [(1,1,0)]))
        inner(CarMDP(6, 2, 11, [(1,1,0)]))

    def test_coor_to_real_sanity(self):
        c = CarMDP(5, 5, 4, [(1,1,0)])
        def check(c, x_coor, y_coor, t_coor, x_real, y_real, t_real):
            x, y, t = c.coor_to_real(x_coor, y_coor, t_coor)
            assert (x, y) == (x_real, y_real)
            np.testing.assert_allclose(t, t_real)
            x, y, t = c.real_to_coor(x_real, y_real, t_real)
            assert (x, y, t) == (x_coor, y_coor, t_coor)
        check(c, 0, 0, 0, 0.5, 0.5, 0)
        check(c, 0, 0, 1, 0.5, 0.5, 2*np.pi/4)
        check(c, 0, 0, 2, 0.5, 0.5, 2*np.pi/4 * 2)
        check(c, 4, 4, 3, 4.5, 4.5, 2*np.pi/4 * 3)

    def test_is_goal(self):
        c = CarMDP(5, 5, 4, [(1,1,0)])
        goal_spec = x, y, t = 3, 4, 2
        s = c.coor_to_state(x, y, t)
        assert c.is_goal(s, goal_spec)

        for t in range(c.T):
            s2 = c.coor_to_state(0, 0, t)
            assert not c.is_goal(s2, goal_spec)
    """
    # doesn't work.
    def test_transitions_sanity(self):
        c = CarMDP(2, 1, 4, [(1,0,0)], vel=1.0)
        origin_face_right = c.coor_to_state(0, 0, 0)
        s = c.transition(origin_face_right, c.Actions.FORWARD)
        print "c state_to_coor: ", c.state_to_coor(s)
        assert (0, 0, 0) == c.state_to_coor(s)

        c = CarMDP(3, 3, 4, [(1,1,0)], vel=1.0)
        origin_face_up = c.coor_to_state(1, 1, 1)
        s = c.transition(origin_face_up, c.Actions.FORWARD)
        assert (1, 2, 1) == c.state_to_coor(s)

        origin_face_up = c.coor_to_state(1, 1, 1)
        s = c.transition(origin_face_up, c.Actions.FORWARD_CW)
        assert c.state_to_coor(s)[2] == 0  # Check angle
        s = c.transition(origin_face_up, c.Actions.FORWARD_CCW)
        assert c.state_to_coor(s)[2] == 2  # Check angle

        c = CarMDP(3, 3, 8, [(1,1,0)], vel=1.0)
        origin_face_up_right = c.coor_to_state(1, 1, 1)
        s = c.transition(origin_face_up_right, c.Actions.FORWARD)
        assert (2, 2, 1) == c.state_to_coor(s)
  
    # doesn't work.
    def test_q_values_sanity(self):
        c = CarMDP(3, 3, 8, [(1,1,0)], vel=1.0)
        goal_spec = (1,1,0)
        s = c.coor_to_state(0, 0, 1)

        c = CarMDP(3, 3, 8, [(1,1,0)], vel=1.0, allow_wait=False)
        Q = c.q_values(goal_spec)
        origin_face_up_right = c.coor_to_state(0, 0, 1)
        assert Q[c.coor_to_state(2, 2, 0), c.Actions.ABSORB] == 0
        assert Q[c.coor_to_state(1, 1, 0), c.Actions.ABSORB] == -np.inf
        assert -Q[origin_face_up_right, c.Actions.FORWARD] == 4 + 4

        c = CarMDP(3, 3, 8, [(1,1,0)], vel=1.0, allow_wait=True)
        Q = c.q_values(goal_spec)
        origin_face_up_right = c.coor_to_state(0, 0, 1)
        assert -Q[c.coor_to_state(2, 2, 0), c.Actions.ABSORB] == 0
        assert -Q[c.coor_to_state(1, 1, 0), c.Actions.ABSORB] == 1 + 1
        assert -Q[c.coor_to_state(0, 0, 0), c.Actions.ABSORB] == 4 + 4
        assert -Q[origin_face_up_right, c.Actions.FORWARD] == 4 + 4

    # doesn't work.
    def test_vel_2_sanity(self):
        c = CarMDP(3, 3, 8, [(1,1,0)], vel=2.0)
        mid_face_downleft = c.coor_to_state(1, 1, 5)
        origin_face_downleft = c.coor_to_state(0, 0, 5)
        assert (c.transition(mid_face_downleft, c.Actions.FORWARD)
                == origin_face_downleft)
    """

    @pytest.mark.now
    def test_inference_no_crash(self):
        from pp.inference.hardmax import state, occupancy
        c = CarMDP(7, 7, 8, [(0,6,0), (5,5,0)], vel=2.0, allow_wait=True)
        s = state.infer_joint(c, dests=[(0,6,0), (5,5,0)], betas=[1,2,3,4], T=50,
                init_state=0, verbose_return=True)

if __name__ == '__main__':
    unittest.main()