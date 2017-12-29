from __future__ import division
import numpy as np

from heapq import heappush, heappop
from ..parameters import inf_default, val_default

def robot_planner(g_R, start_R, g_H, traj=[], start_H=None, inf_mod=inf_default,
        collide_penalty=50, collide_radius=1, max_depth=100,
        val_mod=val_default,
        beta_guess=0.5, calc_beta=True,
        verbose_return=False,
        mk_q_values=None, mk_collide_probs=None, mk_binary_search=None):
    """
    if not verbose_return:
        return plan
    else:
        return plan, expected_cost, final_node, beta
    """
    # TODO: fix duplicated default arguments `collide_*`
    bin_search = mk_binary_search or inf_mod.beta.binary_search
    _CollideProbs = mk_collide_probs or CollideProbs
    q_values = mk_q_values or val_default.q_values

    if calc_beta:
        beta = bin_search(g_H, traj, g_H.goal, guess=beta_guess, verbose=False,
                min_iters=30, max_iters=30, max_beta=100)
    else:
        beta = beta_guess
    collide_probs = _CollideProbs(g_H, traj=traj, start_H=start_H,
            beta=beta, T=max_depth, collide_radius=collide_radius)
    q_values_R = q_values(g_R, g_R.goal)

    heap = []
    heappush(heap, (0, AStarNode(g_R, start_R)))
    visited = set() # Placeholders for return values
    plan = None
    res_expected_cost = np.inf
    final_node = None

    while len(heap) > 0:
        ex_cost, node = heappop(heap)
        if node.t > 1 and node.traj[-1] == (g_R.goal, g_R.Actions.ABSORB):
            plan = node.traj
            expected_cost = ex_cost
            final_node = node
            break
        elif node.t + 1 >= max_depth or node.key() in visited:
            continue
        visited.add(node.key())

        # Expand node.
        for a, s_prime in g_R.neighbors[node.s]:
            reward = g_R.rewards[node.s, a]
            if reward == -np.inf:
                continue
            backward_cost = node.backward_cost - reward
            p = collide_probs.get(node.t + 1, s_prime)
            backward_cost += p * collide_penalty
            forward_cost = -q_values_R[node.s, a]

            child = node.make_child(a, backward_cost)
            cost = backward_cost + forward_cost
            heappush(heap, (backward_cost + forward_cost, child))

    if not verbose_return:
        return plan
    else:
        return plan, expected_cost, final_node, beta

class AStarNode(object):
    def __init__(self, g, state):
        self.g = g
        self.s = state
        self.backward_cost = 0
        self.traj = tuple()
        self.t = 0

    def make_child(self, action, backward_cost):
        """
        Generate a child node to add to the heap.
        Params:
            action -- The next action in this trajectory.
            backward_cost -- The total backward cost after `action` is taken.
        """
        assert backward_cost >= self.backward_cost
        s_prime = self.g.transition(self.s, action)
        tr = self.traj + ((self.s, action),)
        child = AStarNode(self.g, s_prime)
        child.backward_cost = backward_cost
        child.traj = tr
        child.t = self.t + 1
        return child

    def key(self):
        """ Generate a hashable tuple that uniquely identifies this node.
            Useful for pruning redundant, symmetric paths. """
        return (self.t, self.s, self.backward_cost)

class CollideProbs(object):
    def __init__(self, g_H, T, collide_radius=2, traj=[], start_H=None, beta=1,
            start_R=0, state_probs_cached=None, inf_mod=inf_default):
        """
        Helper class for lazily calculating the collide probability at
        various states and timesteps.
        """
        assert len(traj) > 0 or start_H is not None, \
                "Must set at least one of `traj` or `start_H`"
        self.g = g = g_H
        self.beta = beta
        self.T = T
        self.collide_radius = collide_radius

        if state_probs_cached is not None:
            self.state_probs = state_probs_cached.reshape(T+1, g.rows, g.cols)
        else:
            if len(traj) == 0:
                self.state_probs = inf_mod.state.infer_from_start(g, start_H,
                        g.goal, T=T, beta_or_betas=beta)[0].reshape(
                                T+1, g.rows, g.cols)
            else:
                self.state_probs = inf_mod.state.infer(g, traj, g.goal, T=T,
                        beta_or_betas=beta)[0].reshape(T+1, g.rows, g.cols)
        self.cache = {}

    def get(self, t, s):
        if (t, s) in self.cache:
            return self.cache[(t, s)]
        x, y = self.g.state_to_coor(s)
        r = self.collide_radius
        colliding = self.state_probs[t, max(x-r,0):x+r+1, max(y-r,0):y+r+1]
        result = np.sum(colliding)
        self.cache[(t, s)] = result
        return result
