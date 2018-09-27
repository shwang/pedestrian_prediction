from __future__ import division
import numpy as np

from heapq import heappush, heappop
from ..parameters import inf_default, val_default
from ..util.util import display_plan

def forget(traj, k):
    if k is None:
        return traj
    assert k > 0
    return traj[-k:]

def robot_planner_bayes(ctx, state_R, betas, priors, traj_H=[], max_depth=None,
        inf_mod=inf_default, k=None, **kwargs):
    """
    Calculate an A* plan assuming beta_star is in betas, and then finding the
    posterior distribution over beta_star given `traj_H`.

    Returns: plan, expected_cost, final_node, P_betas
    """
    g_H = ctx.g_H
    collide_probs = CollideProbsBayes(ctx=ctx, betas=betas, priors=priors,
            traj=forget(traj_H, k), T=max_depth)
    P_beta = inf_mod.beta.calc_posterior_over_set(g=g_H, traj=traj_H, k=k,
            goal=g_H.goal, betas=betas, priors=priors)
    plan, ex_cost, final_node =  _robot_planner(ctx, state_R=state_R,
            traj_H=traj_H, collide_probs=collide_probs, verbose_return=True,
            k=k, **kwargs)
    return plan, ex_cost, final_node, P_beta


def robot_planner_fixed(ctx, state_R, traj_H=[], beta_guess=0.5,
        max_depth=None, inf_mod=inf_default, **kwargs):
    """
    Calculate an A* plan assuming beta=beta_guess.

    Returns: plan, expected_cost, final_node, beta
    """
    g_H = ctx.g_H
    collide_probs = CollideProbs(ctx=ctx, T=max_depth, traj=traj_H,
            beta=beta_guess)

    plan, ex_cost, final_node =  _robot_planner(ctx, state_R=state_R,
            collide_probs=collide_probs, traj_H=traj_H, inf_mod=inf_mod,
            verbose_return=True,
            **kwargs)
    return plan, ex_cost, final_node, beta_guess


def robot_planner_vanilla(ctx, state_R, traj_H=[], beta_guess=0.5,
        k=None,
        max_depth=None, mk_binary_search=None, inf_mod=inf_default, **kwargs):
    """
    Calculate an A* plan assuming MLE beta based on `traj_H`.

    Returns: plan, expected_cost, final_node, beta
    """
    bin_search = mk_binary_search or inf_mod.beta.binary_search
    g_H = ctx.g_H

    beta = bin_search(g_H, traj_H, g_H.goal, guess=beta_guess,
            verbose=False, k=k, min_iters=15, max_iters=15)

    return robot_planner_fixed(ctx=ctx, state_R=state_R, traj_H=traj_H,
            beta_guess=beta, max_depth=max_depth, k=k,
            inf_mod=inf_mod, **kwargs)


def _robot_planner(ctx, state_R, collide_probs, traj_H=[],
        max_heap_size=200000, k=None, beta_guess=0.5, verbosity=0,
        verbose_return=False, inf_mod=inf_default):
    """
    Verbosity indicates the number of top trajectories to visualize.
    Upon success, always visualize one trajectory.
    Upon failure, visualize up to `verbosity` trajectories.

    if not verbose_return:
        return plan
    else:
        return plan, expected_cost, final_node
    """
    verbose = verbosity > 0
    max_depth = collide_probs.T
    g_R = ctx.g_R
    q_values_R = g_R.q_values(g_R.goal)

    heap = []
    heappush(heap, (0, node_init(state_R, verbose)))
    visited = set() # Placeholders for return values
    plan = None
    expected_cost = np.inf
    final_node = None
    tuple_ref = g_R.transition_cached_t
    A = g_R.A

    while 0 < len(heap) < max_heap_size:
        ex_cost, node = heappop(heap)

        if verbose:
            ex_cost, node = heappop(heap)
            plan = node[NODE_TRAJ]
            heat_nums = np.round(node[NODE_BONUS].collision_probs * 10)\
                    .astype(int)
            display_plan(g_R, plan, state_R, g_R.goal, heat_nums)
            print ""

        t = len(node[NODE_TRAJ])
        if t >= 1 and node[NODE_TRAJ][-1] == (g_R.goal, g_R.Actions.ABSORB):
            plan = node[NODE_TRAJ]
            expected_cost = ex_cost
            final_node = node
            break
        elif node_key(node) in visited or t >= max_depth:
            continue
        visited.add(node_key(node))

        # Expand node.
        for a, s_prime in g_R.neighbors[node[NODE_S]]:
            reward = g_R.rewards[node[NODE_S], a]
            if reward == -np.inf:
                continue
            backward_cost = node[NODE_BACKWARD_COST] - reward
            p = collide_probs.get(t + 1, s_prime)
            backward_cost += p * ctx.collide_penalty
            forward_cost = -q_values_R[node[NODE_S], a]

            child = node_child(node, g_R, a, backward_cost, tuple_ref, A,
                    verbose)
            if verbosity > 0:
                child[NODE_BONUS].collision_probs.append(p)
                assert len(child[NODE_BONUS].collision_probs) == \
                        len(child[NODE_TRAJ])
            cost = backward_cost + forward_cost
            heappush(heap, (backward_cost + forward_cost, child))

    # Indicates failure
    if final_node == None:
        if verbose:
            print "FAILURE: printing top few nodes"
        for _ in range(verbosity):
            ex_cost, node = heappop(heap)
            plan = node[NODE_TRAJ]
            heat_nums = np.round(node[NODE_BONUS].collision_probs * 10)\
                    .astype(int)
            display_plan(g_R, plan, state_R, g_R.goal, heat_nums)
            print ""

    if not verbose_return:
        return plan
    else:
        return plan, expected_cost, node[NODE_TRAJ]

NODE_S = 0
NODE_BACKWARD_COST = 1
NODE_TRAJ = 2
NODE_T = 3
NODE_BONUS = 4

class NodeBonus(object):
    """
    A class for storing debug information
    """
    def __init__(self):
        self.collision_probs = []

    def copy(self):
        new = NodeBonus()
        new.collision_probs = list(self.collision_probs)
        return new

def node_init(state, verbose):
    if verbose:
        return (state, 0, tuple(), 0, NodeBonus())
    else:
        return (state, 0, tuple(), 0, None)
    # return AStarNode(s=state, backward_cost=0, traj=tuple(), t=0)

def node_key(node):
    return (node[NODE_T], node[NODE_S])

def node_child(node, g, action, backward_cost, tuple_ref, A, verbose):
    assert backward_cost >= backward_cost
    ### EQUIVALENT, but much faster than g.transition()
    s_prime = tuple_ref[A * node[NODE_S] + action]
    # s_prime = g.transition(node[NODE_S], action)
    ###

    tr = node[NODE_TRAJ] + ((node[NODE_S], action),)
    t = node[NODE_T] + 1
    if verbose:
        return (s_prime, backward_cost, tr, t, node[NODE_BONUS].copy())
    else:
        return (s_prime, backward_cost, tr, t, None)


class CollideProbs(object):
    def __init__(self, ctx, T=None, traj=[],
            beta=1, start_R=0, state_probs_cached=None, inf_mod=inf_default):
        """
        Helper class for lazily calculating the collide probability at
        various states and timesteps.
        """
        assert len(traj) > 0 or ctx.start_H is not None, \
                "Must set at least one of `traj` or `ctx.start_H`"
        self.ctx = ctx
        self.g = g = ctx.g_H
        self.start_H = ctx.start_H
        self.traj = traj  # Allowed to differ from ctx.traj_H b/c so we can give
                            # simulated Robot a partial trajectory at each given
                            # timestep
        if T is not None:
            self.T = T
        elif ctx.N is not None:
            self.T = ctx.N * 2
        else:
            self.T = ctx.g_R.rows * 2
        self.beta = beta
        self.collide_radius = self.ctx.collide_radius
        self.inf_mod = inf_mod

        if state_probs_cached is not None:
            self.state_probs = state_probs_cached.reshape(T+1, g.rows, g.cols)
        else:
            self.state_probs = self.calc_state_probs()
        self.cache = {}

    def calc_state_probs(self):
        g = self.g
        if len(self.traj) == 0:
            state_probs = self.inf_mod.state.infer_from_start(g, self.start_H,
                    g.goal, T=self.T, beta_or_betas=self.beta)[0].reshape(
                            self.T+1, g.rows, g.cols)
        else:
            state_probs = self.inf_mod.state.infer(g, self.traj,
                    g.goal, T=self.T, beta_or_betas=self.beta)[0].reshape(
                            self.T+1, g.rows, g.cols)
        return state_probs

    def get(self, t, s):
        if (t, s) in self.cache:
            return self.cache[(t, s)]
        x, y = self.g.state_to_coor(s)
        r = self.collide_radius
        colliding = self.state_probs[t, max(x-r,0):x+r+1, max(y-r,0):y+r+1]
        result = np.sum(colliding)
        self.cache[(t, s)] = result
        return result

class CollideProbsBayes(CollideProbs):
    def __init__(self, ctx, betas, priors=None, *args, **kwargs):
        """
        Helper class for lazily calculating the collide probability at
        various states and timesteps.
        """
        self.betas = betas
        self.priors = priors
        CollideProbs.__init__(self, ctx, beta="N/A (Bayes: see self.betas)",
                *args, **kwargs)


    def calc_state_probs(self):
        g = self.g
        state_probs = self.inf_mod.state.infer_bayes(g=g, traj=self.traj,
                init_state=self.start_H, dest=g.goal, T=self.T,
                betas=self.betas, priors=self.priors).reshape(
                        self.T+1, g.rows, g.cols)
        return state_probs
