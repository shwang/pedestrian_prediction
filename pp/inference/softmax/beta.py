from __future__ import division

import numpy as np

from ...mdp.softmax import backwards_value_iter, forwards_value_iter
from ...util import sum_rewards

from .destination import infer_destination
from .occupancy import infer_occupancies, infer_occupancies_from_start
from .. import grad_descent_shared as shared

def compute_score(g, traj, goal, beta, debug=False):
    assert len(traj) > 0, traj
    V = forwards_value_iter(g, goal, beta=beta)
    R_traj = sum_rewards(g, traj)
    start = traj[0][0]
    S_b = g.transition(*traj[-1])

    if debug:
        print V[S_b], V[start], V[S_b] - V[start]
    log_score = R_traj/beta + V[S_b] - V[start]
    return log_score


def compute_gradient(g, traj, goal, beta, debug=False):
    assert len(traj) > 0, traj
    start = traj[0][0]
    curr = g.transition(*traj[-1])
    R_traj = sum_rewards(g, traj)
    if curr == goal:
        ex_1 = 0
    else:
        ex_1 = np.sum(np.multiply(infer_occupancies(g, traj, beta=beta, dest_set={goal}),
                g.state_rewards))
    ex_2 = np.sum(np.multiply(
        infer_occupancies_from_start(g, start, beta=beta, dest_set={goal}),
        g.state_rewards))

    grad_log_score = (-1/beta**2) * (R_traj + ex_1 - ex_2)
    if debug:
        return grad_log_score, ex_1, ex_2
    else:
        return grad_log_score


def simple_search(g, traj, goal, *args, **kwargs):
    kwargs["compute_score"] = compute_score
    return shared.simple_search(g, traj, goal, *args, **kwargs)


def binary_search(g, traj, goal, *args, **kwargs):
    kwargs["compute_grad"] = compute_grad
    return shared.binary_search(g, traj, goal, *args, **kwargs)


def gradient_ascent(g, traj, goal, *args, **kwargs):
    kwargs["compute_score"] = compute_score
    kwargs["compute_grad"] = compute_grad
    return shared.gradient_ascent(g, traj, goal, *args, **kwargs)
