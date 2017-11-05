from __future__ import division
from __future__ import absolute_import
from value_iter import backwards_value_iter, forwards_value_iter
from inference import sample_action, simulate, _display, infer_destination, \
    infer_occupancies, infer_occupancies_from_start, infer_temporal_occupancies, \
    _sum_rewards

import numpy as np

def _compute_score(g, traj, goal, beta):
    assert len(traj) > 0, traj
    V = forwards_value_iter(g, goal, beta=beta)
    R_traj = _sum_rewards(g, traj)
    start = traj[0][0]
    S_b = g.transition(*traj[-1])

    log_score = R_traj/beta + V[S_b] - V[start]
    return log_score

def _compute_gradient(g, traj, goal, beta):
    assert len(traj) > 0, traj
    start = traj[0][0]
    curr = g.transition(*traj[-1])
    R_traj = _sum_rewards(g, traj)
    if curr == goal:
        ex_1 = 0
    else:
        ex_1 = np.sum(np.multiply(infer_occupancies(g, traj, beta=beta, dest_set=set([goal])),
                g.state_rewards))
    ex_2 = np.sum(np.multiply(
        infer_occupancies_from_start(g, start, beta=beta, dest_set=set([goal])),
        g.state_rewards))

    grad_log_score = (-1/beta**2) * (R_traj + ex_1 - ex_2)
    return grad_log_score

def _make_harmonic(k, s=2, base=0.2):
    def harmonic(i):
        return max(k * 1/(s*i+1), base)
    return harmonic

def beta_binary_search(g, traj, goal, guess=None, grad_threshold=1e-9, beta_threshold=5e-5,
        min_iters=10, max_iters=30, min_beta=0.7, max_beta=11, verbose=False):

    if len(traj) == 0:
        return guess

    lo, hi = min_beta, max_beta
    mid = guess
    for i in xrange(max_iters):
        assert lo <= mid <= hi
        grad = _compute_gradient(g, traj, goal, mid)
        if verbose:
            print u"i={}\t mid={}\t grad={}".format(i, mid, grad)

        if i >= min_iters and abs(grad) < grad_threshold:
            break

        if grad > 0:
            lo = mid
        else:
            hi = mid
        if i >= min_iters and hi - lo < beta_threshold:
            break

        mid = (lo + hi)/2

    if verbose:
        print u"final answer: beta=", mid
    return mid

def beta_gradient_ascent(g, traj, goal, guess=3, learning_rate=_make_harmonic(5),
    verbose=False, threshold=1e-9, min_iters=10, max_iters=30, max_update=4,
    min_beta=0.1, max_beta=11):

    if len(traj) == 0:
        return guess

    if type(learning_rate) in [float, int]:
        alpha = lambda i: learning_rate
    else:
        alpha = learning_rate

    history = []
    curr = guess
    for i in xrange(max_iters):
        grad = _compute_gradient(g, traj, goal, curr)
        diff = alpha(i) * grad

        if diff > max_update:
            diff = max_update
        elif diff < -max_update:
            diff = -max_update

        assert diff not in [np.inf, -np.inf, np.nan], curr
        if diff > 1e-5:
            curr += diff
        else:
            curr -= diff * 130

        if verbose:
            history.append((curr, _compute_score(g, traj, goal, curr)))
            print u"{}: beta={}\tscore={}\tgrad={}\tlearning_rate={}\tdiff={}".format(
                i, curr, history[-1][1], grad, alpha(i), diff)

        if i >= min_iters and abs(diff) < threshold:
            break

    return curr
