from __future__ import division

import numpy as np

from .. import grad_descent_shared as shared
from ...parameters import val_default

def _make_compute_grad(k=None, decay_rate=None):
    def inner(*args, **kwargs):
        if k is not None:
            kwargs['k'] = k
        if decay_rate is not None:
            kwargs["decay_rate"] = decay_rate
        return compute_grad(*args, **kwargs)
    return inner


def _make_compute_score(k=None, decay_rate=None):
    def inner(*args, **kwargs):
        if k is not None:
            kwargs['k'] = k
        if decay_rate is not None:
            kwargs["decay_rate"] = decay_rate
        return compute_score(*args, **kwargs)
    return inner


# TODO: implement decay_rate (soft forget)
def compute_score(g, traj, goal, beta, cached_P=None, debug=False,
        k=np.inf, decay_rate=0,
        val_mod=val_default):
    assert len(traj) > 0, traj
    if len(traj) > k:
        traj = traj[-k:]
    if cached_P is None:
        P = g.action_probabilities(goal, beta=beta)
    else:
        P = cached_P

    # XXX: Keeping here for now, since it is useful for debugging, and I was
    #   recently confused by some beta MLE results.
    # A = g.Actions
    # s, a = traj[0]
    # V = val_mod.forwards_value_iter(g, goal)
    # Q = g.q_values(goal)
    # import pdb; pdb.set_trace()

    score = np.zeros(len(traj))
    for i, (s, a) in enumerate(traj):
        score[i] = P[s, a]

    log_score = np.log(score, out=score)
    return np.sum(log_score)

# TODO: implement decay_rate (soft forget). Maybe also add a `soft` flag.
                    # That way I can have a decay_rate default != 0.
def compute_grad(g, traj, goal, beta, k=np.inf, decay_rate=0, debug=False,
        val_mod=val_default):
    assert len(traj) > 0, traj
    if len(traj) > k:
        traj = traj[-k:]
    Q = g.q_values(goal)
    # Prevent -inf * 0 in the multiply(P,Q) operation.
    # REQUIRES NUMPY VERSION 1.13
    np.nan_to_num(Q, copy=False)
    P = g.action_probabilities(goal, beta=beta)
    assert Q.shape == P.shape

    q_sum = 0
    ex_q = np.multiply(P, Q)
    ex_q_sum = 0

    for s, a in traj:
        q_sum += Q[s, a]
        ex_q_sum += np.sum(ex_q[s])

    return -(1/beta/beta) * (q_sum - ex_q_sum)

def simple_search(g, traj, goal, k=None, decay_rate=None, *args, **kwargs):
    kwargs["compute_score"] = _make_compute_score(k=k, decay_rate=decay_rate)
    return shared.simple_search(g, traj, goal, *args, **kwargs)

def binary_search(g, traj, goal, k=None, decay_rate=None, *args, **kwargs):
    kwargs["compute_grad"] = _make_compute_grad(k=k, decay_rate=decay_rate)
    return shared.binary_search(g, traj, goal, *args, **kwargs)

def gradient_ascent(g, traj, goal, *args, **kwargs):
    kwargs["compute_score"] = compute_score
    kwargs["compute_grad"] = compute_grad
    return shared.gradient_ascent(g, traj, goal, *args, **kwargs)

def calc_posterior_over_set(g, traj, goal, betas, priors=None, k=None):
    assert betas is not None
    if priors is None:
        priors = np.ones(len(betas))
        priors /= len(betas)
    assert len(priors) == len(betas)

    if k is not None:
        assert k > 0
        traj = traj[-k:]

    g.set_goal(goal)
    P_beta = np.copy(priors)
    for i, beta in enumerate(betas):
        P_beta[i] *= g.trajectory_probability(goal, traj=traj,
                beta=beta)
    np.divide(P_beta, np.sum(P_beta), out=P_beta)

    return P_beta
