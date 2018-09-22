from __future__ import division

def _make_harmonic(k, s=2, base=0.2):
    def harmonic(i):
        return max(k * 1/(s*i+1), base)
    return harmonic

def simple_search(g, traj, goal, compute_score,
        guess=None, delta=1e-2, beta_threshold=5e-6,
        verbose=False, min_beta=0.7, max_beta=11, min_iters=5, max_iters=20):

    if len(traj) == 0:
        return guess
    lo, hi = min_beta, max_beta
    mid = guess or (hi + lo)/2

    for i in xrange(max_iters):
        assert lo <= mid <= hi
        diff = min(hi - mid, mid - lo)
        mid_minus = mid - delta
        mid_plus = mid + delta

        s_minus = compute_score(g, traj, goal, mid_minus)
        s_plus = compute_score(g, traj, goal, mid_plus)

        if verbose:
            s_mid = compute_score(g, traj, goal, mid)
            print "i={}\t mid={}\tscore={}\tgrad={}".format(
                    i, mid, s_mid, (s_plus-s_minus)*2/delta)

        if i >= min_iters and hi - lo < beta_threshold:
            break

        if s_plus - s_minus > 0:
            lo = mid
        else:
            hi = mid
        if i >= min_iters and hi - lo < beta_threshold:
            break

        mid = (lo + hi)/2

    if verbose:
        print "final answer: beta=", mid
    return mid

def binary_search(g, traj, goal, compute_grad,
        guess=None, grad_threshold=1e-9, beta_threshold=5e-5,
        min_iters=3, max_iters=30, min_beta=0.2, max_beta=1e2, verbose=False):

    lo, hi = min_beta, max_beta
    if guess is None:
        mid = (lo + hi) / 2
    else:
        mid = guess

    if len(traj) == 0:
        return guess

    for i in xrange(max_iters):
        assert lo <= mid <= hi
        grad = compute_grad(g, traj, goal, mid)
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

def gradient_ascent(g, traj, goal, compute_score, compute_grad,
    guess=3, learning_rate=_make_harmonic(5),
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
        grad = compute_grad(g, traj, goal, curr)
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
            history.append((curr, compute_score(g, traj, goal, curr)))
            print u"{}: beta={}\tscore={}\tgrad={}\tlearning_rate={}\tdiff={}".format(
                i, curr, history[-1][1], grad, alpha(i), diff)

        if i >= min_iters and abs(diff) < threshold:
            break

    return curr
