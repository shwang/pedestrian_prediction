import numpy as np

from itertools import imap

from ..parameters import inf_default

def sum_rewards(mdp, traj):
    return sum(imap(lambda x: mdp.rewards[x[0], x[1]], traj))

def normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def display(mdp, traj, init_state, goal_state, overlay=True):
    init_state = mdp.state_to_coor(init_state)
    goal_state = mdp.state_to_coor(goal_state)

    visited = set(mdp.state_to_coor(s) for s, a in traj)
    if len(traj) > 0:
        visited.add(mdp.state_to_coor(mdp.transition(*traj[-1])))
    else:
        visited.add(init_state)

    lines = []
    for c in xrange(mdp.cols):
        line = ['_'] * mdp.rows
        for r in xrange(mdp.rows):
            if (r, c) in visited:
                line[r] = '#'
        if overlay:
            if c == init_state[1]:
                line[init_state[0]] = 'A' if init_state in visited else 'a'
            if c == goal_state[1]:
                line[goal_state[0]] = 'G' if goal_state in visited else 'g'
        lines.append(line)

    for l in reversed(lines):
        print l

def build_traj_from_actions(g, init_state, actions):
    s = init_state
    traj = []
    for a in actions:
        traj.append((s, a))
        s = g.transition(s, a)
    return traj

def traj_stats(g, start, goal, traj, beta=1, dest_set=None,
        T=0, c_0=-20, sigma_0=5, sigma_1=5, heat_maps=(), zmin=None, zmax=None,
        inf_mod=inf_default):

    print "Task: Start={}, Goal={}".format(g.state_to_coor(start), g.state_to_coor(goal))
    print "Assumed beta={}".format(beta)
    print "Possible goals:"
    if dest_set == None:
        dest_set = {goal}

    dest_set=set(d if type(d) is int else g.coor_to_state(*d) for d in dest_set)
    print dest_set

    print "Raw trajectory:"
    print [(g.state_to_coor(s), g.Actions(a)) for s, a in traj]
    display(g, traj, start, goal, overlay=True)

    P = infer_destination(g, traj, beta=beta, dest_set=dest_set)
    print "goal probabilities (softmax):"
    print P.reshape(g.rows, g.cols)

    occ = inf_mod.occupancy
    D = occ.infer(g, traj, beta=beta, dest_set=dest_set).reshape(g.rows, g.cols)
    print "expected occupancies:\n"
    print D.reshape(g.rows, g.cols)
