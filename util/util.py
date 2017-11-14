import numpy as np

from itertools import imap

def sum_rewards(mdp, traj):
    return sum(imap(lambda x: mdp.rewards[x[0], x[1]], traj))

def normalize(vec):
    x = np.array(vec)
    return x/sum(x)

def display(mdp, traj, init_state, goal_state, overlay=False):
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
