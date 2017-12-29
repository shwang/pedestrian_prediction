from robot import experiment_plot
from analysis import inspect_probs

experiment_plot("hug walls 3", N=16, collide_radius=2, fixed_beta=0.2)
# experiment_plot("diag shuffle", N=14, collide_radius=1)
# experiment_plot("swap corners false", N=12, collide_radius=1)
# inspect_probs()
