from comparison import *
from robot import experiment_k, experiment_est_vs_fixed
from analysis import inspect_probs

experiment_k("treacherous turn", N=10, collide_radius=2, k1=3, k2=2,
        verbose=True,
        fixed_beta=1.6,
        beta_fallback=1.1)

# experiment_est_vs_fixed("criss-cross", N=16, collide_radius=4,
#         fixed_beta=1.6, beta_fallback=1.1, verbose=True)



# obvious_experiment(beta_stars= [0.5, 1, 1.5, 2], collide_radius=3,
#         collide_penalty=10, num_trials=10, N=15, mode="criss-cross")
#
