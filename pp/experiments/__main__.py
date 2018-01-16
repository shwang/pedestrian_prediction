from comparison import *
from context import HRContext
from robot import experiment_k, experiment_est_vs_fixed

# experiment_k("treacherous turn", N=10, collide_radius=2, k1=3, k2=2,
#         verbose=True,
#         fixed_beta=1.6,
#         beta_fallback=1.1)

# experiment_est_vs_fixed("criss-cross", N=16, collide_radius=7,
#         fixed_beta=1.6, beta_fallback=1.1, verbose=True)
# def check():
#     experiment_est_vs_fixed("criss-cross", N=16, collide_radius=7,
#             fixed_beta=1.6, beta_fallback=0.65, verbose=True)

# check()
# import cProfile
# cProfile.runctx("check()", globals=globals(), locals=locals())


ctx = HRContext(collide_radius=1, collide_penalty=10)\
        .cfg_mode("criss-cross", N=12)
beta_stars = [2, 1.5, 1, 0.5]
obvious_experiment(ctx, beta_stars=beta_stars, num_trials=1)
