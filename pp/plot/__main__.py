import common
import common_multi
import common_forget
import study_traj

# study_traj.histogram_beta_est(R=-1, true_beta=0.8, samples=1000, N=15, path_length=3)
# study_traj.histogram_beta_est(R=-1, true_beta=0.8, samples=1000, N=15, path_length=5)
# study_traj.histogram_beta_est(R=-1, true_beta=0.8, samples=1000, N=15, path_length=10)
# study_traj.histogram_beta_est(R=-1, true_beta=0.8, samples=1000, N=15, path_length=15)

common_multi.multidest_traj_inf(traj_mode="andrea", mode="andrea", N=24)
