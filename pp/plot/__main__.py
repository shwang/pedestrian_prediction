import common
import common_multi
import common_forget

if __name__ == '__main__':
    common_multi.multidest_traj_inf("diag-fickle", "diag+bot", N=20,
            epsilon=0.15,
            title=("Baysian filter on D_t: t={t}, epsilon={epsilon}, traj={traj}"),
            hmm=True, save_png=True)
