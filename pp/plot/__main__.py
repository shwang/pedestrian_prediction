import common
import common_multi

if __name__ == '__main__':
    common.simple_traj_inf("diag", R=-1, traj_len=7, save_png=False)
    # common_multi.multidest_traj_inf(
    #         "diag", "tri", N=40, R=-1, save_png=True)
