import common
import common_multi
import common_forget

if __name__ == '__main__':
    common_forget.traj_inf("diag-fickle2", R=-1,
            N=20,
            traj_lens=[6,8,None], save_png=True)
    # common_multi.multidest_traj_inf(
    #         "diag", "tri", N=40, R=-1, save_png=True)
