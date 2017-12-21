import inference.hardmax
import mdp.euclid
import mdp.hardmax

# import inference.hardmax as inf_hardmax  # FAILS!
# TODO: Figure this crazy import mess another day.
# I'm guessing this is because hardmax.occupancy
# which is part of inference.hardmax, is importing
# this module.
import sys
if 'pp.inference.hardmax' in sys.modules:
    inf_hardmax = sys.modules['pp.inference.hardmax']
else:
    inf_hardmax = sys.modules['pedestrian_prediction.pp.inference.hardmax']

inf_default = inf_hardmax  # default type of inference
val_default = mdp.euclid
