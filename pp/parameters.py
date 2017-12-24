import sys

# TODO: Figure this crazy import mess another day.
# I'm guessing this is because hardmax.occupancy
# which is part of inference.hardmax, is importing
# this module.
import mdp.euclid
if 'pp.mdp.euclid' in sys.modules:
    val_default = sys.modules['pp.mdp.euclid']
else:
    val_default = sys.modules['pedestrian_prediction.pp.mdp.euclid']

import inference.hardmax
if 'pp.inference.hardmax' in sys.modules:
    inf_default = sys.modules['pp.inference.hardmax']
else:
    inf_default = sys.modules['pedestrian_prediction.pp.inference.hardmax']
