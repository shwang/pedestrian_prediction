import sys

def load(module, prefix='pedestrian_prediction.'):
    """
    Attempt to return the module and then trying with the prefix.
    Prefix is used for compatibility with crazyflie_human repo.

    KeyError if module is not found.
    """
    return sys.modules.get(module, None) or sys.modules[prefix+module]

import mdp.euclid
import mdp.hardmax
val_euclid = load('pp.mdp.euclid')
val_hardmax = load('pp.mdp.hardmax')
val_default = val_hardmax

import inference.hardmax
inf_hardmax = load('pp.inference.hardmax')
inf_default = load('pp.inference.hardmax')
