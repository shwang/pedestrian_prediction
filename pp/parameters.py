"""
Update val_default and inf_default to choose the default type of value
iteration and occupancy inference.

Here we access the sys.modules dictionary to circumvent some problems with
cyclic imports.
"""

import sys

import mdp.euclid
import mdp.hardmax

if 'pp.mdp.euclid' in sys.modules:
    val_euclid = sys.modules['pp.mdp.euclid']
else:
    val_euclid = sys.modules['pedestrian_prediction.pp.mdp.euclid']

if 'pp.mdp.hardmax' in sys.modules:
    val_hardmax = sys.modules['pp.mdp.hardmax']
else:
    val_hardmax = sys.modules['pedestrian_prediction.pp.mdp.hardmax']

val_default = val_hardmax

import inference.hardmax
if 'pp.inference.hardmax' in sys.modules:
    inf_default = sys.modules['pp.inference.hardmax']
else:
    inf_default = sys.modules['pedestrian_prediction.pp.inference.hardmax']