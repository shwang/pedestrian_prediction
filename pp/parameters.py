"""
Update val_default and inf_default to choose the default type of value
iteration and occupancy inference.

Here we access the sys.modules dictionary to circumvent some problems with
cyclic imports.
"""

import sys

import mdp.euclid
import mdp.hardmax
val_euclid = sys.modules['pp.mdp.euclid']
val_hardmax = sys.modules['pp.mdp.hardmax']
val_default = val_hardmax

import inference.hardmax
inf_hardmax = sys.modules['pp.inference.hardmax']
inf_default = inf_hardmax
