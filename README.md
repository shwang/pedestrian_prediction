# pedestrian_prediction
Realtime, confidence-varying trajectory prediction for "Probabilistically Safe Robot Planning with Confidence-Based Human Predictions". RSS '18

## Installation
```
git clone https://github.com/sirspinach/pedestrian_prediction.git
cd pedestrian_prediction
pip install -e .
```

## Module contents
  * `pp.mdp`: Data structures and value iteration algorithms for two types of GridWorlds -- a standard GridWorld with only lateral and diagonal movement, and a GridWorldExpanded in which actions operate in "gridless" continuous space but are snapped to the nearest grid cell.
  * `pp.inference`: Algorithms for inferring destinations, occupancies, and states given a GridWorld, a list of destinations, and the pedestrian's trajectory so far.
  * `pp.util`: Utilities for parsing and trajectory manipulation.
  * `pp.plots`: Plot simulated trajectories, overlayed with heatmaps of predicted states. Produce some test plots by executing `python -m pp.plot`. (Requires plotly)
