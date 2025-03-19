# GUFIC_mujoco

Mujoco Implementation of Geometric Unified Force Impedance Control

Author: Joohwan Seo (Ph.D. Candidate UC Berkeley, Mechanical Engineering)

Submitted to CDC 2025 as

"Geometric Formulation of Unified Force-Impedance Control on SE(3) For Robotic Manipulator"

## Tested with
```
python == 3.10.16, scipy == 1.15.2, mujoco == 3.3.0
```

Tex Exporter does not work that well, go to tikzplotlib github and search for the issues.
You need to modify the source code, or download the forked version and install from source. 

## Usage
### Directly running the environment files:
GUFIC
```source
python gufic_env/env_gufic_velocity_field.py
```
GIC
```source
python gufic_env/env_gic_trajectory_tracking.py
```

### Using wrap-up codes:
```source
python scripts/simulation_runner.py
```
For the visualization:
```source
python scripts/data_exporter_tikz.py
```
