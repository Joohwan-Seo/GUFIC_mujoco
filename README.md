# GUFIC_mujoco

Mujoco Implementation of Geometric Unified Force Impedance Control

Author: Joohwan Seo (Ph.D. Candidate UC Berkeley, Mechanical Engineering)

Implementation of the paper titled as:

"Geometric Formulation of Unified Force-Impedance Control on SE(3) For Robotic Manipulator"

## Tested with
```
python == 3.10.16, scipy == 1.15.2, mujoco == 3.3.0
```

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

**NOTE**
For the ``data_exporter_tikz.py``, use ``export_tikz = False`` as the default tikz exporter is not working. Tikz exporter is not compatible with the current matplotlib version, so it needs to be updated. Go to the source ``tikzplotlib`` github and search for the issues. You may need to modify the source code, or download the modified branch and install from the source. 


### Citation:
```source
@article{seo2025geometric,
  title={Geometric Formulation of Unified Force-Impedance Control on SE (3) for Robotic Manipulators},
  author={Seo, Joohwan and Prakash, Nikhil Potu Surya and Lee, Soomi and Kruthiventy, Arvind and Teng, Megan and Choi, Jongeun and Horowitz, Roberto},
  journal={arXiv preprint arXiv:2504.17080},
  year={2025}
}
```
