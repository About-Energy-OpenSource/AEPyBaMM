# AEPyBaMM

**AEPyBaMM** (`aepybamm`) is a Python library that supports the use of [About:Energy](https://www.aboutenergy.io/)'s **Electrochemical** models (such as [About:DFN](https://aboutenergy.notion.site/About-DFN-Documentation-0c4a5b0ebb974441ab4783dd2f1d4d81#c73e7cd04ac64c0bbc061bbf74087e28)) in the [PyBaMM](https://pybamm.org/) implementation.

AEPyBaMM is an interface between the [PyBaMM](https://pybamm.org) package for battery modelling and the [BPX](https://bpxstandard.com) package for expression of electrochemical parameter sets. AEPyBaMM v0.2.1 requires PyBaMM v25.10+ (last supported version PyBaMM v25.10) and BPX v0.5.

The core functionality of AEPyBaMM is expressed through the function `get_params`, which combines a BPX parameter set and any user-defined options to yield a tuple of self-consistent `pybamm.ParameterValues` and `pybamm.lithium_ion.{model}` objects.

Example use case:

```python
import pybamm
from aepybamm import get_params

fp_bpx = "*.json" # filepath of BPX JSON file containing parameter set

parameter_values, model = get_params(
    fp_bpx,
    **kwargs, # additional options to get_params()
)

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    **kwargs, # additional options to pybamm.Simulation
)

# The pybamm.Simulation 'sim' can now be solved and post-processed as desired.
```

AEPyBaMM additionally provides functions `solve_from_expdata` and `compare`, to aid simple solution and comparison operations on experimental data without using the PyBaMM API directly.
