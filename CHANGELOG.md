# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1 - 2025-02-04

Initial release.

**AEPyBaMM** (`aepybamm`) is a Python library that supports the use of About:Energy's **Electrochemical** models (such as [About:DFN](https://aboutenergy.notion.site/About-DFN-Documentation-0c4a5b0ebb974441ab4783dd2f1d4d81#c73e7cd04ac64c0bbc061bbf74087e28)) in the [PyBaMM](https://pybamm.org/) implementation.

### Added

- `get_params` function to yield self-consistent `pybamm.ParameterValues` and `pybamm.lithium_ion.{model}` objects, given a BPX v0.5 parameter set, according to user-defined options:
  - Initial SOC according to any OCV-SOC definition
  - OCV-based initialisation (undegraded single-phase electrodes only)
  - Degradation states including degradation modes and resistance increase (single-phase electrodes only)
  - Blended negative electrodes (up to two components)
  - Hysteresis, including initialisation from a specific hysteresis state for blended electrodes
- `solve_from_expdata` function to support simulation from an experimentally defined current drive cycle and, optionally, temperature drive cycle
- `compare` function to compare simulated data to experimental data
