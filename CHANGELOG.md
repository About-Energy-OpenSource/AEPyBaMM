# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.1 - 2026-01-08

### Added
- Support for material-specific faradaic resistance increase in multi-material negative electrode degradation state.

### Fixed
- Fixed bug setting hysteresis initial state for positive electrode.
- Improved stability of lithiation bounds calculation across parameter sets.

## v0.2.0 - 2025-11-05

### Added

- `pybamm_print_tools` module, with function `pybamm_print_tools.as_string()` to generate a readable plain text summary of the mathematical definition of a PyBaMM model.
- Helper function `pybamm_tools.update_PyBaMM_experiment()` to update the arguments of an existing `pybamm.Experiment` instance.

### Fixed

- Rescaled hysteresis decay rate parameters imported from BPX files so that one-state hysteresis follows PyBaMM 25.10 parameter definitions.

### Changed

- Updated `pybamm` dependency to `>=25.10`. PyBaMM <= 25.8 is no longer supported.
- One-state hysteresis is now implemented internally using the simpler "one-state hysteresis" model (formerly 'Axen' model) rather than by editing the "one-state differential capacity hysteresis" model (formerly 'Wycisk' model). The actual mathematical model is unchanged.
- Various internal refactors.

### Removed

- Removed `add_hysteresis_heat_source` option from `get_params()` as PyBaMM's default model now includes this heat source contribution.
- Removed various internal functions that are no longer required for PyBaMM >= 25.10.

## v0.1.4 - 2025-10-23

### Added
- Degradation state initialisation support for hysteresis and blended negative electrodes.
- Support for hysteresis initial state to be specified separately from preceding state.
- Parameterised hysteresis decay rate can now be a function of lithiation extent.

## v0.1.3 - 2025-09-22

### Added

- Branch-dependent OCV-SOC determination and voltage-based SOC initialisation is now supported for parameter sets using hysteresis and/or multi-material negative electrodes.

### Fixed

- Fixed newly identified PyBaMM [BPX import bug #5193](https://github.com/pybamm-team/PyBaMM/issues/5193) (incorrect porosity import).
- Fixed incorrect implementation (from v0.1.2) of hysteresis definitions in PyBaMM 25.8.

### Changed

- Minor internal refactors.

## v0.1.2 - 2025-09-11

### Fixed
- Fixed convergence issues for specific initial condition and parameter set combinations.

### Changed

- Updated `solve_from_expdata()` to use the IDAKLU solver.
- Updated `pybamm` dependency to `>=25.4,<=25.8`. PyBaMM 25.1 is no longer supported.
- Updated `Python` dependency to `>=3.10`.
- Hysteresis definition has been updated to support up to PyBaMM 25.8.

## v0.1.1 - 2025-03-27

### Added

- Added supplementary examples and parameters for an NMC cell and an LFP cell (from legacy [BPX release information](https://github.com/About-Energy-OpenSource/About-Energy-BPX-Parameterisation)).

### Fixed

- The minimum Python version requirement (3.9) is stated correctly.
- Fixed error cases when calling `get_params()` with functional open-circuit potential input.
- Fixed incorrect RMSE in `compare()` if the time series does not start at t = 0.
- Tidied argument validation in various functions.

### Changed

- Updated the About:Energy Gen1 demo cell parameters (in BPX JSON) to v2.0, correcting unphysical porosity for positive electrode.
- Updated the time-averaged RMSE evaluation method in `compare()` to weight each neighbouring data point equally in each time interval.
- Minor internal refactors.

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
