# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.2.3 - 2026-08-04

### Added

- `get_params()` now accepts a parsed `bpx.BPX` object as well as a BPX JSON filepath.
- `get_params(hysteresis_model=...)` now accepts a `(negative, positive)` tuple, allowing hysteresis to be configured independently for each electrode. A single string applies to both electrodes subject to appropriate hysteresis data being available, as before.
- Support for BPX v1.1 parameter files. Legacy About:Energy BPX v0.x files are automatically converted to the v1.x schema on load. :
  - the generic structural conversion is performed by the `bpx` library via `bpx.convert_v0_to_v1()`
  - the About:Energy-specific hysteresis parameters are moved out of `User-defined` and renamed (`electrode lithiation OCP [V]` → `OCP (lithiation) [V]`, `electrode delithiation OCP [V]` → `OCP (delithiation) [V]`, `particle hysteresis decay rate` → `OCP hysteresis decay constant`)

### Changed

- Default `voltage as a state` to `"false"` to preserve the validated PyBaMM <=26.6 formulation for interpolated-current drive cycles after PyBaMM 26.7 changed its upstream default to `"true"`. ([pybamm-team/PyBaMM#5573](https://github.com/pybamm-team/PyBaMM/pull/5573))
- Updated `pybamm` dependency to `>=26.7,<26.8` and runtime `bpx` dependency to `==1.1.1`. PyBaMM < 26.7 and runtime BPX < 1.1.1 are no longer supported.
- Parameter input data now conforms to the BPX v1.1 definition.

### Removed

- Removed the redundant `apply_one_state_hysteresis()` function; copying the decay rate to the lithiation/delithiation branches is no longer required under PyBaMM >= 26.6.
- Dropped support for functional (lithiation-dependent) hysteresis decay rates. BPX v1.1 only permits a scalar `OCP hysteresis decay constant`, so legacy A:E BPX v0.x files that specify the `particle hysteresis decay rate` as a function of lithiation extent (e.g. interpolated data) are no longer supported and now raise a clear error on load. Use `aepybamm <= 0.2.2` if a functional decay rate is required.


## v0.2.2 - 2026-04-27

### Changed

- Updated `pybamm` dependency to `>=26.4`. PyBaMM <= 26.3 is not supported.
- Removed deprecated `check_already_exists` kwarg from all `ParameterValues.update()` calls.


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
