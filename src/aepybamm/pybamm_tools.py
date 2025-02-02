import logging
import warnings
from numbers import Number

import numpy as np
import pybamm
from packaging.version import Version

from .bpx_tools import (
    _get_material_names,
    as_bpx,
    validate_BPX_version,
)
from .func import (
    _make_OCP,
)

PYBAMM_VERSION_MINIMUM = Version("25.1")
PYBAMM_VERSION_LATEST = Version("25.1.1")

ELECTRODES = ["Negative", "Positive"]
PYBAMM_MATERIAL_NAMES = ["Primary", "Secondary"]


def get_PyBaMM_version():
    return Version(pybamm.__version__)


def validate_PyBaMM_version():
    PyBaMM_version = get_PyBaMM_version()
    if PyBaMM_version < PYBAMM_VERSION_MINIMUM:
        raise RuntimeError(
            f"aepybamm requires PyBaMM {PYBAMM_VERSION_MINIMUM} or later. Detected version: {PyBaMM_version}."
        )
    if PyBaMM_version > PYBAMM_VERSION_LATEST:
        print(
            f"Warning: running with PyBaMM {PyBaMM_version}. Latest tested version is {PYBAMM_VERSION_LATEST}. Functionality is not guaranteed."
        )

    validate_BPX_version()


class quiet_pybamm:
    """
    Keep PyBaMM quiet (do not print WARNINGS from the logger to stdout/stderr).
    """

    def __init__(self):
        pass

    def __enter__(self):
        self.warn_context = warnings.catch_warnings()
        self.warn_context.__enter__()
        warnings.simplefilter("ignore")

        self.original_logging_level = pybamm.logger.level
        pybamm.logger.setLevel(logging.ERROR)

    def __exit__(self, *exc):
        self.warn_context.__exit__()
        pybamm.logger.setLevel(self.original_logging_level)


def _scale_param(param, scaling):
    if callable(param):
        def func_revised(*args, **kwargs):
            return scaling * param(*args, **kwargs)
        
        return func_revised
    else:
        return scaling * param


def _make_hysteresis_compatible(func_PyBaMM):
    try:
        if "dUdT" in func_PyBaMM.keywords:
            func_PyBaMM = func_PyBaMM.keywords["dUdT"]

        if isinstance(func_PyBaMM, Number):
            # Float-valued entropy coefficient, cast to PyBaMM object so it can be differentiated
            return pybamm.Scalar(func_PyBaMM)

        # Rebuild interpolant as hysteresis-compatible
        data = _extract_interp_PyBaMM_BPX(func_PyBaMM)
        return _make_OCP(data, hysteresis_compatible=True)
    except AttributeError:
        # Not a float or interpolant so assume that it is a correctly defined PyBaMM function and leave alone
        return func_PyBaMM


def _add_hysteresis_heat_source(model):
    thermal_model = model.submodels["thermal"]

    # Compute supplementary heat source variables
    Q_hys_by_electrode = {}
    for electrode in ELECTRODES:
        electrode_options = getattr(model.options, electrode.lower())
        num_phases = int(electrode_options["particle phases"])
        
        phases = [""]
        if num_phases > 1:
            phases = [(s.lower() + " ") for s in PYBAMM_MATERIAL_NAMES]
        
        hysteresis_models = electrode_options["open-circuit potential"]
        if isinstance(hysteresis_models, str):
            hysteresis_models = [hysteresis_models]

        Q_hys_el = 0
        for phase, hysteresis_model in zip(phases, hysteresis_models):
            if hysteresis_model == 'single':
                # No hysteresis for this phase, so no contributing heat source
                V_hys_el = 0
            elif hysteresis_model == 'current sigmoid':
                raise NotImplementedError("Hysteresis heat source is not yet supported for zero-state hysteresis.")
            elif hysteresis_model == 'Wycisk':
                V_hys_el = (
                    model.variables[f"{electrode} electrode {phase}OCP hysteresis [V]"]
                    * model.variables[f"{electrode} electrode {phase}hysteresis state distribution"]
                )

            ifar_vol_el = model.variables[
                f"{electrode} electrode {phase}volumetric interfacial current density [A.m-3]"
            ]
            Q_hys_el += V_hys_el * ifar_vol_el

        Q_hys_by_electrode[electrode] = Q_hys_el

    Q_hys = pybamm.concatenation(
        Q_hys_by_electrode["Negative"],
        pybamm.FullBroadcast(0, "separator", "current collector"),
        Q_hys_by_electrode["Positive"],
    )
    Q_hys_av = thermal_model._x_average(Q_hys, 0, 0)
    Q_hys_Wm2 = Q_hys_av * model.param.L
    Ageom_tot = model.param.n_electrodes_parallel *  model.param.L_y * model.param.L_z
    Q_hys_W = thermal_model._yz_average(Q_hys_Wm2) * Ageom_tot
    Q_hys_vol_av = Q_hys_W / model.param.V_cell

    model.variables.update(
        {
            "Hysteresis electrochemical heating [W.m-3]": Q_hys,
            "X-averaged hysteresis electrochemical heating [W.m-3]": Q_hys_av,
            "Hysteresis electrochemical heating per unit electrode-pair area [W.m-2]": Q_hys_Wm2,
            "Hysteresis electrochemical heating [W]": Q_hys_W,
            "Volume-averaged hysteresis electrochemical heating [W.m-3]": Q_hys_vol_av,
        }
    )

    model.variables["Total heating [W.m-3]"] += Q_hys
    model.variables["X-averaged total heating [W.m-3]"] += Q_hys_av
    model.variables["Total heating per unit electrode-pair area [W.m-2]"] += Q_hys_Wm2
    model.variables["Total heating [W]"] += Q_hys_W
    model.variables["Volume-averaged total heating [W.m-3]"] += Q_hys_vol_av

    if model.options["thermal"] == "lumped":
        # Note (UNCERTAIN IMPLEMENTATION):
        # not clear why this is still apparently necessary when the
        # total electrochemical heating is already augmented,
        # but required for energy conservation from testing.

        # Add additional heat source to lumped heat equation, targeting existing entry
        T_vol_av = model.variables["Volume-averaged cell temperature [K]"]
        rho_c_p_eff_av = model.variables["Volume-averaged effective heat capacity [J.K-1.m-3]"]
        thermal_model.rhs[T_vol_av] += Q_hys_vol_av  / rho_c_p_eff_av

        model._rhs.update({ T_vol_av: thermal_model.rhs[T_vol_av] })


def _extract_interp_PyBaMM_BPX(func_PyBaMM):
    """
    Get np.ndarrays out of an existing BPX-generated PyBaMM interpolant.
    """
    try:
        data = [func_PyBaMM.keywords[k] for k in ["x", "y"]]
    except (AttributeError, KeyError):
        raise ValueError(
            "Argument is not an interpolant generated from pybamm.ParameterValues.create_from_bpx()."
        )

    return np.column_stack(tuple(data))


def _eval_OCP(Ufunc, xLi):
    """
    Numpy- and PyBaMM-cross-compatible evaluation of callable U(x)
    """
    try:
        value = Ufunc(xLi)
    except ValueError as err:
        if isinstance(xLi, np.ndarray):
            value = Ufunc(pybamm.Vector(xLi))
        else:
            raise RuntimeError(err)
    except TypeError:
        # Some PyBaMM callables expect floats to be packed in np.ndarray, so try converting to list
        value = Ufunc([xLi])

    if isinstance(value, pybamm.Symbol):
        # PyBaMM functions may return pybamm.Symbol, or
        # if Ufunc uses a PyBaMM lookup table, return a pybamm.Interpolant.
        # Use evaluate() to yield a num x 1 x 1 np.ndarray
        value = np.ravel(value.evaluate())
        if isinstance(xLi, Number):
            # Return scalar as float
            value = value[0]

    return value


def get_default_parameter_values(fp):
    # Call library create_from_bpx()
    # Any known bugs in library method will be fixed before parameter_values is returned
    # Ignore corresponding warnings
    with quiet_pybamm():
        # Do not pass SOC_init, initial concentrations will be added by the package
        parameter_values = pybamm.ParameterValues.create_from_bpx(fp)

    process_userdefined_parameters(parameter_values, fp)
    fix_parameter_values(parameter_values)
    strip_parameter_values(parameter_values)

    return parameter_values


def process_userdefined_parameters(parameter_values, fp):
    """
    Apply processing to convert scalar parameters in user-defined BPX fields
    from About:Energy standards to expected PyBaMM format.
    """

    # Patch material names to PyBaMM requirements
    all_material_names = _get_material_names(as_bpx(fp))
    for electrode_material_names in all_material_names:
        if len(electrode_material_names) > 1:
            # Substitute imported materials with appropriate PyBaMM equivalents
            for original_name, pybamm_name in zip(electrode_material_names, PYBAMM_MATERIAL_NAMES):
                original_prefix = original_name + ": "
                new_prefix = pybamm_name + ": "              
                
                params_to_replace = [k for k in parameter_values if k.startswith(original_prefix)]
                
                params_new = {
                    param.replace(original_prefix, new_prefix): parameter_values.pop(param)
                    for param in params_to_replace
                }

                parameter_values.update(
                    params_new,
                    check_already_exists=False,
                )

    # Handle peculiar definition of decay rate. A:E BPX JSON specifies a 'true' decay rate from the Plett model.
    # (discussion at https://github.com/pybamm-team/PyBaMM/issues/4332, not yet fixed)
    decay_rate_params = [k for k in parameter_values if "hysteresis decay rate" in k]
    for param in decay_rate_params:
        if ":" in param:
            # Multi-material electrode
            phase = param.split(":")[0] + ": "
            electrode = param.removeprefix(phase).split()[0]
        else:
            # Single-material electrode
            phase = ""
            electrode = param.split()[0]
        electrode += " electrode"

        avol_phase = parameter_values[f"{phase}{electrode} surface area per unit volume [m-1]"]
        L_el = parameter_values[f"{electrode} thickness [m]"]
        Ageom_cell = parameter_values["Electrode area [m2]"]

        # Note(ED): assumes scalar value
        parameter_values[param] *= avol_phase * L_el * Ageom_cell / 3600


def fix_parameter_values(parameter_values):
    """
    Fix known bugs in output from pybamm.ParameterValues.create_from_bpx()
    """
    PyBaMM_version = get_PyBaMM_version()

    # No known bugs in PyBaMM 25.1 (only supported version)
    # When needed, filter on PyBaMM_version to implement version-specific bug fixes
    pass


def strip_parameter_values(parameter_values):
    """
    Remove any unwanted parameters that are introduced by the standard BPX import.
    """
    unused_param_substrings = [
        # "surface area per unit volume" was used by BPX import to create "active material volume fraction",
        # but it is later overwritten in the PyBaMM interface submodel: remove now for clarity.
        # Note: this parameter is also used in process_userdefined_parameters but only when hysteresis model is active,
        # so never relevant when degradation_state is specified. This needs to be reviewed when degradation_state
        # becomes hysteresis-compatible.
        "surface area per unit volume [m-1]",
    ]

    params_to_strip = set()
    for s in unused_param_substrings:
        params_to_strip.update([k for k in parameter_values if s in k])
    
    for param in params_to_strip:
        del parameter_values[param]


def get_model_class(model_name):
    return getattr(pybamm.lithium_ion, model_name)
