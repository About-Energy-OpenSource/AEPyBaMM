import functools
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
    _allow_unused_args_1d,
)

PYBAMM_VERSION_MINIMUM = Version("25.10")
PYBAMM_VERSION_LATEST = Version("25.10.0")

ELECTRODES = ["Negative", "Positive"]
PYBAMM_MATERIAL_NAMES = ["Primary", "Secondary"]

PYBAMM_HYSTERESIS_MODELS = {
    "zero-state": "current sigmoid",
    "one-state": "one-state hysteresis",
}


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


def _as_PyBaMM_option(x):
    '''
    Convert a 2-tuple of lists, representing two multi-material electrodes, to a 2-tuple of tuples or single values
    '''
    out = tuple([
        tuple(x_el) if len(x_el) > 1 else x_el[0]
        for x_el in x
    ])
    return out


def _scale_param(param, scaling):
    if callable(param):
        def func_revised(*args, **kwargs):
            return scaling * param(*args, **kwargs)

        return func_revised
    else:
        return scaling * param


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
    except (AttributeError, ValueError) as err:
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

    params_bpx = as_bpx(fp)
    process_userdefined_parameters(parameter_values, params_bpx)
    fix_parameter_values(parameter_values, params_bpx)
    strip_parameter_values(parameter_values)

    return parameter_values


def process_userdefined_parameters(parameter_values, params_bpx):
    """
    Apply processing to convert scalar parameters in user-defined BPX fields
    from About:Energy standards to expected PyBaMM format.
    """

    # Patch material names to PyBaMM requirements
    all_material_names = _get_material_names(params_bpx)
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
        
        # Rescale hysteresis decay rate according to About:Energy legacy convention
        decay_rate_multiplier = 2

        if isinstance(parameter_values[param], functools.partial):
            func_original = parameter_values[param].keywords['fun']
            parameter_values[param] = _allow_unused_args_1d(
                _scale_param(func_original, decay_rate_multiplier)
            )
        else:
            # Scalar value
            parameter_values[param] *= decay_rate_multiplier


def fix_parameter_values(parameter_values, params_bpx):
    """
    Fix known bugs in output from pybamm.ParameterValues.create_from_bpx()
    """
    PyBaMM_version = get_PyBaMM_version()
    # No current known bugs
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


def update_PyBaMM_experiment(experiment, **kwargs):
    """
    Update the 'args' attribute of a pybamm.Experiment with the content of kwargs.
    Returns a new pybamm.Experiment to ensure correct argument-based initialization.

    Allowed kwargs match the keyword arguments of pybamm.Experiment.__init__():

    - "operating_conditions", "period", "temperature", "termination"
    """
    kwargs_old = _PyBaMM_experiment_args_as_dict(experiment)
    kwargs_new = kwargs_old | kwargs

    return pybamm.Experiment(**kwargs_new)


def _PyBaMM_experiment_args_as_dict(experiment):
    """
    Return the 'args' attribute of a pybamm.Experiment as a dict.
    """
    PYBAMM_EXPERIMENT_KWARGS = (
        "operating_conditions", "period", "temperature", "termination",
    )
    return dict(zip(PYBAMM_EXPERIMENT_KWARGS, experiment.args))
