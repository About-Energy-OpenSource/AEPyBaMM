import json
import os

import numpy as np
from pybamm import constants

from .func import (
    _make_generic_func_ce_T,
    _make_j0_func,
    _unflatten,
)
from .pybamm_tools import (
    PYBAMM_MATERIAL_NAMES,
    _add_hysteresis_heat_source,
    _make_hysteresis_compatible,
    _scale_param,
    get_default_parameter_values,
    get_model_class,
    validate_PyBaMM_version,
)
from .sci_tools import (
    HYSTERESIS_BRANCH_MAP,
    HYSTERESIS_BRANCHES_ELECTRODE,
    ELECTRODES,
    _get_hysteresis_init_branch_electrode,
    add_initial_concentrations,
    calc_lithium_inventory,
    convert_soc,
    get_ocv_thermodynamic,
)

VALID_MODEL_TYPES = ["DFN", "SPMe", "SPM"]
VALID_HYSTERESIS_MODELS = ["none", "zero-state", "one-state"]
VALID_HYSTERESIS_BRANCHES = ["average", "charge", "discharge"]
VALID_DEGRADATION_KEYS = [
    "LAM_NE",
    "LAM_PE",
    "LLI",
    "RI_far_NE",
    "RI_far_PE",
    "RI_electrolyte",
    "R0_addn [Ohm]",
]

PARAMS_HYSTERESIS_DIFF = [
    "lithiation OCP [V]",
    "delithiation OCP [V]",
    "OCP [V]",
    "OCP entropic change [V.K-1]",
]


def get_params(
    fp,
    parameter_set=None,
    SOC_init=1,
    SOC_definition=None,
    OCV_init=None,
    degradation_state=None,
    htc_ext=None,
    model_type="DFN",
    hysteresis_model="none",
    hysteresis_branch="average",
    hysteresis_preceding_state="average",
    blended_electrode=None,
    add_hysteresis_heat_source=False,
    extra_model_opts=None,
    trim_model_events=True,
):
    """
    Get compatible pybamm.ParameterValues and pybamm.lithium_ion.(model_type) objects
    from a BPX JSON file provided by About:Energy.

    ---
    fp : str
        Filepath of BPX JSON file
    parameter_set : str
        Placeholder, **not currently used**
    SOC_init : float
        Initial thermodynamic SOC (default: 100%)
    SOC_definition : dict (optional)
        Definition for the OCV-SOC relation in which SOC_init is to be interpreted. Expected keys:
            "data": OCV-SOC relation as an n x 2 np.ndarray (columns: SOC, OCV)
            "method" (optional): conversion method between specified OCV-SOC relation and built-in parameters
                Allowed values:
                    "voltage" (default): match initial voltage from specified OCV-SOC relation
                    "linear_endpoints": linear transformation to equivalent SOC using maximum and minimum specified OCV(SOC) values
                    "linear_optimized": linear transformation to equivalent SOC using best fit of specified OCV-SOC relation to built-in parameters
        'SOC_definition' is only compatible with single-phase parameter sets without hysteresis (hysteresis_model="none", blended_electrode=None)
    OCV_init : float (optional)
        Initial open-circuit voltage. If specified (not None), the model is initialized at the defined voltage, and 'SOC_init' and 'SOC_definition' are ignored.
        'OCV_init' is only compatible with single-phase parameter sets without hysteresis (hysteresis_model="none", blended_electrode=None)
    degradation_state : dict
        Dict values are floats with allowed fields any of "LAM_PE", "LAM_NE", "LLI" for corresponding thermodynamic degradation modes,
        and any of "RI_far_PE", "RI_far_NE", "RI_electrolyte", "R0_addn [Ohm]" for corresponding resistance increases.
        'degradation_state' is only compatible with single-phase parameter sets without hysteresis (hysteresis_model="none", blended_electrode=None)
    htc_ext : float (optional)
        External heat transfer coefficient (W/m^2/K) for lumped thermal model.
    model_type : str (optional, default: "DFN")
        Electrochemical model type. Allowed values: "DFN", "SPMe", "SPM"
    hysteresis_model : str (optional)
        "none" (default), "zero-state", "one-state" (Plett-Wycisk model)
    hysteresis_branch : str (optional)
        Hysteresis branch to use when hysteresis_model is "none". Ignored if hysteresis_model is set.
        Allowed values: "average" (default), "charge", "discharge"
    hysteresis_preceding_state : str (optional)
        Set the SOC initialisation according to a particular hysteresis branch, in case of blended electrode.
        This functionality is explicitly targeted at mixed negative electrodes and assumes that there is
        zero hysteresis at the SOC100 limit. SOC_init is interpreted coulombically relative to SOC100.
        Ignored if blended_electrode == None or hysteresis_model == None.
        Allowed values: "average" (default), "charge", "discharge"
    add_hysteresis_heat_source : bool (optional, default: False)
        Set True to add the missing hysteresis heat source in PyBaMM's electrochemical heat source model (https://github.com/pybamm-team/PyBaMM/issues/3867).
        Requires that all heat sources can be computed for isothermal models.
        One-state hysteresis only; not currently compatible with zero-state hysteresis.
    blended_electrode : 2-tuple of bool (optional) or None
        Default: None -> (False, False). Set True to use blended electrode data for (neg, pos) electrodes.
        Ignored if hysteresis_model == "none".
    extra_model_opts : dict (optional)
        Dictionary of additional user-defined fields to include in the "options" dict argument to the PyBaMM model constructor (e.g. pybamm.lithium_ion.DFN(options=[...])).
        If extra_model_opts conflicts with required model options from other settings, an error will be raised.
    trim_model_events : bool (optional, default: True)
        Set True to remove undesired initial model Events that may cause premature voltage cut-off of PyBaMM simulations due to interpolation error.
    """
    validate_PyBaMM_version()

    # Process arguments
    required_model_opts = {}
    extra_model_opts = extra_model_opts or {}
    _validate_args_get_params(**locals())
    fp_bpx = _get_bpx_src(fp, parameter_set)

    # Get parameter values at SOC = 1, fixing bugs where needed
    # SOC initialisation is applied later
    parameter_values = get_default_parameter_values(fp_bpx)

    # Add any functions defined in BPX "User-defined section"
    build_exchange_current_density(parameter_values)
    build_BPX_incompatible(parameter_values)

    # Apply  heat transfer coefficient
    if htc_ext is not None:
        apply_htc_ext(parameter_values, htc_ext)

    # Apply degradation state
    degradation_state = {
        k: v for k, v in degradation_state.items()
        if v != 0
    } if degradation_state is not None else None
    if degradation_state == {}:
        # All user-provided degradation state values were zero
        # Should do nothing, so act hereafter as if no degradation state applies
        degradation_state = None

    if degradation_state is not None:
        apply_degradation_state(parameter_values, degradation_state)

    # Handle series resistance
    if (
        "Contact resistance [Ohm]" in parameter_values
        and parameter_values["Contact resistance [Ohm]"] > 0
    ):
        required_model_opts.update(
            {
                "contact resistance": "true",
            }
        )

    # Handle blended electrodes
    phases_by_electrode = _get_phases_by_electrode(blended_electrode)
    _validate_phases_by_electrode(parameter_values, phases_by_electrode)
    required_model_opts.update(
        {
            "particle phases": tuple(
                str(len(phases)) for phases in phases_by_electrode
            ),
        }
    )

    # Hysteresis model handling
    if hysteresis_model == "none":
        hysteresis_init_branches = None
        if hysteresis_branch != "average":
            apply_hysteresis_branch(parameter_values, hysteresis_branch, phases_by_electrode)
    else:
        # Case where hysteresis_model != "none"
        use_hysteresis = [
            get_hysteresis_model_by_electrode(
                hysteresis_model,
                parameter_values,
                electrode,
                phases,
            )
            for electrode, phases in zip(ELECTRODES, phases_by_electrode)
        ]

        if hysteresis_model == "one-state":
            apply_one_state_hysteresis(parameter_values, use_hysteresis, phases_by_electrode)

        hysteresis_init_branches = _get_hysteresis_init_branch_electrode(
            use_hysteresis,
            hysteresis_preceding_state,
        )

        hysteresis_model_opts = {
            "open-circuit potential": tuple(use_hysteresis),
        }
        if add_hysteresis_heat_source:
            hysteresis_model_opts.update(
                {
                    "calculate heat source for isothermal models": "true",
                },
            )

        required_model_opts.update(hysteresis_model_opts)

    # Apply initial concentrations
    add_initial_concentrations(
        parameter_values,
        phases_by_electrode,
        hysteresis_init_branches=hysteresis_init_branches,
        SOC_init=convert_soc_init(SOC_init, OCV_init, SOC_definition, parameter_values),
        update_bounds=(degradation_state is not None),
    )

    # Create model
    model_opts = _combine_model_opts(required_model_opts, extra_model_opts)
    model_class = get_model_class(model_type)
    model = model_class(options=model_opts)

    if add_hysteresis_heat_source:
        _add_hysteresis_heat_source(model)

    if trim_model_events:
        apply_trim_model_events(model, SOC_init)

    return (parameter_values, model)


def convert_soc_init(SOC_init, OCV_init, SOC_definition, parameter_values):
    if OCV_init is not None:
        ocv_soc = get_ocv_thermodynamic(parameter_values)
        return np.interp(OCV_init, ocv_soc[:, 1], ocv_soc[:, 0])
    elif SOC_definition is not None:
        if "method" not in SOC_definition:
            SOC_definition["method"] = "voltage"

        # Correct SOC from intended value on specified OCV-SOC scale to intended thermodynamic value
        return convert_soc(
            SOC_init,
            SOC_definition["data"],
            get_ocv_thermodynamic(parameter_values),
            SOC_definition["method"],
        )
    else:
        return SOC_init


def apply_degradation_state(parameter_values, degradation_state):
    degradation_scaled_vals = {}

    # Compute cyclable lithium content from beginning-of-life parameter set
    ncyc = calc_lithium_inventory(parameter_values)

    if "LLI" in degradation_state:
        # Apply LLI correction to cyclable lithium content
        ncyc *= (1 - degradation_state["LLI"])

    degradation_scaled_vals.update(
        {"AE: Total cyclable lithium inventory [mol.m-2]": ncyc}
    )

    # Apply LAM multiples
    volume_fraction_params = [
        f"{el} electrode active material volume fraction" for el in ELECTRODES
    ]
    updated_volume_fractions = {k: parameter_values[k] for k in volume_fraction_params}
    for loss_factor, param in zip(["LAM_NE", "LAM_PE"], volume_fraction_params):
        if loss_factor in degradation_state:
            mul_lam = 1 - degradation_state[loss_factor]
            updated_volume_fractions[param] *= mul_lam

    degradation_scaled_vals.update(updated_volume_fractions)

    # Apply resistance increase multiples
    kinetic_params = [
        f"{el} electrode exchange-current density [A.m-2]" for el in ELECTRODES
    ]
    for loss_factor, param in zip(["RI_far_NE", "RI_far_PE"], kinetic_params):
        if loss_factor in degradation_state:
            mul_far = 1 / (1 + degradation_state[loss_factor])
            degradation_scaled_vals[param] = _scale_param(parameter_values[param], mul_far)

    if "RI_electrolyte" in degradation_state:
        mul_electrolyte = 1 / (1 + degradation_state["RI_electrolyte"])
        for param in ["Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]"]:
            degradation_scaled_vals[param] = _scale_param(parameter_values[param], mul_electrolyte)
    
    # Apply supplementary series resisatnce
    if "R0_addn [Ohm]" in degradation_state:
        if "Contact resistance [Ohm]" in parameter_values:
            R0_existing = parameter_values["Contact resistance [Ohm]"]
        else:
            R0_existing = 0

        R0_new = R0_existing + degradation_state["R0_addn [Ohm]"]
        degradation_scaled_vals.update({"Contact resistance [Ohm]": R0_new})

    parameter_values.update(
        degradation_scaled_vals,
        check_already_exists=False,
    )


def apply_htc_ext(parameter_values, htc_ext):
    # Add heat transfer coefficient
    parameter_values.update(
        {
            "Total heat transfer coefficient [W.m-2.K-1]": htc_ext,
        },
        check_already_exists=False,
    )


def _get_phases_by_electrode(blended_electrode):
    blended_electrode = blended_electrode or (False, False)

    phases_by_electrode = [
        [s + ": " for s in PYBAMM_MATERIAL_NAMES] if blended else [""]
        for blended in blended_electrode
    ]

    return phases_by_electrode


def _validate_phases_by_electrode(parameter_values, phases_by_electrode):
    if len(phases_by_electrode[1]) > 1:
        raise ValueError("Positive electrode blends are not yet supported.")

    # Rough compatibility check on presence / absence of blended electrode data
    for electrode, phases in zip(ELECTRODES, phases_by_electrode):
        if any(phases):
            if not (
                any("Primary:" in k for k in parameter_values)
                and any("Secondary:" in k for k in parameter_values)
            ):
                raise ValueError(
                    f"No parameter data found to treat {electrode.lower()} electrode as blended material."
                    + " Try disabling the 'blended_electrode' keyword argument to get_params()."
                )
        else:
            if (
                f"Maximum concentration in {electrode.lower()} electrode [mol.m-3]"
                not in parameter_values
            ):
                raise ValueError(
                    f"No parameter data to treat {electrode.lower()} electrode as single-material."
                    + " Try setting the 'blended_electrode' keyword argument to get_params()."
                )


def build_exchange_current_density(parameter_values):
    # Build BPX-incompatible exchange current density where specified
    tag_j0 = "exchange-current density pre-multiplier"
    param_subst = [param for param in parameter_values if tag_j0 in param]
    for param in param_subst:
        coeffs_const = {
            "j0_ref": constants.F * parameter_values[param.replace(tag_j0, "reaction rate constant [mol.m-2.s-1]")],
            "Ea": parameter_values[param.replace(tag_j0, "reaction rate constant activation energy [J.mol-1]")],
            "Tref": parameter_values["Reference temperature [K]"],
        }

        parameter_values[param.replace("pre-multiplier", "[A.m-2]")] = _make_j0_func(
            coeffs_const,
            func_premul=parameter_values[param],
        )

        del parameter_values[param]
        del parameter_values[param.replace(tag_j0, "reaction rate constant [mol.m-2.s-1]")]    


def build_BPX_incompatible(parameter_values):
    # Build BPX-incompatible functions other than exchange-current density
    func_headers = [
        s.partition(" func_type ") for s in parameter_values if " func_type " in s
    ]
    if len(func_headers) > 0:
        params_BPX_incompatible, _, func_types_BPX_incompatible = tuple(
            list(col) for col in zip(*func_headers)
        )
    else:
        params_BPX_incompatible = []
        func_types_BPX_incompatible = []

    for param, func_type in zip(params_BPX_incompatible, func_types_BPX_incompatible):
        coeffs = _unflatten(
            {
                k.replace(param + " ", ""): v
                for k, v in parameter_values.items()
                if param in k and param != k and "func_type" not in k
            }
        )
        parameter_values[param] = _make_generic_func_ce_T(func_type, coeffs)
        for k in parameter_values.copy():
            if param in k and param != k:
                del parameter_values[k]


def _has_hysteresis_data(parameter_values, electrode, phase):
    return all(
        [
            f"{phase + electrode} electrode {branch} OCP [V]" in parameter_values
            for branch in HYSTERESIS_BRANCHES_ELECTRODE
        ]
    )


def apply_hysteresis_branch(parameter_values, hysteresis_branch, phases_by_electrode):
    for electrode, phases in zip(ELECTRODES, phases_by_electrode):
        branch = HYSTERESIS_BRANCH_MAP[hysteresis_branch][electrode]
        
        for phase in phases:
            if _has_hysteresis_data(parameter_values, electrode, phase):
                ocp_dst = f"{phase + electrode} electrode OCP [V]"
                ocp_src = f"{phase + electrode} electrode {branch} OCP [V]"

                # Move the appropriate (de)lithiation branch to the general OCP parameter
                # Delete the original branch parameters
                parameter_values[ocp_dst] = parameter_values[ocp_src]
                for branch in HYSTERESIS_BRANCHES_ELECTRODE:
                    del parameter_values[f"{phase + electrode} electrode {branch} OCP [V]"]


def get_hysteresis_model_by_electrode(hysteresis_model, parameter_values, electrode, phases):
    if hysteresis_model == "zero-state":
        hysteresis_model_PyBaMM = "current sigmoid"
    elif hysteresis_model == "one-state":
        hysteresis_model_PyBaMM = "Wycisk"
    else:
        raise ValueError(f"Invalid hysteresis model: {hysteresis_model}")

    has_hysteresis_data_by_phase = [
        _has_hysteresis_data(parameter_values, electrode, phase)
        for phase in phases
    ]
    hysteresis_model_by_electrode = tuple(
        hysteresis_model_PyBaMM if has_hysteresis_data
        else "single"
        for has_hysteresis_data in has_hysteresis_data_by_phase
    )

    if len(phases) == 1:
        hysteresis_model_by_electrode = hysteresis_model_by_electrode[0]

    return hysteresis_model_by_electrode


def apply_one_state_hysteresis(parameter_values, use_hysteresis, phases_by_electrode):
    for electrode, phases, use_hysteresis_electrode in zip(ELECTRODES, phases_by_electrode, use_hysteresis):
        if isinstance(use_hysteresis_electrode, str):
            # Place scalar str in 1-tuple
            use_hysteresis_electrode = (use_hysteresis_electrode,)

        for phase, use_hysteresis_phase in zip(phases, list(use_hysteresis_electrode)):
            if use_hysteresis_phase == "Wycisk":
                # Enforce zero switching factor
                key_switch = f"{phase + electrode} particle hysteresis switching factor"
                if key_switch in parameter_values and parameter_values[key_switch] != 0:
                    raise ValueError(key_switch + " only supported with zero value.")

                # Set initial condition
                key_init = f"{phase}Initial hysteresis state in {electrode.lower()} electrode"

                parameter_values.update(
                    {
                        key_switch: 0,
                        key_init: 0,
                    },
                    check_already_exists=False,
                )

                # Rebuild parameters for compatibility with "Wycisk" (one-state) OCP
                for param in PARAMS_HYSTERESIS_DIFF:
                    key = phase + electrode + " electrode " + param
                    parameter_values[key] = _make_hysteresis_compatible(parameter_values[key])


def apply_trim_model_events(model, SOC_init, SOC_tol=0.05):
    # Remove events that might cause unintended early model termination at initial conditions
    if SOC_init > (1 - SOC_tol):
        model.events = [event for event in model.events if "Maximum voltage" not in event.name]
    elif SOC_init < SOC_tol:
        model.events = [event for event in model.events if "Minimum voltage" not in event.name]


def _validate_args_get_params(
    # for runtime validation checking
    SOC_definition,
    OCV_init,
    degradation_state,
    htc_ext,
    model_type,
    hysteresis_model,
    hysteresis_branch,
    hysteresis_preceding_state,
    blended_electrode,
    extra_model_opts,
    # no runtime validation checking on remaining arguments
    **kwargs,
):
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type '{model_type}'. " + 
            f"Supported model types are: {', '.join(VALID_MODEL_TYPES)}"
        )

    if hysteresis_model not in VALID_HYSTERESIS_MODELS:
        raise ValueError(
            f"Unsupported hysteresis model type '{hysteresis_model}'. " + 
            f"Supported hysteresis model types are: {', '.join(VALID_HYSTERESIS_MODELS)}"
        )
    
    if hysteresis_branch not in VALID_HYSTERESIS_BRANCHES:
        raise ValueError(
            f"Unsupported hysteresis branch '{hysteresis_branch}'. " +
            f"Supported model types are: {', '.join(VALID_HYSTERESIS_BRANCHES)}"
        )
    
    if hysteresis_preceding_state not in VALID_HYSTERESIS_BRANCHES:
        raise ValueError(
            f"Unsupported hysteresis preceding state '{hysteresis_preceding_state}'. " +
            f"Supported model types are: {', '.join(VALID_HYSTERESIS_BRANCHES)}"
        )

    if degradation_state is not None:
        if blended_electrode is not None or hysteresis_model != "none":
            raise NotImplementedError(
                "Degradation state is only supported for single-phase electrodes with no hysteresis."
            )

        if not isinstance(degradation_state, dict):
            raise TypeError("'degradation_state' must be a 'dict'.")

        for k in degradation_state:
            if k not in VALID_DEGRADATION_KEYS:
                raise ValueError(
                    f"Unsupported field in 'degradation_state': {k}. Supported fields are: {', '.join(VALID_DEGRADATION_KEYS)}"
                )

    if (
        "thermal" in extra_model_opts
        and extra_model_opts["thermal"] != "isothermal"
        and htc_ext is None
    ):
        raise ValueError(
            "When using a thermal model, specify the heat transfer coefficient in 'get_params' with keyword argument 'htc_ext'."
        )

    if OCV_init is not None:
        if blended_electrode is not None or hysteresis_model != "none":
            raise NotImplementedError(
                "Voltage-based initialisation is only supported for single-phase electrodes with no hysteresis."
            )

    if SOC_definition is not None:
        if not isinstance(SOC_definition, dict) or "data" not in SOC_definition:
            raise TypeError("SOC_definition must be a dict containing a key 'data'")

        if blended_electrode is not None or hysteresis_model != "none":
            raise NotImplementedError(
                "OCV-SOC conversion is only supported for single-phase electrodes with no hysteresis."
            )


def _get_bpx_src(fp, parameter_set=None):
    with open(fp) as src:
        params_info = json.load(src)

    if "Header" not in params_info:
        raise ValueError(f"No valid JSON header at {fp}.")

    if "BPX" in params_info["Header"]:
        if parameter_set:
            raise ValueError(
                "'parameter_set' selection requires a valid 'BPX Parent' JSON file."
            )

        fp_src = fp
    elif "BPX Parent" in params_info["Header"]:
        if not parameter_set:
            parameter_set = params_info["Header"]["default"]
        elif parameter_set not in params_info["Parameter Sets"]:
            raise ValueError(f"Parameter set '{parameter_set}' not defined in {fp}.")

        fp_src = os.path.join(
            os.path.dirname(fp), params_info["Parameter Sets"][parameter_set] + ".json"
        )
    else:
        raise ValueError(f"No valid JSON header at {fp}.")

    return fp_src


def _combine_model_opts(required_model_opts, extra_model_opts):
    for k in extra_model_opts:
        if k in required_model_opts:
            # Do not allow optional settings to override required settings
            raise ValueError(
                f"Required setting '{k}': '{required_model_opts[k]}', derived from other arguments to get_params(), cannot be replaced with setting '{k}': '{extra_model_opts[k]}' from 'extra_model_opts'."
            )

    model_opts = required_model_opts.copy()
    model_opts.update(extra_model_opts)

    return model_opts
