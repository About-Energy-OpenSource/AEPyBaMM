import numpy as np
from scipy.optimize import fsolve, minimize

from .pybamm_tools import (
    ELECTRODES,
    _eval_OCP,
)

HYSTERESIS_BRANCHES_ELECTRODE = ["delithiation", "lithiation"]
HYSTERESIS_BRANCH_MAP = {
    "charge": {
        "Negative": "lithiation",
        "Positive": "delithiation",
    },
    "discharge": {
        "Negative": "delithiation",
        "Positive": "lithiation",
    },
}


def _map_hysteresis_init_state(hysteresis_preceding_state, hysteresis_model_el, el):
    branch = ""
    if hysteresis_model_el != "single" and hysteresis_preceding_state != "average":
        branch = HYSTERESIS_BRANCH_MAP[hysteresis_preceding_state][el]
    
    if len(branch) > 0:
        # Add extra space for insertion in parameter key
        branch += " "
    
    return branch


def _get_hysteresis_init_branch_electrode(use_hysteresis, hysteresis_preceding_state):
    return tuple([
        _map_hysteresis_init_state(hysteresis_preceding_state, use_hysteresis_el, el)
        for use_hysteresis_el, el in zip(use_hysteresis, ELECTRODES)
    ])


def _is_monotonic(data):
    """
    Return True if a 1D function y(x) expressed in num x 2 np.ndarray is monotonic, else False.
    """
    data = data[data[:, 0].argsort()]
    return np.all(np.diff(data[:, 1]) > 0)


def _get_lithiation_bounds(parameter_values, blended_electrode=None):
    if blended_electrode is None:
        blended_electrode = (False, False)

    if not isinstance(blended_electrode, tuple) or len(blended_electrode) != 2:
        raise TypeError("2-tuple expected in _get_lithiation_bounds().")

    lithiation_bounds = {}
    for electrode, is_blended in zip(ELECTRODES, blended_electrode):
        stoichiometry_params = {
            k: v for k, v in parameter_values.items()
            if "stoichiometry" in k and electrode.capitalize() in k
        }

        if len(stoichiometry_params) == 2:
            if not is_blended:
                lithiation_bounds[electrode.lower()] = tuple(
                    sorted(stoichiometry_params.values())
                )
            else:
                raise ValueError(
                    f"Missing stoichiometry specification for blended {electrode.lower()} electrode."
                )
        elif len(stoichiometry_params) >= 4:
            try:
                if is_blended:
                    lithiation_bounds[electrode.lower()] = [
                        tuple(
                            sorted([v for k, v in stoichiometry_params.items() if phase in k])
                        )
                        for phase in ["Primary", "Secondary"]
                    ]
                else:
                    stoichiometry_params_blended = {
                        k: v for k, v in stoichiometry_params.items()
                        if "Primary" not in k and "Secondary" not in k
                    }
                    lithiation_bounds[electrode.lower()] = tuple(
                        sorted(stoichiometry_params_blended.values())
                    )
            except KeyError:
                raise RuntimeError(
                    f"Unexpected stoichiometry values in parameter set: {stoichiometry_params.keys()}."
                )

    return lithiation_bounds


def _fsolve_safe(*args, **kwargs):
    result, _, exit_code, message = fsolve(*args, **kwargs, full_output=True)
    if exit_code == 1:
        # Success case
        return result
    else:
        raise RuntimeError(f"Failure in fsolve: {message}")


def calc_xLi_init(rel_xLi_ave, lithiation_bounds_mat, ocp_mat=None, qprop_mat=None):
    """
    Calculates lithiation extents for materials in a blended electrode given average state of lithiation and lithiation bounds.

    Parameters
    ---
    rel_xLi_ave : lithiation relative to lithiation bounds (scalar proportion)
    lithiation_bounds_mat : tuple (xLi_min, xLi_max) or list of tuples for blended electrode material
    ocp_mat : list of OCP functions for each material (used only if lithiation_bounds_mat is list of len() > 1)
    qprop_mat: list of contributing charge capacity proportions for each material (used only if lithiation_bounds_mat is list of len() > 1)
    ---

    Return
    ---
    xLi_mat - list of xLi at SOC, or scalar if length 1
    ---
    """
    if isinstance(lithiation_bounds_mat, list):
        if len(lithiation_bounds_mat) > 1:
            if (len(lithiation_bounds_mat) != len(ocp_mat)) or (len(lithiation_bounds_mat) != len(qprop_mat)):
                raise ValueError(
                    "For blended material, all _mat inputs must be lists of same length"
                )
        else:
            # Unpack one-entry list
            lithiation_bounds_mat = lithiation_bounds_mat[0]

    if not isinstance(lithiation_bounds_mat, list):
        # Single-material, linear relation only
        xLi_min, xLi_max = lithiation_bounds_mat
        return (xLi_min + rel_xLi_ave * (xLi_max - xLi_min))
    else:
        for ocp in ocp_mat:
            if not callable(ocp):
                raise TypeError("calc_xLi_init: all entries in U_mat must be callable.")
        nmat = len(lithiation_bounds_mat)
        dxLi_mat = [
            lithiation_bounds[1] - lithiation_bounds[0]
            for lithiation_bounds in lithiation_bounds_mat
        ]

        # Initial guesses
        x0 = 0.5 * np.ones(nmat + 1)

        def func(x1):
            xLi_mat = x1[:-1]
            Ueq = x1[-1]

            # Equality of OCPs
            residual = [
                _eval_OCP(ocp, xLi) - Ueq
                for ocp, xLi in zip(ocp_mat, xLi_mat)
            ]

            # Total contents sum to 1
            residual_mat_constraint = sum([
                qprop * (xLi - lithiation_bounds[0]) / dxLi
                for qprop, xLi, lithiation_bounds, dxLi in zip(qprop_mat, xLi_mat, lithiation_bounds_mat, dxLi_mat)
            ]) - rel_xLi_ave

            residual.append(residual_mat_constraint)
            return residual

        x1 = _fsolve_safe(func, x0)
        return x1[:-1]


def calc_lithium_inventory(parameter_values):
    ncyc_ref = 0
    for el in ELECTRODES:
        ncyc_ref += (
            parameter_values[f"Initial concentration in {el.lower()} electrode [mol.m-3]"]
            * parameter_values[f"{el} electrode active material volume fraction"]
            * parameter_values[f"{el} electrode thickness [m]"]
        )

    return ncyc_ref


def compute_lithiation_bounds(parameter_values):
    """
    IMPORTANT: not compatible with hysteresis or multi-phase electrode. Guarded in main function.
    """
    if "AE: Total cyclable lithium inventory [mol.m-2]" not in parameter_values:
        raise ValueError("Cyclable lithium inventory must be computed before calling.")

    ncyc = parameter_values["AE: Total cyclable lithium inventory [mol.m-2]"]
    nsat_neg = (
        parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
        * parameter_values["Negative electrode active material volume fraction"]
        * parameter_values["Negative electrode thickness [m]"]
    )
    nsat_pos = (
        parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        * parameter_values["Positive electrode active material volume fraction"]
        * parameter_values["Positive electrode thickness [m]"]
    )
    Uneg = parameter_values["Negative electrode OCP [V]"]
    Upos = parameter_values["Positive electrode OCP [V]"]
    Veod = parameter_values["Lower voltage cut-off [V]"]
    Veoc = parameter_values["Upper voltage cut-off [V]"]

    def lithium_balance(bounds, Vcell):
        xneg, xpos = tuple(bounds)

        return [
            xneg * nsat_neg + xpos * nsat_pos - ncyc,
            _eval_OCP(Upos, xpos) - _eval_OCP(Uneg, xneg) - Vcell,
        ]

    lower_bounds = _fsolve_safe(lithium_balance, [0.1, 0.9], args=Veod)
    upper_bounds = _fsolve_safe(lithium_balance, [0.9, 0.1], args=Veoc)

    bounds_electrodes = list(zip(lower_bounds, upper_bounds))

    lithiation_bounds = {
        f"{el.lower()}": tuple(sorted(bounds))
        for el, bounds in zip(ELECTRODES, bounds_electrodes)
    }
    return lithiation_bounds


def add_initial_concentrations(parameter_values, phases_by_electrode, hysteresis_init_branches=None, SOC_init=1, update_bounds=False):
    """
    Add initial concentrations to pybamm.ParameterValues.

    Note: currently ONLY supports negative electrode blends (guarded in main function) - so cheats by assuming that the positive electrode is not blended.

    update_bounds - set True to force recomputation of lithiation bounds. IMPORTANT: not compatible with hysteresis or single-phase electrode. Guarded in main function.
    """
    phases_neg, _ = phases_by_electrode
    hysteresis_init_branches = hysteresis_init_branches or ("", "")
    hysteresis_init_branch_neg, hysteresis_init_branch_pos = hysteresis_init_branches

    if hysteresis_init_branch_pos != "":
        raise NotImplementedError("Hysteresis preceding state functionality is only supported for negative electrode blends.")

    blended_electrode = tuple(
        len(phases_el) > 1 for phases_el in phases_by_electrode
    )

    if update_bounds:
        # Evaluate new lithiation bounds and store in the pybamm.ParameterValues object
        lithiation_bounds = compute_lithiation_bounds(parameter_values)
        xLi_vals = {}
        for electrode in ELECTRODES:
            bounds_el = lithiation_bounds[electrode.lower()]
            for str_bound, bound in zip(["minimum", "maximum"], bounds_el):
                xLi_vals[f"{electrode} electrode {str_bound} stoichiometry"] = bound

        parameter_values.update(
            xLi_vals,
            check_already_exists=False,
        )
    else:
        # Existing lithiation bounds are valid so just read them
        lithiation_bounds = _get_lithiation_bounds(parameter_values, blended_electrode=blended_electrode)

    # Positive electrode
    xLi_pos = calc_xLi_init(1 - SOC_init, lithiation_bounds["positive"])
    c0_vals_pos = {
        "Initial concentration in positive electrode [mol.m-3]": xLi_pos * parameter_values["Maximum concentration in positive electrode [mol.m-3]"],
    }

    # Negative electrode
    phases_neg, _ = phases_by_electrode

    if len(phases_neg) > 1:
        # Apply hysteresis setting to Si component only
        hysteresis_init_branch_neg = ("", hysteresis_init_branch_neg)

    if len(phases_neg) == 1:
        xLi_neg = calc_xLi_init(SOC_init, lithiation_bounds["negative"])
        c0_vals_neg = { "Initial concentration in negative electrode [mol.m-3]": xLi_neg * parameter_values["Maximum concentration in negative electrode [mol.m-3]"] }
    else:
        # Blended electrode
        Uneg_phases = [
            parameter_values[f"{phase}Negative electrode {hysteresis_branch}OCP [V]"]
            for phase, hysteresis_branch in zip(phases_neg, hysteresis_init_branch_neg)
        ]
        phiact_neg_phases = [parameter_values[f"{phase}Negative electrode active material volume fraction"] for phase in phases_neg]
        csat_neg_phases = [parameter_values[f"{phase}Maximum concentration in negative electrode [mol.m-3]"] for phase in phases_neg]
        
        # Compute lithiation proportions
        phiact_tot_neg = sum(phiact for phiact in phiact_neg_phases)
        cmax_neg_phases = [csat * phiact / phiact_tot_neg for csat, phiact in zip(csat_neg_phases, phiact_neg_phases)]
        ctot = sum(cmax_neg_phases)
        qprop_neg_phases = [(cmax / ctot) for cmax in cmax_neg_phases]

        xLi_neg_phases = calc_xLi_init(
            SOC_init,
            lithiation_bounds["negative"],
            ocp_mat=Uneg_phases,
            qprop_mat=qprop_neg_phases,
        )

        c0_vals_neg = {
            f"{phase}Initial concentration in negative electrode [mol.m-3]": xLi_neg_phase * parameter_values[f"{phase}Maximum concentration in negative electrode [mol.m-3]"]
            for xLi_neg_phase, phase in zip(xLi_neg_phases, phases_neg)
        }

    # Update initial concentrations
    c0_vals = (c0_vals_neg | c0_vals_pos)
    parameter_values.update(
        c0_vals,
        check_already_exists=False,
    )


def get_ocv_thermodynamic(parameter_values, num=201):
    """
    Get OCV data from pybamm.ParameterValues (SOC vs OCV) as num x 2 np.ndarray.

    IMPORTANT: not compatible with hysteresis or multi-phase electrode. Guarded in main function.

    Parameters
    ---
    parameter_values - pybamm.ParameterValues objecs
    num - number of points (passed to np.linspace, same format)
    branch - if supplied, 2-tuple of values for neg and pos electrode
    ---

    Return
    ---
    (num, 2) np.ndarray with columns for SOC, OCV
    ---
    """
    lithiation_bounds = _get_lithiation_bounds(parameter_values)

    dxLi_neg = lithiation_bounds["negative"][1] - lithiation_bounds["negative"][0]
    dxLi_pos = lithiation_bounds["positive"][1] - lithiation_bounds["positive"][0]

    soc = np.linspace(0, 1, num=num)
    xLi_neg = lithiation_bounds["negative"][0] + (soc * dxLi_neg)
    xLi_pos = lithiation_bounds["positive"][1] - (soc * dxLi_pos)
    
    ocv = (
        _eval_OCP(parameter_values["Positive electrode OCP [V]"], xLi_pos)
        - _eval_OCP(parameter_values["Negative electrode OCP [V]"], xLi_neg)
    )

    return np.column_stack((soc, ocv))


def _scale_ocv_soc_linear(ocv_soc_ref, ocv_soc_new, method):
    """
    Linear scaling of a reference OCV-SOC relationship to a match a new OCV-SOC relationship

    Parameters
    ---
    ocv_new - (num, 2) np.ndarray with columns for SOC, OCV (col0 for soc, col1 for ocv)
    ocv_ref - (num, 2) np.ndarray with columns for SOC, OCV (col0 for soc, col1 for ocv)
    method - str for the method to use for the conversion
        linear_endpoints : linear transformation using OCV-SOC end points,
        linear_optimized : linear transformation using optimized parameters)
    ---

    Return
    ---
    (a,b) - tuple of floats for the linear transformation parameters soc_new = a*soc_ref + b
    ---
    """
    if not _is_monotonic(ocv_soc_ref):
        raise ValueError("Reference OCV is not monotonically increasing")
    if not _is_monotonic(ocv_soc_new):
        raise ValueError("New OCV is not monotonically increasing")

    soc_new = ocv_soc_new[:, 0]
    soc_ref = ocv_soc_ref[:, 0]

    ocv_new = ocv_soc_new[:, 1]
    ocv_ref = ocv_soc_ref[:, 1]

    ocv_ref_max = ocv_ref.max()
    ocv_ref_min = ocv_ref.min()

    soc_new_max = np.interp(ocv_ref_max, ocv_new, soc_new)
    soc_new_min = np.interp(ocv_ref_min, ocv_new, soc_new)

    soc_ref_max = soc_ref.max()
    soc_ref_min = soc_ref.min()

    # Determine endpoint linear transformation parameters
    a = (soc_new_max - soc_new_min) / (soc_ref_max - soc_ref_min)
    b = soc_new_max - soc_ref_max * a

    if method == "linear_optimized":
        # Optimize endpoint solution
        initial_guess = [a, b]

        def objective(params, ocv_soc_new, ocv_soc_ref):
            """
            Minimisation objective function for the Broyden/Fletcher/Goldfarb/Shanno (BFGS) algorithm used by scipy.minimze
            """
            a, b = params
            soc_converted = a * ocv_soc_ref[:, 0] + b

            V_ocv_new = np.interp(soc_converted, ocv_soc_new[:, 0], ocv_soc_new[:, 1])
            V_ocv_ref = ocv_soc_ref[:, 1]

            return np.mean((V_ocv_new - V_ocv_ref) ** 2)

        # Least error between the two OCV relations in the shared SOC range
        result = minimize(objective, initial_guess, args=(ocv_soc_new, ocv_soc_ref))
        if not result.success:
            raise RuntimeError(f"OCV least squares regression failed: {result.message}")
        a = result.x[0]
        b = result.x[1]
    # else method == 'linear_endpoints', return existing endpoint solution

    return (a, b)


def convert_soc(soc_ref_value, ocv_soc_ref, ocv_soc_new, method="voltage"):
    """
    Convert a provided SOC defined within a reference OCV-SOC relation to a new SOC defined within a new OCV-SOC relation.

    Parameters
    ---
    soc_ref_value : float
        SOC value to convert.
    ocv_ref : (num, 2) np.ndarray with columns (SOC, OCV)
        Reference OCV-SOC relation
    ocv_new : (num, 2) np.ndarray with columns (SOC, OCV)
        New OCV-SOC relation
    method : str (optional, default="voltage")
        Method to use for the conversion:
            "voltage" - match initial voltage
            "linear_endpoints" - apply linear transformation using OCV-SOC endpoints
            "linear_optimized" - apply best-fit linear transformation to match OCV-SOC relations
    ---

    Return
    ---
    soc_converted - float for the converted SOC value in the new OCV-SOC relation.
    ---
    """
    METHODS_CONVERT_SOC = ["voltage", "linear_endpoints", "linear_optimized"]
    if method not in METHODS_CONVERT_SOC:
        raise ValueError(
            f"Invalid method: {method}. Allowed methods: {', '.join(METHODS_CONVERT_SOC)}"
        )

    if not _is_monotonic(ocv_soc_ref):
        raise ValueError("Reference OCV is not monotonically increasing")
    if not _is_monotonic(ocv_soc_new):
        raise ValueError("New OCV is not monotonically increasing")

    soc_new = ocv_soc_new[:, 0]
    soc_ref = ocv_soc_ref[:, 0]

    ocv_new = ocv_soc_new[:, 1]
    ocv_ref = ocv_soc_ref[:, 1]

    if soc_ref_value < soc_ref[0] or soc_ref_value > soc_ref[-1]:
        raise ValueError("SOC value is outside the range of the provided SOC-OCV data")

    if method == "voltage":
        ocv_interp = np.interp(soc_ref_value, soc_ref, ocv_ref)
        soc_converted = np.interp(ocv_interp, ocv_new, soc_new)
    else:
        # method == linear_endpoints or method == linear_optimized
        # Determine linear transformation parameters
        a, b = _scale_ocv_soc_linear(ocv_soc_ref, ocv_soc_new, method)
        soc_converted = a * soc_ref_value + b

    return soc_converted
