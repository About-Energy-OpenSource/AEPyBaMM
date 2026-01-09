import numpy as np
from scipy.optimize import fsolve, minimize

from .pybamm_tools import (
    ELECTRODES,
    _eval_OCP,
    get_PyBaMM_version,
)

VALID_HYSTERESIS_BRANCHES = ["average", "charge", "discharge"]
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
HYSTERESIS_INIT_STATE_VALS = {
    "": 0,
    "delithiation": 1,
    "lithiation": -1,
}


def _validate_in_list(val, allowed, description):
    if val not in allowed:
        raise ValueError(
            f"Unsupported {description} '{val}'. "
            f"Supported {description}s are: {', '.join(allowed)}"
        )


def _map_hysteresis_init_state(hysteresis_preceding_state, hysteresis_model_el, el):
    branch = ""
    if any(hysteresis_model_mat != "single" for hysteresis_model_mat in hysteresis_model_el) and hysteresis_preceding_state != "average":
        branch = HYSTERESIS_BRANCH_MAP[hysteresis_preceding_state][el]

    if len(branch) > 0:
        # Add extra space for insertion in parameter key
        branch += " "

    return branch


def _get_hysteresis_branch_electrode(use_hysteresis, branch_cell, by_material=False):
    if by_material:
        return tuple([
            [
                _map_hysteresis_init_state(branch_cell, [use_hysteresis_mat], el)
                for use_hysteresis_mat in use_hysteresis_el
            ]
            for use_hysteresis_el, el in zip(use_hysteresis, ELECTRODES)
        ])
    else:
        return tuple([
            _map_hysteresis_init_state(branch_cell, use_hysteresis_el, el)
            for use_hysteresis_el, el in zip(use_hysteresis, ELECTRODES)
        ])


def _get_null_use_hysteresis(phases_by_electrode):
    return tuple([
        ["single" for phase in phases]
        for electrode, phases in zip(ELECTRODES, phases_by_electrode)
    ])


def _is_monotonic(data):
    """
    Return True if a 1D function y(x) expressed in num x 2 np.ndarray is monotonic, else False.
    """
    data = data[data[:, 0].argsort()]
    return np.all(np.diff(data[:, 1]) > 0)


def _get_lithiation_bounds(parameter_values, phases_by_electrode=None):
    """
    Get existing specification of lithiation bounds as:
    
    - a 2-tuple (neg, pos)
      - of lists (materials by electrode)
        - of 2-tuples (min, max lithiation bounds by material)

    Inputs:

    parameter_values - pybamm.ParameterValues
    phases_by_electrode (optional) - 2-tuple (neg, pos) of lists of phase prefixes
    """
    phases_by_electrode = phases_by_electrode or ([""], [""])

    def _get_bounds(parameter_values, electrode, phase):
        try:
            bounds = tuple([
                parameter_values[f"{phase}{electrode} electrode {bound} stoichiometry"]
                for bound in ["minimum", "maximum"]
            ])
        except KeyError:
            raise ValueError(
                f"Missing stoichiometry specification for '{phase}{electrode} electrode'"
            )
        
        return bounds

    lithiation_bounds = tuple([
        [
            _get_bounds(parameter_values, electrode, phase)
            for phase in phases
        ]
        for electrode, phases in zip(ELECTRODES, phases_by_electrode)
    ])

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
    lithiation_bounds_mat : list of 2-tuples (min, max) for materials in electrode
    ocp_mat : list of OCP functions for each material (used only if lithiation_bounds_mat is list of len() > 1)
    qprop_mat: list of contributing charge capacity proportions for each material (used only if lithiation_bounds_mat is list of len() > 1)
    ---

    Return
    ---
    xLi_mat - list of xLi at SOC, or scalar if length 1
    ---
    """
    if not isinstance(lithiation_bounds_mat, list):
        raise TypeError("'lithiation_bounds_mat' must be list of tuples.")

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
            
        # Try a sequence of increasingly diverse starting guesses
        for guess in (0.5, 0.01, 0.001, 0.0001):
            try:
                x0 = guess * np.ones(nmat + 1)
                x1 = _fsolve_safe(func, x0)
                return x1[:-1]
            except RuntimeError:
                continue
        raise RuntimeError("Failed to find initial lithiation extents")




def calc_lithium_inventory(parameter_values, phases_by_electrode):
    phases_neg, _ = phases_by_electrode
    # Add initial concentrations for multi-phase electrodes
    if len(phases_neg) > 1:
        add_initial_concentrations(parameter_values, phases_by_electrode)

    ncyc_ref = 0    
    for el, phases in zip(ELECTRODES, phases_by_electrode):
        for phase in phases:
            ncyc_ref += (
                parameter_values[f"{phase}Initial concentration in {el.lower()} electrode [mol.m-3]"]
                * parameter_values[f"{phase}{el} electrode active material volume fraction"]
                * parameter_values[f"{el} electrode thickness [m]"]
            )
            
    return ncyc_ref


def compute_lithiation_bounds(parameter_values, phases_by_electrode, use_hysteresis=None):
    """
    Evaluate lithiation bounds as:
    
    - a 2-tuple (neg, pos)
      - of lists (materials by electrode)
        - of 2-tuples (min, max lithiation bounds by material)
    """
    if "AE: Total cyclable lithium inventory [mol.m-2]" not in parameter_values:
        raise ValueError("Cyclable lithium inventory must be computed before calling.")

    use_hysteresis = use_hysteresis or _get_null_use_hysteresis(phases_by_electrode)

    ncyc = parameter_values["AE: Total cyclable lithium inventory [mol.m-2]"]
    phases_neg, _ = phases_by_electrode

    nsat_neg = [
        parameter_values[f"{phase}Maximum concentration in negative electrode [mol.m-3]"]
        * parameter_values[f"{phase}Negative electrode active material volume fraction"]
        * parameter_values["Negative electrode thickness [m]"]
        for phase in phases_neg
    ]
    hysteresis_branches_neg = {
        branch_cell: _get_hysteresis_branch_electrode(
            use_hysteresis,
            branch_cell,
            by_material=True,
        )[0] for branch_cell in HYSTERESIS_BRANCH_MAP
    }
    ocp_neg = {
        branch_cell: [
            parameter_values[f"{phase}Negative electrode {branch}OCP [V]"]
            for phase, branch in zip(phases_neg, hysteresis_branches_neg[branch_cell])
        ]
        for branch_cell in HYSTERESIS_BRANCH_MAP
    }

    nsat_pos = (
        parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        * parameter_values["Positive electrode active material volume fraction"]
        * parameter_values["Positive electrode thickness [m]"]
    )
    ocp_pos = parameter_values["Positive electrode OCP [V]"]
    Veod = parameter_values["Lower voltage cut-off [V]"]
    Veoc = parameter_values["Upper voltage cut-off [V]"]

    def balance(bounds, Vcell, hysteresis_branch):
        # Unknowns:
        # n lithiation extents for n-material negative electrode
        # 1 lithiation extent for single material positive electrode
        # 1 unknown negative electrode potential
        xLi_neg = bounds[:-2]
        xLi_pos = bounds[-2]
        Uneg = bounds[-1]

        ntot_neg = sum(
            xLi_phase * nsat_phase
            for xLi_phase, nsat_phase in zip(xLi_neg, nsat_neg)
        )
        ntot_pos = xLi_pos * nsat_pos

        residual = [
            # Equilibration of all negative electrode materials
            (_eval_OCP(ocp, xLi) - Uneg)
            for ocp, xLi in zip(ocp_neg[hysteresis_branch], xLi_neg)
        ] + [
            # Lithium inventory balance
            ntot_neg + ntot_pos - ncyc,
            # Thermodynamic voltage
            _eval_OCP(ocp_pos, xLi_pos) - Uneg - Vcell,
        ]
        return residual

    def _fsolve_try_bound_init(func, guess_iter, *args, **kwargs):
        for x0 in guess_iter():
            try:
                return _fsolve_safe(func, x0, *args, **kwargs)
            except RuntimeError as err:
                continue
        raise RuntimeError("Failed to find initial bounds")

    def _lower_bound_init_iter():
        for g_low in (0.1, 0.01, 1e-3, 1e-4):
            yield [g_low] * len(phases_neg) + [0.9, _eval_OCP(ocp_neg["discharge"][0], g_low)]

    def _upper_bount_init_iter():
        for g_upper in (0.1, 0.2, 0.3, 0.4, 0.5):
            yield [0.9] * len(phases_neg) + [g_upper, _eval_OCP(ocp_neg["charge"][0], 0.9)]
    
    lower_bounds = _fsolve_try_bound_init(balance, _lower_bound_init_iter, args=(Veod, "discharge"))
    upper_bounds = _fsolve_try_bound_init(balance, _upper_bount_init_iter, args=(Veoc, "charge"))

    bounds_all = [tuple(sorted(bounds)) for bounds in zip(lower_bounds, upper_bounds)]
    bounds_neg = bounds_all[:-2]
    bounds_pos = bounds_all[-2:-1]
    
    return bounds_neg, bounds_pos


def add_initial_concentrations(
    parameter_values,
    phases_by_electrode,
    use_hysteresis=None,
    hysteresis_preceding_branches=None,
    hysteresis_initial_branches=None,
    SOC_init=1,
    update_bounds=False,
):
    """
    Add initial concentrations to pybamm.ParameterValues.

    Note: currently ONLY supports negative electrode blends (guarded in main function) - so cheats by assuming that the positive electrode is not blended.

    update_bounds - set True to force recomputation of lithiation bounds. IMPORTANT: not compatible with hysteresis or single-phase electrode. Guarded in main function.
    """
    PyBaMM_version = get_PyBaMM_version()

    use_hysteresis = use_hysteresis or _get_null_use_hysteresis(phases_by_electrode)

    phases_neg, _ = phases_by_electrode
    hysteresis_preceding_branches = hysteresis_preceding_branches or ("", "")
    hysteresis_preceding_branch_neg, hysteresis_preceding_branch_pos = hysteresis_preceding_branches
    hysteresis_initial_branches = hysteresis_initial_branches or ("", "")

    hysteresis_initial_branches = tuple([
        [hysteresis_initial_branch_el] * len(phases_el)
        for phases_el, hysteresis_initial_branch_el in zip(phases_by_electrode, hysteresis_initial_branches)
    ])

    hysteresis_initial_branch_neg, hysteresis_initial_branch_pos = hysteresis_initial_branches

    if hysteresis_preceding_branch_pos != "":
        raise NotImplementedError("Hysteresis preceding state functionality is only supported for negative electrode blends.")

    if update_bounds:
        # Evaluate new lithiation bounds and store in the pybamm.ParameterValues object
        lithiation_bounds = compute_lithiation_bounds(parameter_values, phases_by_electrode, use_hysteresis=use_hysteresis)

        xLi_vals = {
            f"{phase}{electrode} electrode {bound} stoichiometry": val
            for electrode, phases, bounds_el in zip(ELECTRODES, phases_by_electrode, lithiation_bounds)
            for phase, bounds_mat in zip(phases, bounds_el)
            for bound, val in zip(["minimum", "maximum"], bounds_mat)           
        }

        parameter_values.update(
            xLi_vals,
            check_already_exists=False,
        )
    else:
        # Existing lithiation bounds are valid so just read them
        lithiation_bounds = _get_lithiation_bounds(parameter_values, phases_by_electrode=phases_by_electrode)

    # Positive electrode
    lithiation_bounds_neg, lithiation_bounds_pos = lithiation_bounds
    xLi_pos = calc_xLi_init(1 - SOC_init, lithiation_bounds_pos)
    c0_vals_pos = {
        "Initial concentration in positive electrode [mol.m-3]": xLi_pos * parameter_values["Maximum concentration in positive electrode [mol.m-3]"],
    }


    # Update initial hysteresis state for selected hysteresis branch
    if len(phases_neg) > 1:
        hysteresis_initial_branch_neg[0] = ""

    hysteresis_initial_branches = (hysteresis_initial_branch_neg, hysteresis_initial_branch_pos)

    for electrode, phases, use_hysteresis_electrode, hysteresis_initial_branches_el in zip(ELECTRODES, phases_by_electrode, use_hysteresis, hysteresis_initial_branches):
        for phase, use_hysteresis_phase, hysteresis_initial_branch_phase in zip(phases, use_hysteresis_electrode, hysteresis_initial_branches_el):
            if use_hysteresis_phase == "one-state hysteresis":
                key_init_hysteresis_state = f"{phase}Initial hysteresis state in {electrode.lower()} electrode"
                parameter_values.update(
                    {key_init_hysteresis_state: HYSTERESIS_INIT_STATE_VALS[hysteresis_initial_branch_phase.rstrip()]},
                    check_already_exists=False,
                )

    if len(phases_neg) == 1:
        xLi_neg = calc_xLi_init(SOC_init, lithiation_bounds_neg)
        c0_vals_neg = { "Initial concentration in negative electrode [mol.m-3]": xLi_neg * parameter_values["Maximum concentration in negative electrode [mol.m-3]"] }
    else:
        hysteresis_preceding_branch_neg = ["", hysteresis_preceding_branch_neg]
        
        # Blended electrode
        Uneg_phases = [
            parameter_values[f"{phase}Negative electrode {hysteresis_branch}OCP [V]"]
            for phase, hysteresis_branch in zip(phases_neg, hysteresis_preceding_branch_neg)
        ]
        qprop_neg_phases = _get_qprop_phases(parameter_values, "Negative", phases_neg)

        xLi_neg_phases = calc_xLi_init(
            SOC_init,
            lithiation_bounds_neg,
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

def get_ocv_thermodynamic(parameter_values, phases_by_electrode, use_hysteresis=None, branch="average", num=201):
    """
    Get OCV data from pybamm.ParameterValues (SOC vs OCV) as num x 2 np.ndarray.

    Parameters
    ---
    parameter_values - pybamm.ParameterValues object
    phases_by_electrode - tuple of list of phases (output of _get_phases_by_electrode)
    branch - if supplied, hysteresis branch on which OCV is returned:
             "average" (default), "charge" or "discharge"
    num - number of points (passed to np.linspace, same format)
    ---

    Return
    ---
    (num, 2) np.ndarray with columns for SOC, OCV
    ---
    """
    _validate_in_list(branch, VALID_HYSTERESIS_BRANCHES, "hysteresis branch")
    use_hysteresis = use_hysteresis or _get_null_use_hysteresis(phases_by_electrode)

    soc = np.linspace(0, 1, num=num)

    lithiation_bounds_neg, lithiation_bounds_pos = _get_lithiation_bounds(
        parameter_values,
        phases_by_electrode=phases_by_electrode,
    )
    hysteresis_init_branch_neg, hysteresis_init_branch_pos = (
        _get_hysteresis_branch_electrode(use_hysteresis, branch_cell=branch)
    )
    
    phases_neg, _ = phases_by_electrode
    if len(phases_neg) > 1:
        # Evaluate from primary phase for which no hysteresis by definition
        # Place hysteresis on Si material
        phase_neg = phases_neg[0]
        branch_mat_neg = ["", hysteresis_init_branch_neg]
        branch_neg = branch_mat_neg[0]
    else:
        phase_neg = ""
        branch_neg = hysteresis_init_branch_neg

    branch_pos = hysteresis_init_branch_pos

    # Positive electrode OCP lookup
    Upos = parameter_values[f"Positive electrode {branch_pos}OCP [V]"]
    dxLi_pos = lithiation_bounds_pos[0][1] - lithiation_bounds_pos[0][0]
    xLi_pos = lithiation_bounds_pos[0][1] - (soc * dxLi_pos)

    # Negative electrode OCP lookup
    Uneg = parameter_values[f"{phase_neg}Negative electrode {branch_neg}OCP [V]"]
    if len(phases_neg) > 1:
        # Evaluate xLi_neg for primary phase only
        xLi_neg = []
        for soc_inst in soc:
            try:
                xLi_neg.append(
                    calc_xLi_init(
                        soc_inst,
                        lithiation_bounds_neg,
                        ocp_mat=[
                            parameter_values[f"{phase}Negative electrode {branch_mat}OCP [V]"]
                            for phase, branch_mat in zip(phases_neg, branch_mat_neg)
                        ],
                        qprop_mat=_get_qprop_phases(parameter_values, "Negative", phases_neg),
                    )[0]
                )
            except RuntimeError:
                xLi_neg.append(np.nan) 
        xLi_neg = np.array(xLi_neg)

        # Interpolate or fill NaNs due to fsolve convergence errors         
        if np.isnan(xLi_neg).any():
            valid_mask = ~np.isnan(xLi_neg)
            if np.sum(valid_mask) >= 2:
                xLi_neg = np.interp(soc, soc[valid_mask], xLi_neg[valid_mask])
            else:
                raise ValueError("Not enough valid data points to interpolate")

    else:
        dxLi_neg = lithiation_bounds_neg[0][1] - lithiation_bounds_neg[0][0]
        xLi_neg = lithiation_bounds_neg[0][0] + soc * dxLi_neg

    ocv = _eval_OCP(Upos, xLi_pos) - _eval_OCP(Uneg, xLi_neg)
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


def convert_ocv_to_soc(OCV_init, parameter_values, phases_by_electrode, use_hysteresis=None, branch="average"):
    """
    Convert an OCV value to the SOC value for a corresponding branch.

    Parameters
    ---
    parameter_values - pybamm.ParameterValues object
    phases_by_electrode - tuple of list of phases (output of _get_phases_by_electrode)
    use_hysteresis - tuple of list of hysteresis models
    branch - if supplied, hysteresis branch on which OCV is returned:
             "average" (default), "charge" or "discharge"
    ---

    Return
    ---
    SOC_init - SOC value corresponding to specified OCV.
    ---
    """
    _validate_in_list(branch, VALID_HYSTERESIS_BRANCHES, "hysteresis branch")
    use_hysteresis = use_hysteresis or _get_null_use_hysteresis(phases_by_electrode)

    lithiation_bounds_neg, lithiation_bounds_pos = _get_lithiation_bounds(
        parameter_values,
        phases_by_electrode=phases_by_electrode,
    )
    hysteresis_init_branch_neg, hysteresis_init_branch_pos = (
        _get_hysteresis_branch_electrode(use_hysteresis, branch_cell=branch)
    )
    
    phases_neg, _ = phases_by_electrode
    nmat = len(phases_neg)
        
    if nmat > 1:
        # Apply hysteresis to Si component only
        hysteresis_init_branch_neg = ['', hysteresis_init_branch_neg]
    else:
        hysteresis_init_branch_neg = [hysteresis_init_branch_neg]

    dxLi_neg = [
        bounds_mat[1] - bounds_mat[0]
        for bounds_mat in lithiation_bounds_neg
    ]

    # Positive electrode OCP lookup
    Upos = parameter_values[f"Positive electrode {hysteresis_init_branch_pos}OCP [V]"]
    dxLi_pos = lithiation_bounds_pos[0][1] - lithiation_bounds_pos[0][0]

    # Negative electrode OCP lookup
    Uneg = [
        parameter_values[f"{phase}Negative electrode {branch_mat}OCP [V]"]
        for phase, branch_mat in zip(phases_neg, hysteresis_init_branch_neg)
    ]
    qprop_neg = _get_qprop_phases(parameter_values, "Negative", phases_neg)

    def func(x1):
        # Unknowns: xLi_neg_components, soc, Uneg
        xLi_neg = x1[:-2]
        soc = x1[-2]
        val_Uneg = x1[-1]

        # Equality of OCPs
        residual = [
            _eval_OCP(ocp, xLi) - val_Uneg
            for ocp, xLi in zip(Uneg, xLi_neg)
        ]

        # Total negative electrode contents sum to expected value
        residual_mat_constraint = sum([
            qprop * (xLi - bounds_mat[0]) / dxLi
            for qprop, xLi, bounds_mat, dxLi in zip(qprop_neg, xLi_neg, lithiation_bounds_neg, dxLi_neg)
        ]) - soc

        residual.append(residual_mat_constraint)

        # OCV evaluates to expected value
        xLi_pos = lithiation_bounds_pos[0][1] - soc * dxLi_pos
        val_Upos = _eval_OCP(Upos, xLi_pos)
        residual_ocv_constraint = val_Upos - val_Uneg - OCV_init

        residual.append(residual_ocv_constraint)

        return residual

    try:
        x0 = 0.5 * np.ones(nmat + 2)
        x1 = _fsolve_safe(func, x0)
    except RuntimeError:
        # Try with a different initial guess
        x0 = 0.1 * np.ones(nmat + 2)
        x1 = _fsolve_safe(func, x0)

    soc = x1[-2]
    return soc


def _get_qprop_phases(parameter_values, el, phases_el):
    _validate_in_list(el, ELECTRODES, "electrode")

    phiact_phases = [
        parameter_values[f"{phase}{el} electrode active material volume fraction"]
        for phase in phases_el
    ]
    csat_phases = [
        parameter_values[f"{phase}Maximum concentration in {el.lower()} electrode [mol.m-3]"]
        for phase in phases_el
    ]

    # Compute lithiation proportions
    phiact_tot = sum(phiact_phases)
    cmax_phases = [csat * phiact / phiact_tot for csat, phiact in zip(csat_phases, phiact_phases)]
    ctot = sum(cmax_phases)
    qprop_phases = [(cmax / ctot) for cmax in cmax_phases]

    return qprop_phases
