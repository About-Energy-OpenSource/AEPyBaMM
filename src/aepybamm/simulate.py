import numpy as np
import pandas as pd
import pybamm


def solve_from_expdata(
    parameter_values,
    model,
    fp,
    dict_cols,
    Tamb_degC=25,
    use_exp_temp=False,
    dt_safe_drive_cycle=None,
    tmax=None,
    verbose=True,
):
    """
    Returns a pybamm.Solution object containing the simulated results for given experimental data.

    parameter_values : pybamm.ParameterValues
        Parameter values to use for the PyBaMM simulation. Use params.get_params() to get this.
    model : pybamm.lithium_ion.(model)
        Model to use for the PyBaMM simulation. Use params.get_params() to get this.
    fp : str
        File path for experimental data (comma-separated columns). The included current time series is used to load the PyBaMM simulation.
    dict_cols : dict
        Index into the experimental data. Dict with keys indicating data types and values indicating zero-based column numbers in comma-separated data.
        Expected data types include: "t" (time / s), "I" (current / A), "V" (voltage / V), "Q" (charge / Ah), "T" (temperature / K), "T_degC" (temperature / degC)
    Tamb_degC : float (optional)
        Ambient temperature (degC). Default == 25. Ignored if use_exp_temp == True.
    use_exp_temp : bool (optional)
        If True, use the included temperature time series to define the spatially isothermal cell temperature. Default: False.
    dt_safe_drive_cycle : float (optional)
        Enforce time steps with defined maximum duration to handle pulse-like interpolated loading conditions.
        If None, numerical time stepping has default (unconstrained) behaviour.
    tmax : float (optional)
        Experimental data are discarded at t > tmax, if specified. Default: None.
    verbose : bool
        Set True (default) to print additional information to stdout.
    """
    if verbose:
        print(f"Solving with experimental data from '{fp}'.")

    drive_cycle = pd.read_csv(
        fp,
        comment="#"
    ).to_numpy()

    # Offset time to zero, trim after stated tmax
    drive_cycle[:, dict_cols["t"]] -= drive_cycle[0, dict_cols["t"]]
    if tmax is not None:
        drive_cycle = drive_cycle[(drive_cycle[:, dict_cols["t"]] <= tmax), :]
    else:
        tmax = drive_cycle[-1, dict_cols["t"]]

    # Remove duplicate time instances and sort
    # Retain only first instance at each timestamp, in case of duplicates
    _, idx = np.unique(drive_cycle[:, dict_cols["t"]], return_index=True)
    n_removed = len(drive_cycle) - len(idx)
    drive_cycle = drive_cycle[idx, :]

    if verbose and n_removed > 0:
        print(f"Removed {n_removed} data point{'s' if n_removed > 1 else ''}"
               " with duplicate or negligibly differing time values.")

    # Create current interpolant
    current_interpolant = pybamm.Interpolant(
        drive_cycle[:, dict_cols["t"]],
        -drive_cycle[:, dict_cols["I"]], # PyBaMM treats discharge as positive
        pybamm.t
    )

    # Set drive cycle and update initial temperature
    parameter_values.update({
        "Current function [A]": current_interpolant,
        "Initial temperature [K]": 273.15 + Tamb_degC,
        "Ambient temperature [K]": 273.15 + Tamb_degC,
    })

    if use_exp_temp:
        # Check for a single matching experimental temperature data input
        exp_temp_keys = [k for k in ["T", "T_degC"] if k in dict_cols]
        if len(exp_temp_keys) != 1:
            raise ValueError("Controlled temperature from experiment ('use_exp_temp') "
                             "requires a unique specified experimental temperature time series "
                             "with 'dict_cols' key either 'T' (K) or 'T_degC' (degC).")
        exp_temp_key = exp_temp_keys[0]

        temp_drive_cycle = drive_cycle[:,[dict_cols["t"], dict_cols[exp_temp_key]]]
        if exp_temp_key == "T_degC":
            temp_drive_cycle[:,1] += 273.15

        if any([callable(parameter_values[param]) for param in ["Cation transference number", "Thermodynamic factor"]]):
            # Workaround for PyBaMM bug (https://github.com/pybamm-team/PyBaMM/issues/4670), unfixed at current release
            def func_temp_drive_cycle(y,z,t):
                dc = pybamm.Interpolant(
                    temp_drive_cycle[:,0],
                    temp_drive_cycle[:,1],
                    t,
                )

                return (dc + np.finfo(float).eps * y)

            temperature_interpolant = func_temp_drive_cycle
        else:
            # Expected approach without workaround
            temperature_interpolant = pybamm.Interpolant(
                temp_drive_cycle[:,0],
                temp_drive_cycle[:,1],
                pybamm.t,
            )

        parameter_values.update(
            {
                "Initial temperature [K]": temp_drive_cycle[0,1],
                "Ambient temperature [K]": temperature_interpolant,
            }
        )
    
    default_tolerances = {
        "atol": 1e-4,
        "rtol": 1e-6,
    }
    options = {}

    if dt_safe_drive_cycle is not None:
        options.update({"dt_max": dt_safe_drive_cycle})

    solver = pybamm.IDAKLUSolver(
        **default_tolerances,
        options=options,
    )

    solver_opts = {}
    if dt_safe_drive_cycle is not None:
        nsteps = int(np.ceil(tmax / dt_safe_drive_cycle))
        t_eval = np.linspace(drive_cycle[0, dict_cols["t"]], nsteps * dt_safe_drive_cycle, num=(nsteps + 1))
        t_eval[-1] = tmax
    else:
        t_eval = [drive_cycle[0, dict_cols["t"]], tmax]

    solver_opts = {
        "calc_esoh": False,
        "t_eval": t_eval,
        "t_interp": drive_cycle[:, dict_cols["t"]],
    }

    submesh_types = model.default_submesh_types
    domains_micro = ["positive particle", "negative particle"]
    num_pts_domain = 32
    var_pts = { k: num_pts_domain for k in ["x_n", "x_s", "x_p", "r_n", "r_p"]}

    num_domains_micro_electrodes = [float(x) for x in model.options["particle phases"]]
    for electrode, num_domains_micro in zip(["negative", "positive"], num_domains_micro_electrodes):
        if num_domains_micro > 1:
            domains_micro.extend([f"{electrode} {phase} particle" for phase in ["primary", "secondary"]])
            var_pts.update(
                { f"r_{electrode[0]}_{phase}": num_pts_domain for phase in ["prim", "sec"] }
            )

    for domain in domains_micro:
        submesh_types[domain] = pybamm.MeshGenerator(
            pybamm.Exponential1DSubMesh, submesh_params={"side": "right"}
        )

    # Simulation 
    sim = pybamm.Simulation(
        model, 
        parameter_values=parameter_values,
        solver=solver,
        submesh_types=submesh_types,
        var_pts=var_pts,
    )

    # Solve
    sol = sim.solve(**solver_opts)

    return sol
