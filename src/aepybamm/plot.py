import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare(sol, experiment, dict_cols, title=""):
    """
    Compute and plot simulation data against experiment.
    sol : pybamm.Solution
        Solution object containing simulated data.
    experiment : str
        File path for experimental data (comma-separated columns) against which to compare.
    dict_cols : dict
        Index into the experimental data. Dict with keys indicating data types and values indicating zero-based column numbers in comma-separated data.
        Expected data types include: "t" (time / s), "V" (voltage / V)
    title : str (optional)
        Text for validation plot title.
    """
    
    # Load experimental data
    data = pd.read_csv(experiment).to_numpy()

    # Assemble np.ndarrays for key variables
    voltage_data_exp = data[:, [dict_cols["t"], dict_cols["V"]]]
    voltage_data_sim = np.column_stack((sol["Time [s]"].entries, sol["Terminal voltage [V]"].entries))
    current_data_sim = np.column_stack((sol["Time [s]"].entries, -sol["Current [A]"].entries))
    temp_data_sim = np.column_stack((sol["Time [s]"].entries, sol["Volume-averaged cell temperature [K]"].entries - 273.15))
    dV_sim_exp = np.column_stack((voltage_data_exp[:, 0], sol["Terminal voltage [V]"](t=voltage_data_exp[:, 0]) - voltage_data_exp[:, 1]))
    
    # Plot
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0][0].plot(current_data_sim[:, 0], current_data_sim[:, 1], 'k')
    ax[0][1].plot(temp_data_sim[:, 0], temp_data_sim[:, 1], 'r')
    ax[1][0].plot(voltage_data_sim[:, 0], voltage_data_sim[:, 1], "-", label="Model")
    ax[1][0].plot(voltage_data_exp[:, 0], voltage_data_exp[:, 1], "--", label="Experiment")
    ax[1][1].plot(dV_sim_exp[:, 0], dV_sim_exp[:, 1] * 1000)

    # Evaluate errors with midpoint time-averaged RMSE
    idx_notnan = ~np.isnan(dV_sim_exp[:, 1])
    times = dV_sim_exp[idx_notnan, 0]
    weights = (
        np.diff(times, prepend=times[0]) +
        np.diff(times, append=times[-1])
    )

    rmse = np.sqrt(
        np.average(
            dV_sim_exp[idx_notnan, 1] ** 2,
            weights=weights,
        )
    ) * 1000

    ax[1][0].legend()
    ax[1][1].text(0.8, 0.2, f"RMSE: {rmse:.2f} mV",
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax[1][1].transAxes,
    )

    for axis in ax.flatten():
        axis.set_xlabel("Time [s]")
        axis.grid()

    ax[0][0].set_ylabel("Cell current [A]")
    ax[0][1].set_ylabel("Cell temperature [degC]")
    ax[1][0].set_ylabel("Voltage [V]")
    ax[1][1].set_ylabel("Error [mV]")

    plt.suptitle(title)
