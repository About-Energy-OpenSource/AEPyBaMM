import json

import bpx

from packaging.version import Version
import numbers

ELECTRODES = ["Negative", "Positive"]

BPX_VERSION_MINIMUM = Version("1.1.1")
BPX_VERSION_LATEST = Version("1.1.1")
BPX_VERSION_MINIMUM_UNSUPPORTED = Version("1.2.0")


def get_BPX_version():
    return Version(bpx.__version__)


def validate_BPX_version():
    BPX_version = get_BPX_version()
    if BPX_version < BPX_VERSION_MINIMUM:
        raise RuntimeError(
            f"aepybamm requires BPX {BPX_VERSION_MINIMUM} or later. Detected version: {BPX_version}."
        )
    if BPX_version >= BPX_VERSION_MINIMUM_UNSUPPORTED:
        raise RuntimeError(
            f"aepybamm requires BPX < {BPX_VERSION_MINIMUM_UNSUPPORTED}. Detected version: {BPX_version}."
        )
    if BPX_version > BPX_VERSION_LATEST:
        print(
            f"Warning: running with BPX {BPX_version}. Latest tested version is {BPX_VERSION_LATEST}."
            "Functionality is not guaranteed."
        )


def as_bpx(fp):
    params_obj = as_bpx_obj(fp)
    return bpx.parse_bpx_obj(params_obj)


def as_bpx_obj(fp):
    with open(fp) as src:
        params = json.load(src)

    if bpx.is_legacy_bpx(params):
        # Apply A:E specific hysteresis adjustments not covered by BPX or PyBaMM
        _migrate_ae_hysteresis(params)
        params = bpx.convert_v0_to_v1(params)

    return params


def is_multimaterial(electrode):
    return isinstance(electrode, (bpx.schema.ElectrodeBlended, bpx.schema.ElectrodeBlendedSPM))


def _get_material_names(parameter_set):
    def _get_particle_names(electrode):
        if isinstance(electrode, dict):
            return list(electrode["Particle"].keys()) if "Particle" in electrode else ""

        return list(electrode.particle.keys()) if is_multimaterial(electrode) else ""

    if isinstance(parameter_set, dict):
        parameterisation = parameter_set["Parameterisation"]
        if isinstance(parameterisation, dict):
            electrodes = (
                parameterisation["Negative electrode"],
                parameterisation["Positive electrode"],
            )
        else:
            electrodes = (
                parameterisation.negative_electrode,
                parameterisation.positive_electrode,
            )

        return tuple(_get_particle_names(electrode) for electrode in electrodes)

    electrodes = (
        parameter_set.parameterisation.negative_electrode,
        parameter_set.parameterisation.positive_electrode,
    )

    material_names = tuple(
        _get_particle_names(electrode)
        for electrode in electrodes
    )

    return material_names


def _migrate_ae_hysteresis(params):
    """
    Move legacy About:Energy hysteresis parameters in place from the "User-defined"
    section of a BPX v0.x parameter dictionary into the hysteresis fields defined by
    the BPX v1.x schema.

    The general v0.x -> v1.x conversion is handled by bpx itself in
    bpx.convert_v0_to_v1(). This function only covers the About:Energy hysteresis
    parameters, which bpx does not modify.
    """

    if "User-defined" not in params["Parameterisation"]:
        # No hysteresis parameters
        return

    # All parameters are required to specify a hysteresis model
    PARAMS_HYSTERESIS = {
        # A:E user-defined key / BPX v1.x key
        "electrode lithiation OCP [V]": "OCP (lithiation) [V]",
        "electrode delithiation OCP [V]": "OCP (delithiation) [V]",
        "particle hysteresis decay rate": "OCP hysteresis decay constant",
    }
    # Check user-defined properties matching the hysteresis parameters
    params_ud = params["Parameterisation"]["User-defined"]
    hysteresis_materials = set([
        k.replace(suffix, "").rstrip()
        for k in params_ud
        for suffix in PARAMS_HYSTERESIS
        if k.endswith(suffix)
    ])

    # A:E BPX JSON (v0.x) specifies a decay rate proportional to 1 - h, which takes a maximum value of 2.
    # For BPX >= 1.0, A:E standard aligns with BPX where decay rate is proportional to (1 - h) / 2,
    # which takes a maximum value of 1, and hence the corresponding decay rate constant is twice its value in v0.x.
    decay_rate_params = [k for k in params_ud if "hysteresis decay rate" in k]
    for param in decay_rate_params:
        decay_rate_multiplier = 2

        if not isinstance(params_ud[param], numbers.Number):
            # Support for functional hysteresis decay rates is deprecated in line with BPX v1.1.
            raise NotImplementedError(
                f"Functional hysteresis decay rates are no longer supported ('{param}'). "
                "BPX v1.1 requires a scalar 'OCP hysteresis decay constant'; "
                "use aepybamm <= 0.2.2 if a functional decay rate is required."
            )

        # Scalar value
        params_ud[param] *= decay_rate_multiplier

    for fullmat in hysteresis_materials:
        for param_AE in PARAMS_HYSTERESIS:
            key_AE = f"{fullmat} {param_AE}"
            if key_AE not in params_ud:
                raise ValueError("Incomplete hysteresis data.")

        if ":" in fullmat:
            # Multi-material
            mat, el = fullmat.split(": ", maxsplit=1)

            # Check material exists in electrode
            if (
                ("Particle" not in params["Parameterisation"][f"{el} electrode"]) or
                (mat not in params["Parameterisation"][f"{el} electrode"]["Particle"])
            ):
                raise ValueError("Material used in user-defined parameters but missing from electrode specifications.")

            params_mat = params["Parameterisation"][f"{el} electrode"]["Particle"][mat]
        else:
            # Single material
            params_mat = params["Parameterisation"][f"{fullmat} electrode"]

        for param_AE, param_bpx in PARAMS_HYSTERESIS.items():
            # Move parameters from "User-defined" into defined schema locations
            params_mat[param_bpx] = params_ud.pop(f"{fullmat} {param_AE}")
