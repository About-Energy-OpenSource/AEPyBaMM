import bpx
from packaging.version import Version

BPX_VERSION_MINIMUM = Version("0.5.0")
BPX_VERSION_LATEST = Version("0.5.1")
BPX_VERSION_MINIMUM_UNSUPPORTED = Version("0.6.0")


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
    return bpx.parse_bpx_file(fp)


def is_multimaterial(electrode):
    return isinstance(electrode, (bpx.schema.ElectrodeBlended, bpx.schema.ElectrodeBlendedSPM))


def _get_material_names(parameter_set):
    electrodes = (
        parameter_set.parameterisation.negative_electrode,
        parameter_set.parameterisation.positive_electrode,
    )

    material_names = tuple(
        list(electrode.particle.keys()) if is_multimaterial(electrode)
        else ""
        for electrode in electrodes
    )

    return material_names
