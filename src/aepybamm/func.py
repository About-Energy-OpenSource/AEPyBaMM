import pybamm


def _unflatten(coeffs_flat):
    """
    Unflatten a flat dictionary with key hierarchy indicated by '.' as a separator character.
    """

    coeffs = {}

    for k, v in coeffs_flat.items():
        base = coeffs
        kparts = k.split(".")

        while len(kparts) > 1:
            newk = kparts.pop(0)
            if newk not in base:
                base[newk] = {}
            base = base[newk]

        base[kparts[0]] = v

    return coeffs


def _make_generic_func_ce_T(func_type, coeffs):
    """
    Generate generic PyBaMM-compatible functions as used for Landesfeind 2019 electrolyte parameters.
    """
    if func_type not in ["crosspoly", "Landesfeind2019_cond", "Landesfeind2019_diff"]:
        raise NotImplementedError(f"Function type {func_type} is not supported.")

    if func_type == "crosspoly":
        def func(c_e, T):
            val = 0
            if "poly" in coeffs:
                for poly in coeffs["poly"].values():
                    val += poly["a"] * (c_e ** poly["m"]) * (T ** poly["n"])

            return val

    elif func_type == "Landesfeind2019_cond":
        def func(c_e, T):
            cM = c_e / 1000
            fT = pybamm.exp(1000 / T)

            a1 = coeffs["p1"] * (1 + T - coeffs["p2"]) * cM
            a2 = (
                1 + coeffs["p3"] * pybamm.sqrt(cM)
                  + coeffs["p4"] * cM * (1 + coeffs["p5"] * fT)
            )
            a3 = 1 + (cM ** 4) * coeffs["p6"] * fT

            return (a1 * a2 / a3)
    
    elif func_type == "Landesfeind2019_diff":
        def func(c_e, T):
            cM = c_e / 1000

            refval = coeffs["p1"] * pybamm.exp(coeffs["p2"] * cM)
            mul_Tdep = pybamm.exp(coeffs["p3"] / T) * pybamm.exp(coeffs["p4"] * cM / T)

            return (refval * mul_Tdep)

    return func


def _make_j0_func(coeffs_const, func_premul=None):
    """
    Generate PyBaMM-compatible exchange current density functions.

    j0 = mul * j0_ref * exp(-Ea/R * (1/T-1/Tref))

    coeffs_const - dict containing:
     - "j0_ref"
     - "Ea"
     - "Tref"

    func_premul - if None then mul = 1, else mul = func_premul(xLi)
    """
    def func(c_e, c_s_surf, c_s_max, T):
        xLi = c_s_surf / c_s_max

        if func_premul is not None:
            mul = func_premul(xLi)
        else:
            mul = 1
        
        arrhenius = pybamm.exp(coeffs_const["Ea"] / pybamm.constants.R * (1/coeffs_const["Tref"] - 1/T))
        
        return (mul * coeffs_const["j0_ref"] * arrhenius)

    return func


def _allow_unused_args_1d(func):
    def _func_extended(x, *args):
        return func(x)
    
    return _func_extended
