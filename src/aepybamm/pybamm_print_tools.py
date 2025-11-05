import pybamm

from pybamm.expression_tree.symbol import is_scalar_zero, is_matrix_zero

STANDARD_SUBSTS = [
    ("0.0002777777777777778", "(1/3600)"),
    ("0.016666666666666666", "(1/60)"),
    ("0.3333333333333333", "(1/3)"),
    (".0 ", " "),
    (".0)", ")"),
]

EXPANSION_TYPES = (
    pybamm.Concatenation,
    pybamm.BinaryOperator,
    pybamm.UnaryOperator,
    pybamm.Function,
)


def expansion(var, vardefs):
    """
    Expand (with a single layer of recursion) the string representation of a model symbol 'var' in terms of its immediate children.

    Uses model name dictionary 'vardefs' (output of build_name_dict()).
    """
    if isinstance(var, pybamm.BinaryOperator):
        return _expansion_binary(var, vardefs)
    elif isinstance(var, pybamm.UnaryOperator):
        return _expansion_unary(var, vardefs)
    elif isinstance(var, pybamm.Concatenation):
        # Express concatenatation as a quasi-function "concatenation(...)"
        return _expansion_functional(var, vardefs, name="concatenation")
    elif isinstance(var, pybamm.Function):
        # Convert "function ({name})" => "{name}"
        return _expansion_functional(var, vardefs, name=var.name[10:-1])
    else:
        raise TypeError(f"Unsupported type {type(var).__name__}")


def _expansion_functional(var, vardefs, name=None):
    """
    Expand (with a single layer of recursion) the string representation of a pybamm.Function in terms of its immediate children.

    Uses model name dictionary 'vardefs' (output of build_name_dict()).
    """
    name = name or var.name
    args = [to_str(child, vardefs) for child in var.children]
    out = name + f"({', '.join(args)})"
    return out


def _expansion_binary(var, vardefs):
    """
    Expand (with a single layer of recursion) the string representation of a pybamm.BinaryOperator in terms of its immediate children.

    Uses model name dictionary 'vardefs' (output of build_name_dict()).

    Incorporates code from pybamm.BinaryOperator.__str__().
    """
    left_str = to_str(var.left, vardefs)
    right_str = to_str(var.right, vardefs)

    if (
        isinstance(var.left, pybamm.BinaryOperator)
        and var.left.id not in vardefs
        and not (
            (var.left.name == var.name)
            or (var.left.name == "*" and var.name == "/")
            or (var.left.name == "+" and var.name == "-")
            or var.name == "+"
        )
    ):
        left_str = f"({left_str})"
    if (
        isinstance(var.right, pybamm.BinaryOperator)
        and var.right.id not in vardefs
        and not ((var.name == "*" and var.right.name in ["*", "/"]) or var.name == "+")
    ):
        right_str = f"({right_str})"

    return f"{left_str} {var.name} {right_str}"


def _expansion_unary(var, vardefs):
    """
    Expand (with a single layer of recursion) the string representation of a pybamm.UnaryOperator in terms of its immediate children.

    Uses model name dictionary 'vardefs' (output of build_name_dict()).
    """
    name = var.name
    if name.startswith(("* ", "/ ")) and ("integrated" in name):
        # Rewrite integrals in quasi-functional form "Integral{{domain of integration}}({integrand})"
        name = f"Integral{{{name[13:]}}}"
    out = name + f"({to_str(var.child, vardefs)})"
    return out


def to_str(var, vardefs, expand=False):
    """
    Return a string representation of a PyBaMM symbol 'var'.

    Uses model name dictionary 'vardefs' (output of build_name_dict()).

    If 'expand' is True, expands the string representation in terms of the symbol's children.
    If 'expand' is False, returns the symbol's descriptive name from 'vardefs', else the symbol's internal name.
    """    
    if not expand and var.id in vardefs:
        # Use the name description of the variable (key in model.variables) if present
        return vardefs[var.id]
    elif isinstance(var, EXPANSION_TYPES):
        return expansion(var, vardefs)
    else:
        return var.name


def build_name_dict(model):
    """
    Returns a name dictionary for a PyBaMM model, mapping variable unique identifiers to variable descriptive names.

    Variable descriptive names are the keys in the PyBaMM model.variables dict.
    """
    name_dict = {
        var.id: key
        for key, var in model.variables.items()
    }

    return name_dict


def as_string(model):
    """
    Returns a readable plain text summary of the equations and variable definitions of a PyBaMM model.
    """
    vardefs = build_name_dict(model)

    # Dependent variables for which a PDE / ODE / DAE is solved (entries in model.rhs or model.algebraic)
    eqn_vars = [
        key for key, var in model.variables.items()
        if var in model.rhs or var in model.algebraic
    ]

    # Variables which are identically zero. This can be because they are defined for an unused submodel, for example.
    zero_vars = [
        key for key, var in model.variables.items()
        if is_scalar_zero(var) or is_matrix_zero(var)
    ]

    # Variables which are evaluated non-zero expressions derived from dependent variables
    defined_vars = {
        key: var for key, var in model.variables.items()
        if key not in zero_vars
    }

    # Print equations
    s_out = "Equations\n---\n"

    for var, eqn in model.rhs.items():
        s_out += f"d({vardefs[var.id]})/dt = {to_str(eqn, vardefs)}\n"
    s_out += "\n"

    for var, eqn in model.algebraic.items():
        s_out += f"Solve for {vardefs[var.id]} such that : 0 = {to_str(eqn, vardefs)}\n"
    s_out += "\n"

    # Print variables
    # Dynamically identify variables that equate to parameters in the pybamm.ParameterValues object
    # Dynamically identify variables that are concatenations of other variables (that is, share their definition)
    s_out += "Variables\n---\n"
    parameter_vars = []
    concatenated_vars = {}
    for key, var in defined_vars.items():
        definition = to_str(var, vardefs, expand=True)
        if isinstance(var, pybamm.Concatenation):
            concatenated_vars |= {
                to_str(child, vardefs): to_str(var, vardefs)
                for child in var.children
            }
        elif definition != key:
            s_out += f"{key} = {definition}\n"
        elif (key not in concatenated_vars and key not in eqn_vars):
            parameter_vars.append(key)

    s_concat = '\n'.join([
        f"{src} => {dst}"
        for src, dst in concatenated_vars.items()
    ])
    s_parameter = '\n'.join(parameter_vars)
    s_zero_vars = '\n'.join(zero_vars)

    s_out += f"\nConcatenated Variables\n---\n{s_concat}\n"
    s_out += f"\nVariables Equal to Parameters\n---\n{s_parameter}\n"
    s_out += f"\nZero Variables\n---\n{s_zero_vars}"

    for old, new in STANDARD_SUBSTS:
        s_out = s_out.replace(old, new)

    return s_out
