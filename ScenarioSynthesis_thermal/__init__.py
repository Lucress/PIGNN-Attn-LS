"""Scenario generation and a custom Newton--Raphson thermal solver."""

__all__ = [
    "ThermalCase",
    "ThermalNewtonResult",
    "generate_thermal_case",
    "solve_thermal_newton",
]


def __getattr__(name):
    """Load public classes lazily so ``python -m ...newton_raphson`` is clean."""

    if name in {"ThermalCase", "generate_thermal_case"}:
        from .case_generator import ThermalCase, generate_thermal_case

        value = {
            "ThermalCase": ThermalCase,
            "generate_thermal_case": generate_thermal_case,
        }[name]
    elif name in {"ThermalNewtonResult", "solve_thermal_newton"}:
        from .newton_raphson import ThermalNewtonResult, solve_thermal_newton

        value = {
            "ThermalNewtonResult": ThermalNewtonResult,
            "solve_thermal_newton": solve_thermal_newton,
        }[name]
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
