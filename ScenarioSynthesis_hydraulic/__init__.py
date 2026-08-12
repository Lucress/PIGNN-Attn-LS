"""Scenario generation and a custom Newton--Raphson hydraulic solver."""

__all__ = [
    "HydraulicCase",
    "HydraulicNewtonResult",
    "generate_hydraulic_case",
    "solve_hydraulics_newton",
]


def __getattr__(name):
    """Load public classes lazily so ``python -m ...newton_raphson`` is clean."""

    if name in {"HydraulicCase", "generate_hydraulic_case"}:
        from .case_generator import HydraulicCase, generate_hydraulic_case

        value = {
            "HydraulicCase": HydraulicCase,
            "generate_hydraulic_case": generate_hydraulic_case,
        }[name]
    elif name in {"HydraulicNewtonResult", "solve_hydraulics_newton"}:
        from .newton_raphson import HydraulicNewtonResult, solve_hydraulics_newton

        value = {
            "HydraulicNewtonResult": HydraulicNewtonResult,
            "solve_hydraulics_newton": solve_hydraulics_newton,
        }[name]
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
