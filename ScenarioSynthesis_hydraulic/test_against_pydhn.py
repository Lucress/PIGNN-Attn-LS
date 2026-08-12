"""Regression checks against PyDHN's reference hydraulic solver."""

import numpy as np
from pydhn.solving import solve_hydraulics

from ScenarioSynthesis_hydraulic.case_generator import generate_hydraulic_case
from ScenarioSynthesis_hydraulic.newton_raphson import solve_hydraulics_newton


def test_custom_hydraulics_matches_pydhn() -> None:
    case = generate_hydraulic_case(seed=17)
    custom_net = case.net.copy()
    reference_net = case.net.copy()

    custom = solve_hydraulics_newton(
        custom_net,
        case.fluid,
        max_iters=100,
        error_threshold=1e-7,
        compute_hydrostatic=False,
        adaptive=False,
        verbose=0,
    )
    reference = solve_hydraulics(
        reference_net,
        case.fluid,
        max_iters=100,
        error_threshold=1e-7,
        compute_hydrostatic=False,
        adaptive=False,
        verbose=0,
    )

    assert custom.converged
    assert reference["history"]["hydraulics converged"]
    np.testing.assert_allclose(
        custom.mass_flow,
        reference["edges"]["mass_flow"][0],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        custom.delta_p,
        reference["edges"]["delta_p"][0],
        rtol=1e-8,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        custom.node_pressure,
        reference["nodes"]["pressure"][0],
        rtol=1e-10,
        atol=1e-5,
    )


def test_case_generation_is_reproducible() -> None:
    first = generate_hydraulic_case(seed=123)
    second = generate_hydraulic_case(seed=123)
    np.testing.assert_array_equal(first.pipe_lengths, second.pipe_lengths)
    np.testing.assert_array_equal(
        first.consumer_mass_flows, second.consumer_mass_flows
    )
