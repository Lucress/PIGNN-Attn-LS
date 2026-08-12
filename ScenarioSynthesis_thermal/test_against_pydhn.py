"""Regression checks against PyDHN's reference thermal solver."""

import numpy as np
from pydhn.solving import solve_thermal

from ScenarioSynthesis_thermal.case_generator import generate_thermal_case
from ScenarioSynthesis_thermal.newton_raphson import solve_thermal_newton


def test_custom_thermal_matches_pydhn() -> None:
    case = generate_thermal_case(seed=29)
    custom_net = case.net.copy()
    reference_net = case.net.copy()

    custom = solve_thermal_newton(
        custom_net,
        case.fluid,
        case.soil,
        max_iters=50,
        error_threshold=1e-10,
        adaptive=False,
        verbose=0,
    )
    reference = solve_thermal(
        reference_net,
        case.fluid,
        case.soil,
        max_iters=50,
        error_threshold=1e-10,
        adaptive=False,
        verbose=0,
    )

    assert custom.converged
    assert reference["history"]["thermal converged"]
    np.testing.assert_allclose(
        custom.node_temperature,
        reference["nodes"]["temperature"][0],
        rtol=1e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        custom.outlet_temperature,
        reference["edges"]["outlet_temperature"][0],
        rtol=1e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        custom.delta_q,
        reference["edges"]["delta_q"][0],
        rtol=1e-9,
        atol=1e-7,
    )


def test_case_generation_is_reproducible() -> None:
    first = generate_thermal_case(seed=321)
    second = generate_thermal_case(seed=321)
    np.testing.assert_array_equal(
        first.initial_node_temperature, second.initial_node_temperature
    )
    np.testing.assert_array_equal(
        first.consumer_heat_demand, second.consumer_heat_demand
    )
