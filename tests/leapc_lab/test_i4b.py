"""Tests for the i4b parametric MPC planner and OCP definition."""

import pytest

pytest.importorskip("acados_template", reason="requires a local acados build")
i4b = pytest.importorskip("i4b", reason="requires the i4b extra")

import casadi as ca  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import torch  # noqa: E402
from i4b.data.buildings.sfh_1919_1948 import sfh_1919_1948_0_soc  # noqa: E402
from i4b.data.buildings.sfh_2010_2015 import sfh_2010_2015_2_kfw  # noqa: E402
from i4b.data.buildings.sfh_2016_now import sfh_2016_now_0_soc  # noqa: E402
from i4b.models.model_buildings import Building  # noqa: E402
from i4b.models.model_hvac import Heatpump_AW  # noqa: E402

from leapc_lab.i4b.acados_ocp import (  # noqa: E402
    SUPPORTED_METHODS,
    calculate_discrete_dynamics,
    export_parametric_ocp,
)
from leapc_lab.i4b.planner import I4bPlanner, I4bPlannerConfig  # noqa: E402
from leapc_lab.i4b.utils import thermal_power_kw  # noqa: E402


def _building(params, mdot_hp=0.25, method="4R3C"):
    return Building(params=params, mdot_hp=mdot_hp, method=method)


def _observations(batch_size, dtype=torch.float64):
    return {
        "state": torch.full((batch_size, 3), 20.0, dtype=dtype),
        "disturbances": {
            "T_amb": torch.full((batch_size, 1), 5.0, dtype=dtype),
            "Qdot_gains": torch.zeros((batch_size, 1), dtype=dtype),
        },
        "setpoints": {
            "T_set_lower": torch.full((batch_size, 1), 20.0, dtype=dtype),
            "T_set_upper": torch.full((batch_size, 1), 26.0, dtype=dtype),
        },
        "forecast": {},
    }


def _fixed_discrete_dynamics(building, delta_t):
    """Independent reference: ZOH via A^-1 (Ad - I), valid for invertible A.

    `calculate_discrete_dynamics` uses the Van Loan block exponential instead, so
    the two only agree if both are right.
    """
    nx = len(building.state_keys)
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", 1)
    d = ca.SX.sym("d", 2)
    dynamics = building.calc_casadi(x, u[0], d)
    Ac = np.asarray(ca.evalf(ca.jacobian(dynamics, x)))
    Bc = np.asarray(ca.evalf(ca.jacobian(dynamics, u)))
    Ec = np.asarray(ca.evalf(ca.jacobian(dynamics, d)))
    Ad = scipy.linalg.expm(Ac * delta_t)
    integral = np.linalg.solve(Ac, Ad - np.eye(nx))
    return Ad, integral @ Bc, integral @ Ec


@pytest.mark.parametrize("method, nx", [("2R2C", 2), ("4R3C", 3), ("5R4C", 4)])
def test_discrete_dynamics_match_reference_for_every_method(method, nx):
    assert method in SUPPORTED_METHODS
    building = _building(sfh_1919_1948_0_soc, method=method)

    actual = calculate_discrete_dynamics(building, 900.0)
    expected = _fixed_discrete_dynamics(building, 900.0)

    assert [matrix.shape for matrix in actual] == [(nx, nx), (nx, 1), (nx, 2)]
    for actual_matrix, expected_matrix in zip(actual, expected):
        np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-9, atol=1e-12)


def test_discrete_dynamics_differ_between_houses():
    old_house = calculate_discrete_dynamics(_building(sfh_1919_1948_0_soc), 900.0)
    new_house = calculate_discrete_dynamics(_building(sfh_2010_2015_2_kfw), 900.0)

    assert any(not np.allclose(old, new) for old, new in zip(old_house, new_house))


def test_discrete_dynamics_rejects_method_without_casadi_export():
    # Guard: if 6R4C ever gets a CasADi export, pick another unsupported method.
    assert "6R4C" not in SUPPORTED_METHODS
    building = _building(sfh_1919_1948_0_soc, method="6R4C")

    with pytest.raises(ValueError, match="no CasADi export"):
        calculate_discrete_dynamics(building, 900.0)


def test_export_ocp_returns_space_and_defaults_for_runtime_matrices():
    Ad = np.arange(1.0, 10.0).reshape(3, 3)
    Bd = np.arange(11.0, 14.0).reshape(3, 1)
    Ed = np.arange(21.0, 27.0).reshape(3, 2)
    _, _, param_space, defaults = export_parametric_ocp(
        Ad, Bd, Ed, 0.25, n_horizon=2, hp_model=Heatpump_AW(mdot_HP=0.25)
    )

    assert list(defaults) == ["Ad", "Bd", "Ed", "mdot_hp"]
    np.testing.assert_array_equal(defaults["Ad"], Ad)
    np.testing.assert_array_equal(defaults["Bd"], Bd)
    np.testing.assert_array_equal(defaults["Ed"], Ed)
    np.testing.assert_array_equal(defaults["mdot_hp"], np.array([0.25]))
    for name, default in defaults.items():
        assert param_space[name].shape == default.shape
        assert np.all(param_space[name].low <= default)
        assert np.all(param_space[name].high >= default)


def test_export_ocp_rejects_mismatched_dynamics_shapes():
    with pytest.raises(ValueError, match="Dynamics require shapes"):
        export_parametric_ocp(
            np.eye(3),
            np.ones((3, 1)),
            np.ones((3, 3)),  # one disturbance column too many
            0.25,
            n_horizon=2,
            hp_model=Heatpump_AW(mdot_HP=0.25),
        )


def test_stagewise_parameter_order_and_gain_routing():
    _, manager, _, _ = export_parametric_ocp(
        np.eye(3),
        np.ones((3, 1)),
        np.ones((3, 2)),
        0.25,
        n_horizon=2,
        hp_model=Heatpump_AW(mdot_HP=0.25),
    )
    values = manager.combine_non_differentiable_parameters(
        batch_size=2,
        T_amb=np.full((2, 3, 1), 4.0),
        Qdot_gains=np.arange(6.0).reshape(2, 3, 1),
        T_set_lower=np.full((2, 3, 1), 19.0),
        T_set_upper=np.full((2, 3, 1), 25.0),
        grid_signal=np.full((2, 3, 1), 1.5),
    )

    assert values.shape == (2, 3, 5)
    np.testing.assert_array_equal(values[..., 0], 4.0)
    np.testing.assert_array_equal(values[..., 1], np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(values[..., 2], 19.0)
    np.testing.assert_array_equal(values[..., 3], 25.0)
    np.testing.assert_array_equal(values[..., 4], 1.5)


def test_ocp_dynamics_use_runtime_matrices_and_mass_flow():
    Ad = np.arange(1.0, 10.0).reshape(3, 3) / 10
    Bd = np.arange(11.0, 14.0).reshape(3, 1) / 10
    Ed = np.arange(21.0, 27.0).reshape(3, 2) / 100
    ocp, manager, _, _ = export_parametric_ocp(
        Ad, Bd, Ed, 0.25, n_horizon=2, hp_model=Heatpump_AW(mdot_HP=0.25)
    )
    evaluate = ca.Function(
        "evaluate",
        [ocp.model.x, ocp.model.u, ocp.model.p, ocp.model.p_global],
        [ocp.model.disc_dyn_expr, ocp.model.con_h_expr[2]],
    )

    x = np.array([18.0, 19.0, 17.0])
    u = np.array([35.0])
    stagewise = np.array([5.0, 300.0, 20.0, 26.0, 1.0])
    p_global = manager.differentiable_default_flat
    next_state, qth = evaluate(x, u, stagewise, p_global)

    np.testing.assert_allclose(
        np.asarray(next_state).reshape(-1),
        Ad @ x + Bd[:, 0] * u[0] + Ed @ stagewise[:2],
    )
    assert float(qth) == pytest.approx(thermal_power_kw(0.25, 35.0, 17.0))

    p_global[-1] = 0.3
    _, changed_qth = evaluate(x, u, stagewise, p_global)
    assert float(changed_qth) == pytest.approx(thermal_power_kw(0.3, 35.0, 17.0))


def test_planner_defaults_match_building_matrices():
    building = _building(sfh_1919_1948_0_soc)
    planner = I4bPlanner(
        cfg=I4bPlannerConfig(building_params=sfh_1919_1948_0_soc, n_horizon=2),
        building_model=building,
    )

    defaults = planner.default_param()
    ad, bd, ed = calculate_discrete_dynamics(building, 900.0)
    np.testing.assert_array_equal(defaults["Ad"], ad)
    np.testing.assert_array_equal(defaults["Bd"], bd)
    np.testing.assert_array_equal(defaults["Ed"], ed)


def test_batched_heterogeneous_solve_matches_separate_solves(tmp_path):
    # Two 4R3C houses: same state dimension (one compiled OCP), different dynamics
    # matrices. Mixing methods (different nx) in one batch is not supported.
    old_building = _building(sfh_1919_1948_0_soc)
    new_building = _building(sfh_2010_2015_2_kfw)
    hp = Heatpump_AW(mdot_HP=0.25)
    planner = I4bPlanner(
        cfg=I4bPlannerConfig(
            building_params=sfh_1919_1948_0_soc,
            n_horizon=4,
            n_batch_init=2,
            dtype=torch.float64,
        ),
        building_model=old_building,
        hp_model=hp,
        export_directory=tmp_path,
    )
    ad_old, bd_old, ed_old = calculate_discrete_dynamics(old_building, 900.0)
    ad_new, bd_new, ed_new = calculate_discrete_dynamics(new_building, 900.0)
    params = {
        name: torch.tensor(np.stack([old, new]), dtype=torch.float64)
        for name, old, new in [
            ("Ad", ad_old, ad_new),
            ("Bd", bd_old, bd_new),
            ("Ed", ed_old, ed_new),
        ]
    }
    params["mdot_hp"] = torch.full((2, 1), 0.25, dtype=torch.float64)

    batch_result = planner(_observations(2), params=params)
    separate_results = [
        planner(_observations(1), params={k: v[index : index + 1] for k, v in params.items()})
        for index in range(2)
    ]

    assert np.all(batch_result[0].status == 0)
    for output_index in range(1, 5):
        expected = torch.cat([result[output_index] for result in separate_results])
        torch.testing.assert_close(batch_result[output_index], expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(batch_result[2][0], batch_result[2][1])


def test_long_horizon_solve_well_insulated_house(tmp_path):
    """Regression: zero-initialized solves failed for well-insulated houses.

    "Well insulated" means low heat losses per floor area and hence a slow house:
    sfh_2016_now_0_soc loses about 1.3 W/(m^2 K) (H_tr + H_ve per floor area,
    4R3C) with a time constant of roughly 35 h, against 3.7 W/(m^2 K) and 19 h
    for sfh_1919_1948_0_soc. Cold-started zero iterates led to ACADOS_MINSTEP
    (status 4, u0 returned as 0) at the 24 h default horizon before
    I4bInitializer was added.
    """
    planner = I4bPlanner(
        cfg=I4bPlannerConfig(building_params=sfh_2016_now_0_soc),
        export_directory=tmp_path,
    )

    ctx, u0, *_ = planner(_observations(1))

    assert np.all(ctx.status == 0)
    assert 5.0 <= float(u0[0, 0]) <= 65.0


def test_forward_accepts_plain_inputs_and_uses_ocp_defaults(tmp_path):
    """Plain scalar/list obs normalize to the same solve as batched tensors.

    The minimal obs omits the setpoints (OCP defaults 20/26 °C) and forecast
    (constant channels), matching `_observations` exactly.
    """
    planner = I4bPlanner(
        cfg=I4bPlannerConfig(building_params=sfh_1919_1948_0_soc, n_horizon=4),
        export_directory=tmp_path,
    )

    plain = planner(
        {"state": [20.0, 20.0, 20.0], "disturbances": {"T_amb": 5.0, "Qdot_gains": 0.0}}
    )
    batched = planner(_observations(1))

    assert np.all(plain[0].status == 0)
    for output_index in range(1, 5):
        torch.testing.assert_close(plain[output_index], batched[output_index], rtol=1e-5, atol=1e-6)
