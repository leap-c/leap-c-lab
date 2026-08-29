"""Parametric acados OCP for i4b building heat-pump control."""

from collections import OrderedDict
from copy import deepcopy

import casadi as ca
import gymnasium as gym
import numpy as np
import scipy.linalg
from acados_template import ACADOS_INFTY, AcadosOcp, AcadosOcpFlattenedIterate
from i4b.constants import C_WATER_SPEC
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal
from leap_c.diff_mpc.data import AcadosOcpSolverInput
from leap_c.diff_mpc.initializer import AcadosDiffMpcInitializer
from leap_c.parameters import AcadosParameterManager

SUPPORTED_METHODS = ("2R2C", "4R3C", "5R4C")
"""Building methods the OCP supports (those with a ``calc_*_casadi`` implementation)."""


def calculate_discrete_dynamics(
    building_model: Building, delta_t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the linear building dynamics.

    Args:
        building_model: Instantiated i4b ``Building`` model.
        delta_t: Sampling time [s].

    Returns:
        ``(Ad, Bd, Ed)`` with shapes (nx, nx), (nx, 1), (nx, 2) such that
        ``x_next = Ad @ x + Bd @ u + Ed @ d`` with ``d = (T_amb, Qdot_gains)``.
    """
    if building_model.method not in SUPPORTED_METHODS:
        raise ValueError(f"Building method {building_model.method!r} has no CasADi export")

    nx = len(building_model.state_keys)
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", 1)
    d = ca.SX.sym("d", 2)
    f_expl_expr = building_model.calc_casadi(x, u[0], d)

    Ac = np.asarray(ca.evalf(ca.jacobian(f_expl_expr, x)))
    Bc = np.asarray(ca.evalf(ca.jacobian(f_expl_expr, u)))
    Ec = np.asarray(ca.evalf(ca.jacobian(f_expl_expr, d)))

    Ad = scipy.linalg.expm(Ac * delta_t)
    integral = np.linalg.solve(Ac, Ad - np.eye(nx))
    return Ad, integral @ Bc, integral @ Ec


def _cop_casadi(hp_model: Heatpump, T_HP: ca.SX, T_amb: ca.SX) -> ca.SX:
    """COP as a CasADi expression, matching the HP model's .COP() method."""
    if isinstance(hp_model, Heatpump_AW):
        # Pure polynomial arithmetic, works on SX symbols directly.
        return hp_model.COP(T_HP, T_amb)
    elif isinstance(hp_model, Heatpump_Vitocal):
        # TODO: delegate to hp_model.COP as above, once i4b fixes the interface:
        # upstream COP() lazily imports pvlib via i4b.disturbances and uses
        # np.tanh for the ground temperature, so we replicate it symbolically.
        T_source = 6.645 * ca.tanh(0.188 * (T_amb - 9.177)) + 7.872
        z0, a, b, c, d, f = 10.893436, -0.228602, 0.266006, 0.001461, 0.000501, -0.003546
        return z0 + a * T_HP + b * T_source + c * T_HP**2 + d * T_source**2 + f * T_HP * T_source
    else:
        raise TypeError(f"Unsupported HP model type: {type(hp_model).__name__}")


def export_parametric_ocp(
    ad: np.ndarray,
    bd: np.ndarray,
    ed: np.ndarray,
    mdot_hp: float,
    n_horizon: int,
    hp_model: Heatpump,
    name: str = "i4b",
    ws: float = 1.0,
    delta_t: float = 900.0,
) -> tuple[AcadosOcp, AcadosParameterManager, gym.spaces.Dict, dict[str, np.ndarray]]:
    """Export the i4b MPC as a parametric acados OCP.

    Building dynamics enter as differentiable runtime matrices
    (``Ad``, ``Bd``, ``Ed``, ``mdot_hp``), so one compiled OCP serves all houses
    of the same building method. Cost and constraints only use the room
    temperature (``x[0]``) and the HP return temperature (``x[-1]``), present
    in every method in :data:`SUPPORTED_METHODS`.

    Cost: electrical energy ``Qth / COP / 100 * grid_signal`` per stage with
    ``Qth = mdot_hp * c_water * (T_HP - T_hp_ret) [kW]``. Comfort band on
    ``T_room`` via slack-weighted nonlinear constraints (quadratic weight
    ``ws``), hard limits on ``Qth`` (26 kW) and ``T_HP`` (5-65 degC).

    Args:
        ad: Discrete-time state matrix, shape (nx, nx), from
            :func:`calculate_discrete_dynamics`.
        bd: Discrete-time input matrix, shape (nx, 1).
        ed: Discrete-time disturbance matrix, shape (nx, 2).
        mdot_hp: Mass-flow rate of the HP circuit [kg/s].
        n_horizon: Number of shooting intervals.
        hp_model: Instantiated i4b ``Heatpump`` model (used for the COP curve).
        name: Acados model name, also used for the generated C code.
        ws: Quadratic weight on the T_room comfort slacks.
        delta_t: Sampling time [s].

    Returns:
        ``(ocp, parameter_manager, param_space, default_param)``.
    """
    nx = ad.shape[0]
    if ad.shape != (nx, nx) or bd.shape != (nx, 1) or ed.shape != (nx, 2):
        raise ValueError(
            f"Dynamics require Ad=(nx, nx), Bd=(nx, 1), Ed=(nx, 2); "
            f"got {ad.shape}, {bd.shape}, {ed.shape}"
        )

    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = n_horizon
    ocp.solver_options.tf = n_horizon * delta_t

    manager = AcadosParameterManager(N_horizon=n_horizon)

    spaces: OrderedDict[str, gym.spaces.Box] = OrderedDict()
    defaults: dict[str, np.ndarray] = {}

    def register_runtime_matrix(name: str, default: np.ndarray) -> ca.SX:
        # Param-space bounds are nominal (0.5-1.5x the default, sign-aware);
        # they do not constrain the solver.
        symbol = manager.register_parameter(name, default=default, differentiable=True)
        spaces[name] = gym.spaces.Box(
            low=np.minimum(0.5 * default, 1.5 * default),
            high=np.maximum(0.5 * default, 1.5 * default),
            dtype=np.float64,
        )
        defaults[name] = default
        return symbol

    # Differentiable runtime matrices. The manager stores each matrix flattened,
    # matching the column-major convention of the ca.reshape calls below.
    ad_sym = register_runtime_matrix("Ad", ad)
    bd_sym = register_runtime_matrix("Bd", bd)
    ed_sym = register_runtime_matrix("Ed", ed)
    mdot_hp_sym = register_runtime_matrix("mdot_hp", np.array([mdot_hp]))

    # Non-differentiable stage-wise forecasts
    T_amb = manager.register_parameter("T_amb", default=np.array([5.0]))
    Qdot_gains = manager.register_parameter("Qdot_gains", default=np.array([0.0]))
    T_set_lower = manager.register_parameter("T_set_lower", default=np.array([20.0]))
    T_set_upper = manager.register_parameter("T_set_upper", default=np.array([26.0]))
    grid_signal = manager.register_parameter("grid_signal", default=np.array([1.0]))

    ######## Model ########
    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", nx)
    ocp.model.u = ca.SX.sym("u", 1)  # T_HP [degC]

    x = ocp.model.x
    u = ocp.model.u
    T_hp = u[0]
    T_room = x[0]
    T_hp_ret = x[-1]

    # Assign the manager's symbols and defaults to the OCP
    manager.assign_to_ocp(ocp)

    Ad = ca.reshape(ad_sym, nx, nx)
    Bd = ca.reshape(bd_sym, nx, 1)
    Ed = ca.reshape(ed_sym, nx, 2)
    mdot_hp = mdot_hp_sym[0]

    d = ca.vertcat(T_amb, Qdot_gains)
    ocp.model.disc_dyn_expr = Ad @ x + Bd @ u + Ed @ d

    ######## Cost ########
    Qth = mdot_hp * C_WATER_SPEC * (T_hp - T_hp_ret) / 1000  # [kW]
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = Qth / (_cop_casadi(hp_model, T_hp, T_amb) * 100) * grid_signal
    ocp.model.cost_expr_ext_cost_e = ca.SX(0)

    ######## Constraints ########
    # h[0]: soft T_room >= T_set_lower, h[1]: soft T_room <= T_set_upper,
    # h[2]: hard 0 <= Qth <= 26 kW
    h_expr = ca.vertcat(T_room - T_set_lower, T_set_upper - T_room, Qth)
    ocp.model.con_h_expr = h_expr
    ocp.constraints.lh = np.array([0.0, 0.0, 0.0])
    ocp.constraints.uh = np.array([ACADOS_INFTY, ACADOS_INFTY, 26.0])

    # Stage 0 gets no con_h constraints implicitly (acados requires an explicit
    # con_h_expr_0); without this, the Qth power cap only holds from stage 1.
    ocp.model.con_h_expr_0 = h_expr
    ocp.constraints.lh_0 = ocp.constraints.lh
    ocp.constraints.uh_0 = ocp.constraints.uh
    ocp.constraints.idxsh_0 = np.array([0, 1])

    ocp.model.con_h_expr_e = h_expr[:2]  # no control at the terminal stage
    ocp.constraints.lh_e = np.array([0.0, 0.0])
    ocp.constraints.uh_e = np.array([ACADOS_INFTY, ACADOS_INFTY])

    ocp.constraints.idxsh = np.array([0, 1])
    ocp.constraints.idxsh_e = np.array([0, 1])
    # acados treats Z as the Hessian of the slack cost (0.5 * Z * s^2);
    # 2 * ws reproduces the ws * s^2 of the original i4b formulation.
    ocp.cost.zl = np.zeros(2)
    ocp.cost.zu = np.zeros(2)
    ocp.cost.Zl = 2.0 * ws * np.ones(2)
    ocp.cost.Zu = 2.0 * ws * np.ones(2)
    ocp.cost.zl_e = np.zeros(2)
    ocp.cost.zu_e = np.zeros(2)
    ocp.cost.Zl_e = 2.0 * ws * np.ones(2)
    ocp.cost.Zu_e = 2.0 * ws * np.ones(2)

    ocp.constraints.lbu = np.array([5.0])
    ocp.constraints.ubu = np.array([65.0])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = 20.0 * np.ones(nx)

    ######## Solver configuration ########
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.print_level = 0

    param_space = gym.spaces.Dict(spaces)

    return ocp, manager, param_space, defaults


class I4bInitializer(AcadosDiffMpcInitializer):
    """Initial iterate: constant x0 along the horizon, mild heating control.

    The all-zero default places the iterate far outside the feasible region
    (T_HP < T_hp_ret violates ``Qth >= 0``) and the QP can fail to recover
    (ACADOS_MINSTEP) for well-insulated houses with long horizons.
    """

    def __init__(self, ocp: AcadosOcp, u_init: float = 35.0):
        iterate = ocp.create_default_initial_iterate().flatten()
        iterate.u[:] = u_init
        self.default_iterate = iterate
        self.n_horizon = ocp.solver_options.N_horizon

    def single_iterate(self, solver_input: AcadosOcpSolverInput) -> AcadosOcpFlattenedIterate:
        iterate = deepcopy(self.default_iterate)
        iterate.x = np.tile(solver_input.x0, self.n_horizon + 1)
        return iterate
