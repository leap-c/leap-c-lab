"""Parametric acados MPC planner for i4b building heat-pump control."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from leap_c.diff_mpc.function import AcadosDiffMpcCtx
from leap_c.parameters.utils import broadcast_default_param, n_segments
from leap_c.torch import AcadosDiffMpcTorch
from leap_c.utils.collate import collate_acados_diff_mpc_ctx

from leapc_lab.i4b.acados_ocp import (
    SUPPORTED_METHODS,
    I4bInitializer,
    calculate_discrete_dynamics,
    export_parametric_ocp,
)
from leapc_lab.planner import ParameterizedPlanner

if TYPE_CHECKING:
    from i4b.models.model_buildings import Building
    from i4b.models.model_hvac import Heatpump


@dataclass(kw_only=True)
class I4bPlannerConfig:
    """Configuration for the i4b planner.

    The objective mixes energy and price costs per stage via
    ``grid_signal = lam * price / pi_ref + (1 - lam)``, and penalizes comfort
    violations with the quadratic slack weight ``ws``.

    Attributes:
        building_params: Building parameter dict used to construct the Building
            model when none is passed explicitly to ``I4bPlanner``.
        method: Building thermal-model type, one of :data:`SUPPORTED_METHODS`.
        mdot_hp: Mass-flow rate of the HP circuit [kg/s].
        n_horizon: Number of shooting intervals (default: 96 = 24 h in 15-min
            steps, as in the reference i4b MPC).
        ws: Quadratic weight on the T_room comfort slacks.
        lam: Price-vs-energy mixing weight for ``grid_signal`` (0 = energy-only).
            Takes effect only with a "price" forecast, not provided by the i4b
            env yet.
        pi_ref: Reference price for normalizing the price forecast.
        delta_t: Sampling time [s].
        discount_factor: Discount factor along the MPC horizon.
        n_batch_init: Initially supported batch size for the batch OCP solver.
        num_threads_batch_solver: Number of parallel threads for the batch solver.
        dtype: Output tensor dtype. Uses PyTorch default if None.
    """

    building_params: dict | None = None
    method: str = "4R3C"
    mdot_hp: float = 0.25
    n_horizon: int = 96  # 96 x 900 s = 24 h
    ws: float = 1.0
    lam: float = 0.0
    pi_ref: float = 1.0
    delta_t: float = 900.0
    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {SUPPORTED_METHODS}, got {self.method!r}")


# TODO: Batch solving across building methods (e.g. a 2R2C and a 4R3C house in
# the same batch) is not supported yet — batch elements must share the same nx.
# Mixing different houses of the same method works.
class I4bPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Parametric acados MPC planner for i4b building heat-pump control.

    Each house is a linear thermal model, discretized to

    ``x_next = Ad @ x + Bd @ u + Ed @ d``   with ``d = (T_amb, Qdot_gains)``,

    where ``x`` are the building temperatures, ``u`` the HP supply
    temperature, and ``mdot_hp`` the HP mass-flow rate. One compiled OCP
    serves all houses of the same building method: ``Ad``, ``Bd``, ``Ed``
    and ``mdot_hp`` are differentiable runtime parameters that can be
    overwritten per solve and per batch element
    (see :func:`~leapc_lab.i4b.acados_ocp.calculate_discrete_dynamics`).

    Args:
        cfg: Planner configuration.
        building_model: Optional pre-built i4b ``Building`` model.
        hp_model: Optional pre-built i4b ``Heatpump`` model.
        export_directory: Directory for generated acados C code.
    """

    cfg: I4bPlannerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: I4bPlannerConfig | None = None,
        building_model: "Building | None" = None,
        hp_model: "Heatpump | None" = None,
        export_directory: Path | None = None,
    ) -> None:
        super().__init__()

        self.cfg = I4bPlannerConfig() if cfg is None else cfg

        if building_model is None:
            from i4b.models.model_buildings import Building

            building_model = Building(
                params=self.cfg.building_params,
                mdot_hp=self.cfg.mdot_hp,
                method=self.cfg.method,
            )
        if hp_model is None:
            from i4b.models.model_hvac import Heatpump_AW

            hp_model = Heatpump_AW(mdot_HP=self.cfg.mdot_hp)

        ad, bd, ed = calculate_discrete_dynamics(building_model, self.cfg.delta_t)
        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            ad=ad,
            bd=bd,
            ed=ed,
            mdot_hp=hp_model.mdot_HP,
            n_horizon=self.cfg.n_horizon,
            hp_model=hp_model,
            ws=self.cfg.ws,
            delta_t=self.cfg.delta_t,
        )
        diff_mpc = AcadosDiffMpcTorch(
            ocp=ocp,
            parameter_manager=param_manager,
            initializer=I4bInitializer(ocp),
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        self.param_manager = param_manager
        self.diff_mpc = diff_mpc
        self._param_space = param_space
        self._default_param = default_param

    def forward(
        self,
        obs: dict[str, Any],
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[AcadosDiffMpcCtx, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the MPC problem.

        Args:
            obs: Dict observation with the current state, disturbances and
                setpoints, plus optional forecasts::

                    "state":        (B, nx) building temperatures [degC]
                    "disturbances": {"T_amb": (B, 1), "Qdot_gains": (B, 1)}
                    "setpoints":    {"T_set_lower": (B, 1), "T_set_upper": (B, 1)}
                    "forecast":     optional per-timestep forecasts (B, N_fc)
                                    for any of the keys above, plus "price".
                                    Short forecasts are padded with their last
                                    value, long ones cut.

                Without a "price" forecast the objective is energy-only
                (``grid_signal == 1``), see :class:`I4bPlannerConfig` for the
                mixing formula.
            action: Optional warm-start action.
            params: Optional override of the discrete dynamics
                (see class docstring), batched as
                ``{"Ad": (B, nx, nx), "Bd": (B, nx, 1), "Ed": (B, nx, 2),
                "mdot_hp": (B, 1)}``. Defaults (the dynamics of
                ``cfg.building_params``) are used if None.
            ctx: Optional solver context for warm-starting.

        Returns:
            ``(ctx, u0, x_traj, u_traj, cost)`` with ``u0`` the first supply
            temperature [degC], shape (B, 1).
        """
        x0 = obs["state"]
        batch_size = x0.shape[0]
        n_horizon = self.cfg.n_horizon

        t_amb_now = obs["disturbances"]["T_amb"].detach().cpu().numpy()
        qdot_gains_now = obs["disturbances"]["Qdot_gains"].detach().cpu().numpy()
        t_set_lower_now = obs["setpoints"]["T_set_lower"].detach().cpu().numpy()
        t_set_upper_now = obs["setpoints"]["T_set_upper"].detach().cpu().numpy()

        fc = obs.get("forecast", {})

        t_amb_staged = _forecast_to_stagewise(fc.get("T_amb"), t_amb_now, n_horizon)
        qdot_gains_staged = _forecast_to_stagewise(fc.get("Qdot_gains"), qdot_gains_now, n_horizon)
        t_set_lower_staged = _forecast_to_stagewise(
            fc.get("T_set_lower"), t_set_lower_now, n_horizon
        )
        t_set_upper_staged = _forecast_to_stagewise(
            fc.get("T_set_upper"), t_set_upper_now, n_horizon
        )

        # TODO: no env provides a "price" forecast yet; until then the price
        # path amounts to a constant grid_signal == 1 (energy-only objective).
        price_fc = fc.get("price")
        if price_fc is not None:
            price_now = price_fc[:, :1].detach().cpu().numpy()
        else:
            price_now = np.full((batch_size, 1), self.cfg.pi_ref)
        price_staged = _forecast_to_stagewise(price_fc, price_now, n_horizon)
        grid_signal_staged = self.cfg.lam * (price_staged / self.cfg.pi_ref) + (1.0 - self.cfg.lam)

        params_dict: dict[str, Any] = {
            "T_amb": t_amb_staged,
            "Qdot_gains": qdot_gains_staged,
            "T_set_lower": t_set_lower_staged,
            "T_set_upper": t_set_upper_staged,
            "grid_signal": grid_signal_staged,
        }
        if params is not None:
            params_dict.update(_flatten_matrix_params(params))

        return self.diff_mpc(x0=x0, u0=action, params=params_dict, ctx=ctx)

    def default_param(self, obs: np.ndarray | torch.Tensor | None = None) -> dict[str, np.ndarray]:
        return broadcast_default_param(self._default_param, obs)


def _forecast_to_stagewise(
    forecast: torch.Tensor | None,
    current: np.ndarray,
    n_horizon: int,
) -> np.ndarray:
    """Reshape a per-timestep forecast to one value per MPC stage: (B, N+1, 1).

    Missing stages are filled with the last forecast value; without a forecast
    the constant ``current`` value is used. Longer forecasts are truncated.
    """
    n_stages = n_segments("stagewise", n_horizon)
    batch_size = current.shape[0]
    if forecast is None:
        return np.broadcast_to(current[:, np.newaxis, :], (batch_size, n_stages, 1)).copy()
    staged = forecast.detach().cpu().numpy()[:, :, np.newaxis]  # (B, N_fc, 1)
    if staged.shape[1] >= n_stages:
        return staged[:, :n_stages, :].copy()
    pad = np.broadcast_to(staged[:, -1:, :], (batch_size, n_stages - staged.shape[1], 1)).copy()
    return np.concatenate([staged, pad], axis=1)


def _flatten_matrix_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten batched matrix parameters: (B, m, n) -> (B, m * n), column-wise.

    Differentiable overwrites must match each parameter's flattened layout;
    for a matrix that is column order, hence the transpose before the reshape.
    """
    return {
        name: value.transpose(-1, -2).reshape(value.shape[0], -1) for name, value in params.items()
    }
