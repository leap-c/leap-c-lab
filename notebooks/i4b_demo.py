"""Demo of the parametric i4b MPC planner."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    return mo, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # i4b heat-pump MPC demo

    Model predictive control of a heat pump in a single-family house
    (the standard 1919-1948 terraced house from the i4b data set). The
    controller plans the heat-pump **supply temperature** over a 24 h
    horizon in 15 min steps, trading electrical energy against thermal
    comfort.

    **Variables along a trajectory** (the house is a linear RC model,
    method `4R3C`):

    | symbol | meaning |
    |---|---|
    | `T_room` | room air temperature [°C], kept in the *comfort band* |
    | `T_wall` | wall temperature [°C] (building's thermal storage) |
    | `T_hp_ret` | return temperature of the heating water loop [°C] |
    | `T_hp_sup` | supply temperature [°C] — **this is the control** |
    | `T_amb` | ambient (outside) temperature [°C], a disturbance |
    | `Qdot_gains` | internal heat gains (people, appliances) [W] |

    Comfort: `T_room` should stay within 20-26 °C (violations are
    penalized softly). The planner is an acados OCP whose building
    dynamics are runtime parameters — one compiled solver serves any
    house of the same model class.

    Install the dependencies with `uv sync --extra i4b` (needs a built acados).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup

    One house, one planner (compiles the acados solver once), and a small
    helper to build observation dicts.
    """)
    return


@app.cell
def _(np):
    from i4b.models.model_buildings import Building
    from i4b.models.model_hvac import Heatpump_AW
    from i4b_data.buildings import sfh_1919_1948_0_soc
    from leapc_lab.i4b.planner import I4bPlanner, I4bPlannerConfig

    building = Building(params=sfh_1919_1948_0_soc, mdot_hp=0.25, method="4R3C")
    planner = I4bPlanner(cfg=I4bPlannerConfig(building_params=sfh_1919_1948_0_soc))
    house = {"params": sfh_1919_1948_0_soc, "model": building}
    return Heatpump_AW, building, house, np, planner


@app.cell
def _(torch):
    def make_obs(state, t_amb, qdot_gains=300.0):
        """Observation dict: current state + disturbances at one moment in time."""
        one = lambda v: torch.tensor([[v]], dtype=torch.float32)  # noqa: E731
        return {
            "state": torch.tensor([state], dtype=torch.float32),
            "disturbances": {"T_amb": one(t_amb), "Qdot_gains": one(qdot_gains)},
            "setpoints": {"T_set_lower": one(20.0), "T_set_upper": one(26.0)},
        }

    return (make_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Open-loop plan

    A single MPC solve from a cold start (everything at 20 °C) on a cold
    winter day (-5 °C ambient, 300 W internal gains). The plot shows the
    full 24-hour plan: how the MPC pre-heats room and walls, and how it
    schedules the supply temperature. Note the horizon end: with nothing
    left to optimize beyond 24 h, the controller lets the room drift —
    the classic receding-horizon artifact; the next re-solve fixes it.
    """)
    return


@app.cell
def _(make_obs, np, planner, plt):
    _ctx_ol, _u0_ol, x_ol, u_ol, _cost_ol = planner(make_obs([20.0, 20.0, 20.0], -5.0))

    x_ol_np = x_ol.detach().cpu().numpy()[0]
    u_ol_np = u_ol.detach().cpu().numpy()[0, :, 0]
    t_ol = np.arange(len(x_ol_np)) * 0.25

    open_loop_fig, ax_ol = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    # Room climate
    ax_ol[0].axhspan(20.0, 26.0, color="#2ca02c", alpha=0.12, label="comfort band")
    ax_ol[0].plot(t_ol, x_ol_np[:, 0], color="#4477aa", lw=1.8, label="T_room")
    ax_ol[0].plot(t_ol, x_ol_np[:, 1], color="#999999", lw=1.0, label="T_wall")
    ax_ol[0].set_ylabel("temperature [°C]")
    ax_ol[0].set_ylim(15, 30)
    ax_ol[0].legend(fontsize=8, frameon=False, loc="lower right")
    ax_ol[0].grid(alpha=0.25)
    ax_ol[0].set_title("Open-loop plan (24 h, -5 °C ambient) — room climate")

    # Hydraulics
    ax_ol[1].axhspan(5.0, 65.0, color="#888888", alpha=0.08, label="u limits")
    ax_ol[1].step(t_ol[:-1], u_ol_np, where="post", color="#4477aa", lw=1.8, label="T_hp_sup (u)")
    ax_ol[1].plot(t_ol, x_ol_np[:, 2], color="#cc6677", lw=1.0, ls="--", label="T_hp_ret")
    ax_ol[1].set_ylabel("temperature [°C]")
    ax_ol[1].set_xlabel("hours")
    ax_ol[1].legend(fontsize=8, frameon=False, loc="lower right")
    ax_ol[1].grid(alpha=0.25)
    ax_ol[1].set_title("Hydraulics — supply/return spread drives thermal power")
    open_loop_fig.tight_layout()
    open_loop_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Closed-loop rollout

    In operation the MPC re-plans every 15 minutes: solve, apply only the
    first control, advance the i4b simulator, repeat. The ambient
    temperature follows a sinusoidal warm-up over 12 hours. The solver
    context (`ctx`) is chained between steps for warm-starting.
    """)
    return


@app.cell
def _(Heatpump_AW, building, make_obs, np, planner, plt):
    from i4b.simulator import Model_simulator

    sim_cl = Model_simulator(Heatpump_AW(mdot_HP=0.25), building, 900)

    state_cl = {k: 20.0 for k in building.state_keys}
    n_steps_cl = 48
    ctx_cl = None
    room_cl, sup_cl, amb_cl = [], [], []
    for step in range(n_steps_cl):
        amb_val = -5.0 + 3.0 * np.sin(step * 0.1)
        obs_cl = make_obs([float(state_cl[k]) for k in building.state_keys], amb_val)
        ctx_cl, u0_cl, _x_cl, _u_cl, _cost_cl = planner(obs_cl, ctx=ctx_cl)
        action_cl = float(u0_cl[0, 0])
        room_cl.append(state_cl["T_room"])
        sup_cl.append(action_cl)
        amb_cl.append(amb_val)
        res_cl = sim_cl.get_next_state(
            state_cl,
            action_cl,
            {"T_amb": amb_val, "Qdot_gains": 300.0},
        )
        state_cl = res_cl["state"]

    t_cl = np.arange(n_steps_cl) * 0.25
    closed_loop_fig, ax_cl = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax_cl[0].axhspan(20.0, 26.0, color="#2ca02c", alpha=0.12, label="comfort band")
    ax_cl[0].plot(t_cl, room_cl, color="#242424", lw=1.5, label="T_room")
    ax_cl[0].set_ylabel("room [°C]")
    ax_cl[0].legend(fontsize=8, frameon=False, loc="upper right")
    ax_cl[0].grid(alpha=0.25)
    ax_cl[0].set_title("12-hour closed-loop rollout")
    ax_cl[1].axhspan(5.0, 65.0, color="#888888", alpha=0.08, label="u limits")
    ax_cl[1].step(t_cl, sup_cl, where="post", color="#4477aa", label="T_hp_sup (u)")
    ax_cl[1].set_ylabel("supply [°C]")
    ax_cl[1].legend(fontsize=8, frameon=False, loc="upper right")
    ax_cl[1].grid(alpha=0.25)
    ax_cl[2].plot(t_cl, amb_cl, color="#888888")
    ax_cl[2].set_ylabel("ambient [°C]")
    ax_cl[2].set_xlabel("hours")
    ax_cl[2].grid(alpha=0.25)
    closed_loop_fig.tight_layout()
    closed_loop_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison with the original i4b MPC

    Same open-loop scenario, this time solved with the reference
    implementation from the i4b repo (`MPC_solver`: IPOPT interior-point
    solver + Radau collocation on the *continuous* building dynamics),
    overlaid with our acados port (SQP on the exactly discretized linear
    dynamics). Both share the same objective (electrical energy weighted
    by the grid signal) and the same soft comfort band — remaining
    differences are numerical: integration scheme, slack-cost quadrature,
    and the solver itself.
    """)
    return


@app.cell
def _(Heatpump_AW, building, make_obs, np, planner, plt):
    import os
    import tempfile

    from i4b.controller.mpc.casadi_framework import MPC_solver

    n_horizon_cp = planner.cfg.n_horizon
    mpc_ref = MPC_solver(
        ".",
        "comparison",
        Heatpump_AW(mdot_HP=0.25),
        building,
        nx=3,
        ns=2,
        nc=4,
        h=int(planner.cfg.delta_t),
        nk=n_horizon_cp,
        ws=planner.cfg.ws,
    )
    # P rows: [T_amb, Qdot_gains, unused, T_set_lower, grid_signal]
    p_stage = np.array([-5.0, 300.0, 0.0, 20.0, 1.0])
    mpc_ref.update_NLP(np.full(3, 20.0))
    with tempfile.TemporaryDirectory() as tmpdir:  # keep ipopt.log out of the repo
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            _uk_cp, _xk_cp, res_cp = mpc_ref.solve_NLP(
                np.tile(p_stage, (n_horizon_cp, 1)), return_res=True
            )
        finally:
            os.chdir(cwd)

    d_cp, nx_cp, ns_cp, nu_cp = 3, 3, 2, 1
    chunk_cp = (d_cp + 1) * (nx_cp + ns_cp) + nu_cp
    sol_cp = np.asarray(res_cp["x"]).reshape(-1)
    x_ref = np.array(
        [sol_cp[k * chunk_cp : k * chunk_cp + nx_cp] for k in range(n_horizon_cp)]
        + [sol_cp[-(nx_cp + ns_cp) : -ns_cp]]
    )
    u_ref = np.array(
        [sol_cp[k * chunk_cp + (d_cp + 1) * (nx_cp + ns_cp)] for k in range(n_horizon_cp)]
    )

    _ctx_cp, _u0_cp, x_ours, u_ours, _cost_cp = planner(make_obs([20.0, 20.0, 20.0], -5.0))
    x_ours_np = x_ours.detach().cpu().numpy()[0]
    u_ours_np = u_ours.detach().cpu().numpy()[0, :, 0]
    t_cp = np.arange(n_horizon_cp + 1) * 0.25

    droom_max = np.abs(x_ref[:, 0] - x_ours_np[:, 0]).max()
    droom_mean = np.abs(x_ref[:, 0] - x_ours_np[:, 0]).mean()

    cmp_fig, ax_cp = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax_cp[0].axhspan(20.0, 26.0, color="#2ca02c", alpha=0.12)
    ax_cp[0].plot(t_cp, x_ref[:, 0], color="#888888", lw=1.5, label="original i4b MPC (IPOPT)")
    ax_cp[0].plot(
        t_cp, x_ours_np[:, 0], color="#4477aa", lw=1.5, ls="--", label="leap-c acados (ours)"
    )
    ax_cp[0].set_ylabel("room [°C]")
    ax_cp[0].legend(fontsize=8, frameon=False)
    ax_cp[0].grid(alpha=0.25)
    ax_cp[0].set_title(f"Open-loop comparison — max ΔT_room {droom_max:.2f} K, mean {droom_mean:.3f} K")
    ax_cp[1].step(t_cp[:-1], u_ref, where="post", color="#888888", lw=1.5)
    ax_cp[1].step(t_cp[:-1], u_ours_np, where="post", color="#4477aa", lw=1.5, ls="--")
    ax_cp[1].set_ylabel("supply [°C]")
    ax_cp[1].set_xlabel("hours")
    ax_cp[1].grid(alpha=0.25)
    cmp_fig.tight_layout()
    cmp_fig
    return


if __name__ == "__main__":
    app.run()
