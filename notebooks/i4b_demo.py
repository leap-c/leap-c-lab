"""Demo of the parametric i4b MPC planner."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    # i4b physics + reference controller
    from i4b.controller.mpc.casadi_framework import MPC_solver
    from i4b.models.model_buildings import Building
    from i4b.models.model_hvac import Heatpump_AW
    from i4b.simulator import Model_simulator
    from i4b_data.buildings import sfh_1919_1948_0_soc

    # leap-c-lab registered planners
    from leapc_lab import create_planner

    return (
        Building,
        Heatpump_AW,
        MPC_solver,
        Model_simulator,
        create_planner,
        mo,
        np,
        plt,
        sfh_1919_1948_0_soc,
    )


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

    Comfort: `T_room` should stay within the comfort band (violations are
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

    Two ways into the same stack:

    1. **Generic**: `create_planner("i4b", ...)` — the registered planner,
       one line, compiles the acados solver once.
    2. **Explicit**: `Building` + `Heatpump_AW` + `Model_simulator` — the
       "true" physics the MPC plans against, built by hand so the pieces
       are visible (this mirrors what the upstream i4b docs describe).

    The scenario is a cold winter day: ambient around 0 °C with a diurnal
    cycle and 300 W of internal gains, plus a **night set-back** — the
    comfort lower bound drops from 20 °C to 16 °C between 22:00 and
    06:00.
    """)
    return


@app.cell
def _(
    Building,
    Heatpump_AW,
    Model_simulator,
    create_planner,
    np,
    sfh_1919_1948_0_soc,
):
    # Generic: registered parametric planner
    planner = create_planner("i4b", building_params=sfh_1919_1948_0_soc)

    # Explicit: the "true" house (heat pump ODEs + building ODEs)
    building = Building(params=sfh_1919_1948_0_soc, mdot_hp=0.25, method="4R3C")
    hp = Heatpump_AW(mdot_HP=0.25)
    sim = Model_simulator(hp, building, 900)

    # Scenario: cold winter day with a diurnal ambient cycle...
    D_H = 0.25  # 15 min in hours
    t_amb = lambda h: 0.5 + 2.0 * np.sin(2 * np.pi * (h - 10.0) / 24.0)  # noqa: E731
    qdot_gains = 300.0  # W

    # ...and night set-back of the lower comfort bound
    T_SET_DAY, T_SET_NIGHT, T_SET_MAX = 20.0, 16.0, 26.0
    NIGHT_START, NIGHT_END = 22.0, 30.0  # 22:00 -> 06:00 next day

    def t_set_lower(h):
        clock = h % 24.0
        return T_SET_NIGHT if clock >= NIGHT_START or clock < NIGHT_END % 24.0 else T_SET_DAY

    return (
        D_H,
        T_SET_MAX,
        building,
        planner,
        qdot_gains,
        sim,
        t_amb,
        t_set_lower,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Open-loop plan

    A single MPC solve from a cold start (everything at 20 °C) on the
    winter day described above. The plot shows the full 24-hour plan: how
    the MPC pre-heats room and walls, and how it schedules the supply
    temperature. Note the horizon end: with nothing left to optimize
    beyond 24 h, the controller lets the room drift — the classic
    receding-horizon artifact; the next re-solve fixes it.

    Observations for the planner are plain dicts. The dict here makes
    what's actually fed to the solver explicit.
    """)
    return


@app.cell
def _(D_H, T_SET_MAX, np, planner, plt, qdot_gains, t_amb):
    _day = np.arange(planner.cfg.n_horizon + 1) * D_H
    obs_ol = {
        "state": [20.0, 20.0, 20.0],
        "disturbances": {"T_amb": float(t_amb(18.0)), "Qdot_gains": qdot_gains},
        "setpoints": {"T_set_lower": 20.0, "T_set_upper": T_SET_MAX},
    }
    _ctx_ol, _u0_ol, x_ol, u_ol, _cost_ol = planner(obs_ol)

    x_ol_np = x_ol.detach().cpu().numpy()[0]
    u_ol_np = u_ol.detach().cpu().numpy()[0, :, 0]

    open_loop_fig, ax_ol = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)

    # Room climate
    ax_ol[0].axhspan(20.0, T_SET_MAX, color="#2ca02c", alpha=0.12, label="comfort band")
    ax_ol[0].plot(_day, x_ol_np[:, 0], color="#4477aa", lw=1.8, label="T_room")
    ax_ol[0].set_ylabel("room [°C]")
    ax_ol[0].set_ylim(15, 30)
    ax_ol[0].legend(fontsize=8, frameon=False, loc="lower right")
    ax_ol[0].grid(alpha=0.25)
    ax_ol[0].set_title("Open-loop plan from 18:00 (24 h) — room climate")

    # Wall (thermal storage)
    ax_ol[1].plot(_day, x_ol_np[:, 1], color="#999999", lw=1.8, label="T_wall")
    ax_ol[1].set_ylabel("wall [°C]")
    ax_ol[1].legend(fontsize=8, frameon=False, loc="lower right")
    ax_ol[1].grid(alpha=0.25)

    # Hydraulics
    ax_ol[2].axhspan(5.0, 65.0, color="#888888", alpha=0.08, label="u limits")
    ax_ol[2].step(_day[:-1], u_ol_np, where="post", color="#4477aa", lw=1.8, label="T_hp_sup (u)")
    ax_ol[2].plot(_day, x_ol_np[:, 2], color="#cc6677", lw=1.0, ls="--", label="T_hp_ret")
    ax_ol[2].set_ylabel("temperature [°C]")
    ax_ol[2].set_xlabel("hours")
    ax_ol[2].legend(fontsize=8, frameon=False, loc="lower right")
    ax_ol[2].grid(alpha=0.25)
    open_loop_fig.tight_layout()
    open_loop_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Closed-loop rollout

    In operation the MPC re-plans every 15 minutes: build the observation
    (current state + disturbances + setpoints + **forecasts**), solve,
    apply only the first control, advance the i4b simulator, repeat. The
    solver context (`ctx`) is chained between steps for warm-starting.

    The rollout starts at **18:00**: two hours in, the night set-back
    kicks in and the lower comfort bound drops to 16 °C — the MPC sees
    this coming in its **forecast** and lets room temperature sag
    overnight, saving energy exactly where comfort is relaxed.
    """)
    return


@app.cell
def _(
    D_H,
    T_SET_MAX,
    building,
    np,
    planner,
    qdot_gains,
    sim,
    t_amb,
    t_set_lower,
):
    H_START, N_STEPS = 18.0, 48  # 18:00 -> 06:00 next day
    N = planner.cfg.n_horizon

    state_cl = {k: 20.0 for k in building.state_keys}
    ctx_cl = None
    plans = []  # per-step planned trajectories for the interactive view below
    history = {"hour": [], "room": [], "wall": [], "sup": [], "amb": [], "low": []}
    for step in range(N_STEPS):
        now = H_START + step * D_H
        fc_hours = now + np.arange(N + 1) * D_H
        obs_cl = {
            "state": [float(state_cl[k]) for k in building.state_keys],
            "disturbances": {"T_amb": float(t_amb(now)), "Qdot_gains": qdot_gains},
            "setpoints": {
                "T_set_lower": float(t_set_lower(now)),
                "T_set_upper": T_SET_MAX,
            },
            "forecast": {
                "T_amb": [float(t_amb(h)) for h in fc_hours],
                "Qdot_gains": [qdot_gains for _h in fc_hours],
                "T_set_lower": [float(t_set_lower(h)) for h in fc_hours],
            },
        }
        ctx_cl, u0_cl, x_cl, u_cl, _cost_cl = planner(obs_cl, ctx=ctx_cl)
        plans.append(
            (
                fc_hours,
                x_cl.detach().cpu().numpy()[0],
                u_cl.detach().cpu().numpy()[0, :, 0],
                np.asarray(obs_cl["forecast"]["T_amb"]),
                np.asarray(obs_cl["forecast"]["Qdot_gains"]),
            )
        )

        history["hour"].append(now)
        history["room"].append(state_cl["T_room"])
        history["wall"].append(state_cl["T_wall"])
        history["sup"].append(float(u0_cl[0, 0]))
        history["amb"].append(float(t_amb(now)))
        history["low"].append(float(t_set_lower(now)))

        res_cl = sim.get_next_state(
            state_cl,
            float(u0_cl[0, 0]),
            {"T_amb": float(t_amb(now)), "Qdot_gains": qdot_gains},
        )
        state_cl = res_cl["state"]
    return history, plans


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What is the MPC thinking?

    Drag the slider to any replan step. The shaded grey interval marks
    the selected time and stays one-third across the plot: 12 hours of
    realized history are on the left and the 24-hour plan is on the
    right. Dashed curves show the plan computed at that moment; solid
    curves show only the history available then. The lower and upper
    green comfort constraints use the same stepped rendering.

    Watch how plans through the night anticipate the set-back: pre-heating
    is skipped when comfort relaxes, and recovery is timed to 06:00.
    """)
    return


@app.cell
def _(mo, plans):
    plan_slider = mo.ui.slider(
        steps=range(len(plans)), value=10, label="replan step", show_value=True
    )
    return (plan_slider,)


@app.cell
def _(D_H, T_SET_MAX, history, np, plan_slider, plans, plt, t_set_lower):
    _fc_hours, x_pl, u_pl, amb_pl, gains_pl = plans[plan_slider.value]
    _history_end = plan_slider.value + 1
    _hours = np.asarray(history["hour"][:_history_end])
    _now = _fc_hours[0]
    _window_start, _window_end = _now - 12.0, _now + 24.0
    _constraint_hours = np.arange(_window_start, _window_end + D_H, D_H)
    plan_fig, ax_p = plt.subplots(5, 1, figsize=(10, 9), sharex=True)

    # Room: history + set-back band + active plan
    constraint_kw = {
        "where": "post",
        "color": "#2ca02c",
        "lw": 1.1,
        "ls": "--",
    }
    ax_p[0].step(
        _constraint_hours,
        [float(t_set_lower(h)) for h in _constraint_hours],
        **constraint_kw,
        label="T_set_lower",
    )
    ax_p[0].step(
        _constraint_hours,
        np.full_like(_constraint_hours, T_SET_MAX),
        **constraint_kw,
        label="T_set_upper",
    )
    ax_p[0].plot(
        _hours,
        history["room"][:_history_end],
        color="#242424",
        lw=1.5,
        label="T_room (realized)",
    )
    ax_p[0].plot(_fc_hours, x_pl[:, 0], color="#4477aa", lw=1.8, ls="--", label="T_room (plan)")
    ax_p[0].set_ylabel("room [°C]")
    ax_p[0].set_ylim(14, 28)
    ax_p[0].legend(fontsize=8, frameon=False, loc="upper right")
    ax_p[0].grid(alpha=0.25)
    ax_p[0].set_title(f"Active plan at replan step {plan_slider.value} (t = {_now:.2f} h)")

    # Wall: history + active plan
    ax_p[1].plot(
        _hours,
        history["wall"][:_history_end],
        color="#999999",
        lw=1.5,
        label="T_wall (realized)",
    )
    ax_p[1].plot(_fc_hours, x_pl[:, 1], color="#4477aa", lw=1.8, ls="--", label="T_wall (plan)")
    ax_p[1].set_ylabel("wall [°C]")
    ax_p[1].legend(fontsize=8, frameon=False, loc="upper right")
    ax_p[1].grid(alpha=0.25)

    # Supply: history + active plan
    ax_p[2].axhspan(5.0, 65.0, color="#888888", alpha=0.08, label="u limits")
    step_kw = {"where": "post", "color": "#4477aa", "lw": 1.5}
    ax_p[2].step(_hours, history["sup"][:_history_end], **step_kw, label="T_hp_sup (realized)")
    ax_p[2].step(_fc_hours[:-1], u_pl, **step_kw, ls="--", label="u (plan)")
    ax_p[2].set_ylabel("supply [°C]")
    ax_p[2].legend(fontsize=8, frameon=False, loc="upper right")
    ax_p[2].grid(alpha=0.25)

    # Exogenous inputs: realized history + active forecasts
    ax_p[3].plot(
        _hours,
        history["amb"][:_history_end],
        color="#777777",
        lw=1.5,
        label="T_amb (realized)",
    )
    ax_p[3].plot(_fc_hours, amb_pl, color="#777777", lw=1.5, ls="--", label="T_amb (forecast)")
    ax_p[3].set_ylabel("ambient [°C]")
    ax_p[3].legend(fontsize=8, frameon=False, loc="upper right")
    ax_p[3].grid(alpha=0.25)
    ax_p[4].step(
        _hours,
        np.full(_history_end, gains_pl[0]),
        where="post",
        color="#cc6677",
        lw=1.5,
        label="Qdot_gains (realized)",
    )
    ax_p[4].step(
        _fc_hours,
        gains_pl,
        where="post",
        color="#cc6677",
        lw=1.5,
        ls="--",
        label="Qdot_gains (forecast)",
    )
    ax_p[4].set_ylabel("gains [W]")
    ax_p[4].set_xlabel("hours")
    ax_p[4].legend(fontsize=8, frameon=False, loc="upper right")
    ax_p[4].grid(alpha=0.25)

    for _axis in ax_p:
        _axis.axvspan(_now - D_H / 2, _now + D_H / 2, color="#888888", alpha=0.18)
        _axis.set_xlim(_window_start, _window_end)
        _axis.margins(x=0)

    plan_fig.tight_layout(pad=0.4)
    return (plan_fig,)


@app.cell
def _(mo, plan_fig, plan_slider):
    mo.vstack([plan_slider, plan_fig], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Comparison with the original i4b MPC

    Same open-loop scenario solved with the reference implementation
    from the i4b repo (`MPC_solver`: IPOPT + Radau collocation on the
    continuous dynamics), overlaid with our acados port (SQP on the
    exactly discretized linear dynamics). Same objective (energy × grid
    signal) and soft comfort band — remaining differences are numerical:
    integration scheme, slack-cost quadrature, solver. Construction
    follows the usage documented in the i4b repository itself. Solve
    latency is the median of three runs after construction; one-time
    model and code-generation costs are excluded.
    """)
    return


@app.cell
def _(
    Building,
    Heatpump_AW,
    MPC_solver,
    np,
    planner,
    plt,
    qdot_gains,
    sfh_1919_1948_0_soc,
    t_amb,
):
    import os
    import tempfile
    from time import perf_counter

    n_horizon_cp = planner.cfg.n_horizon
    mpc_ref = MPC_solver(
        ".",
        "comparison",
        Heatpump_AW(mdot_HP=0.25),
        Building(params=sfh_1919_1948_0_soc, mdot_hp=0.25, method="4R3C"),
        nx=3,
        ns=2,
        nc=4,
        h=int(planner.cfg.delta_t),
        nk=n_horizon_cp,
        ws=planner.cfg.ws,
    )
    # P rows: [T_amb, Qdot_gains, unused, T_set_lower, grid_signal]
    comparison_t_amb = float(t_amb(18.0))
    p_stage = np.array([comparison_t_amb, qdot_gains, 0.0, 20.0, 1.0])
    p_horizon = np.tile(p_stage, (n_horizon_cp, 1))
    mpc_ref.update_NLP(np.full(3, 20.0))

    ref_solve_times = []
    with tempfile.TemporaryDirectory() as tmpdir:  # keep ipopt.log out of the repo
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            for _trial in range(3):
                _start = perf_counter()
                _uk_cp, _xk_cp, res_cp = mpc_ref.solve_NLP(p_horizon, return_res=True)
                ref_solve_times.append(perf_counter() - _start)
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

    comparison_obs = {
        "state": [20.0, 20.0, 20.0],
        "disturbances": {"T_amb": comparison_t_amb, "Qdot_gains": qdot_gains},
    }
    ours_solve_times = []
    for _trial in range(3):
        _start = perf_counter()
        _ctx_cp, _u0_cp, x_ours, u_ours, _cost_cp = planner(comparison_obs)
        ours_solve_times.append(perf_counter() - _start)

    x_ours_np = x_ours.detach().cpu().numpy()[0]
    u_ours_np = u_ours.detach().cpu().numpy()[0, :, 0]
    t_cp = np.arange(n_horizon_cp + 1) * 0.25

    droom_max = np.abs(x_ref[:, 0] - x_ours_np[:, 0]).max()
    droom_mean = np.abs(x_ref[:, 0] - x_ours_np[:, 0]).mean()
    ref_ms = 1e3 * np.median(ref_solve_times)
    ours_ms = 1e3 * np.median(ours_solve_times)
    solve_speedup = ref_ms / ours_ms

    cmp_fig, ax_cp = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax_cp[0].axhspan(20.0, 26.0, color="#2ca02c", alpha=0.12)
    ax_cp[0].plot(t_cp, x_ref[:, 0], color="#888888", lw=1.5, label="original i4b MPC (IPOPT)")
    ax_cp[0].plot(
        t_cp, x_ours_np[:, 0], color="#4477aa", lw=1.5, ls="--", label="leap-c acados (ours)"
    )
    ax_cp[0].set_ylabel("room [°C]")
    ax_cp[0].legend(fontsize=8, frameon=False)
    ax_cp[0].grid(alpha=0.25)
    ax_cp[0].set_title(
        f"Open-loop comparison — max ΔT_room {droom_max:.2f} K, mean {droom_mean:.3f} K"
    )
    ax_cp[1].step(t_cp[:-1], u_ref, where="post", color="#888888", lw=1.5)
    ax_cp[1].step(t_cp[:-1], u_ours_np, where="post", color="#4477aa", lw=1.5, ls="--")
    ax_cp[1].set_ylabel("supply [°C]")
    ax_cp[1].set_xlabel("hours")
    ax_cp[1].set_xlim(t_cp[0], t_cp[-1])
    ax_cp[1].grid(alpha=0.25)
    cmp_fig.tight_layout()
    cmp_fig
    return ours_ms, ref_ms, solve_speedup


@app.cell(hide_code=True)
def _(mo, ours_ms, ref_ms, solve_speedup):
    mo.md(f"""
    **Online solve latency** (median of three solves; construction excluded)

    | Solver | Median solve time | Relative speed |
    |---|---:|---:|
    | Original i4b MPC (IPOPT) | {ref_ms:.1f} ms | 1.0x |
    | leap-c acados | {ours_ms:.1f} ms | {solve_speedup:.1f}x faster |
    """)
    return


if __name__ == "__main__":
    app.run()
