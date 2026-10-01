"""Deterministic bifurcation analysis of the single Morris-Lecar node.

Locates, for the parameter set used in the manuscript
(C = 20, gCa = 4.4, gK = 8, gL = 2, VCa = 120, VK = -84, VL = -60,
 V1 = -1.2, V2 = 18, V3 = 2, V4 = 30, phi = 0.04):

  * the equilibrium branch V*(I) and its eigenvalues,
  * the Hopf bifurcation current I_H (trace of the Jacobian = 0),
  * the first Lyapunov coefficient l1 at the Hopf point (criticality),
  * the fold of limit cycles (double-cycle bifurcation) I_fold, which bounds
    the bistable window from below,
  * the stable and unstable limit-cycle branches,
  * the distance of the working point I = 88 from both bifurcations.

Outputs (written to --output-dir):
  hopf_distance_summary.json     scalar results
  working_point_table.csv        eigenvalues / distances for the scanned currents
  branch_equilibrium.csv         V*(I) on the equilibrium branch
  branch_limit_cycles.csv        Vmax/Vmin of the stable and unstable cycles
  fig_bifurcation.png/.pdf       panel (a) of the new figure (optional)

Requires numpy and scipy; matplotlib only when --make-figure is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# Model, identical to the parameters quoted in Sec. 2.1 of the manuscript
# ----------------------------------------------------------------------------
CAP = 20.0
G_CA, G_K, G_L = 4.4, 8.0, 2.0
V_CA, V_K, V_L = 120.0, -84.0, -60.0
V1, V2, V3, V4 = -1.2, 18.0, 2.0, 30.0
PHI = 0.04

WORKING_CURRENTS = (80.0, 82.0, 84.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0)


def m_inf(v):
    return 0.5 * (1.0 + np.tanh((v - V1) / V2))


def dm_inf(v):
    return 0.5 / V2 / np.cosh((v - V1) / V2) ** 2


def w_inf(v):
    return 0.5 * (1.0 + np.tanh((v - V3) / V4))


def dw_inf(v):
    return 0.5 / V4 / np.cosh((v - V3) / V4) ** 2


def tau_w(v):
    return 1.0 / np.cosh((v - V3) / (2.0 * V4))


def rhs(_t, y, current):
    v, w = y
    dv = (current - G_CA * m_inf(v) * (v - V_CA) - G_K * w * (v - V_K) - G_L * (v - V_L)) / CAP
    dw = PHI * (w_inf(v) - w) / tau_w(v)
    return [dv, dw]


def rhs_backward(t, y, current):
    dv, dw = rhs(t, y, current)
    return [-dv, -dw]


def current_of_equilibrium(v):
    """Applied current for which v is an equilibrium (with w = w_inf(v))."""
    return G_CA * m_inf(v) * (v - V_CA) + G_K * w_inf(v) * (v - V_K) + G_L * (v - V_L)


def equilibrium(current, lo=-80.0, hi=60.0):
    grid = np.linspace(lo, hi, 140001)
    values = current_of_equilibrium(grid) - current
    crossings = np.where(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    if len(crossings) == 0:
        raise RuntimeError(f"no equilibrium found for I = {current}")
    roots = [brentq(lambda x: current_of_equilibrium(x) - current, grid[i], grid[i + 1], xtol=1e-14)
             for i in crossings]
    return roots


def jacobian(v):
    w = w_inf(v)
    j11 = (-G_CA * (dm_inf(v) * (v - V_CA) + m_inf(v)) - G_K * w - G_L) / CAP
    j12 = -G_K * (v - V_K) / CAP
    j21 = PHI * dw_inf(v) / tau_w(v)
    j22 = -PHI / tau_w(v)
    return np.array([[j11, j12], [j21, j22]])


# ----------------------------------------------------------------------------
# Hopf point and its criticality
# ----------------------------------------------------------------------------
def hopf_point():
    trace = lambda v: np.trace(jacobian(v))
    v_hopf = brentq(trace, -40.0, -10.0, xtol=1e-14)
    current = current_of_equilibrium(v_hopf)
    omega = float(np.sqrt(np.linalg.det(jacobian(v_hopf))))
    return v_hopf, current, omega


def first_lyapunov_coefficient(v_hopf, current_hopf, omega, step=0.1):
    """Kuznetsov's planar formula, evaluated with central differences."""
    w_hopf = w_inf(v_hopf)
    jac = jacobian(v_hopf)
    eigenvalues, eigenvectors = np.linalg.eig(jac)
    index = int(np.argmax(eigenvalues.imag))
    q = eigenvectors[:, index]
    basis = np.column_stack([q.real, -q.imag])
    basis_inv = np.linalg.inv(basis)

    def field(x, y):
        v = v_hopf + basis[0, 0] * x + basis[0, 1] * y
        w = w_hopf + basis[1, 0] * x + basis[1, 1] * y
        return basis_inv @ np.asarray(rhs(0.0, [v, w], current_hopf))

    def g(component, x, y):
        return field(x, y)[component]

    h = step

    def derivative(component, orders):
        nx, ny = orders
        f = lambda x, y: g(component, x, y)
        if (nx, ny) == (2, 0):
            return (f(h, 0) - 2 * f(0, 0) + f(-h, 0)) / h ** 2
        if (nx, ny) == (0, 2):
            return (f(0, h) - 2 * f(0, 0) + f(0, -h)) / h ** 2
        if (nx, ny) == (1, 1):
            return (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h ** 2)
        if (nx, ny) == (3, 0):
            return (f(2 * h, 0) - 2 * f(h, 0) + 2 * f(-h, 0) - f(-2 * h, 0)) / (2 * h ** 3)
        if (nx, ny) == (0, 3):
            return (f(0, 2 * h) - 2 * f(0, h) + 2 * f(0, -h) - f(0, -2 * h)) / (2 * h ** 3)
        if (nx, ny) == (1, 2):
            return (f(h, h) - 2 * f(h, 0) + f(h, -h) - f(-h, h) + 2 * f(-h, 0) - f(-h, -h)) / (2 * h ** 3)
        if (nx, ny) == (2, 1):
            return (f(h, h) - 2 * f(0, h) + f(-h, h) - f(h, -h) + 2 * f(0, -h) - f(-h, -h)) / (2 * h ** 3)
        raise ValueError(orders)

    fxx, fyy, fxy = derivative(0, (2, 0)), derivative(0, (0, 2)), derivative(0, (1, 1))
    gxx, gyy, gxy = derivative(1, (2, 0)), derivative(1, (0, 2)), derivative(1, (1, 1))
    fxxx, fxyy = derivative(0, (3, 0)), derivative(0, (1, 2))
    gxxy, gyyy = derivative(1, (2, 1)), derivative(1, (0, 3))

    return float(
        (fxxx + fxyy + gxxy + gyyy) / 16.0
        + (fxy * (fxx + fyy) - gxy * (gxx + gyy) - fxx * gxx + fyy * gyy) / (16.0 * omega)
    )


# ----------------------------------------------------------------------------
# Limit-cycle branches
# ----------------------------------------------------------------------------
def cycle_amplitude(current, state, transient=20000.0, window=3000.0, rtol=1e-10, atol=1e-12):
    settled = solve_ivp(rhs, [0.0, transient], state, args=(current,), rtol=rtol, atol=atol)
    end = settled.y[:, -1]
    measured = solve_ivp(rhs, [0.0, window], end, args=(current,), rtol=rtol, atol=atol,
                         t_eval=np.linspace(0.0, window, 30001))
    v = measured.y[0]
    return float(v.max() - v.min()), float(v.max()), float(v.min()), end


def stable_branch(current_start=105.0, current_stop=88.0, step=0.25, threshold=5.0):
    settle = solve_ivp(rhs, [0.0, 20000.0], [0.0, 0.1], args=(current_start,), rtol=1e-10, atol=1e-12)
    state = settle.y[:, -1]
    rows, last_alive, last_state = [], None, None
    for current in np.arange(current_start, current_stop - 1e-9, -step):
        amplitude, v_max, v_min, end = cycle_amplitude(float(current), state, transient=12000.0)
        if amplitude <= threshold:
            break
        state = end
        last_alive, last_state = float(current), end
        rows.append({"current": float(current), "v_max": v_max, "v_min": v_min, "amplitude": amplitude})
    return rows, last_alive, last_state


def locate_fold(last_alive, last_state, step=0.25, iterations=12, threshold=5.0):
    lo, hi, state = last_alive - step, last_alive, last_state
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        amplitude, _, _, end = cycle_amplitude(mid, state, transient=40000.0)
        if amplitude > threshold:
            hi, state = mid, end
        else:
            lo = mid
    return lo, hi


def unstable_branch(currents, threshold=0.5):
    rows = []
    for current in currents:
        v_star = equilibrium(current)[0]
        start = [v_star + 0.5, w_inf(v_star)]
        settled = solve_ivp(rhs_backward, [0.0, 20000.0], start, args=(current,), rtol=1e-10, atol=1e-12)
        end = settled.y[:, -1]
        measured = solve_ivp(rhs_backward, [0.0, 3000.0], end, args=(current,), rtol=1e-10, atol=1e-12,
                             t_eval=np.linspace(0.0, 3000.0, 30001))
        v = measured.y[0]
        amplitude = float(v.max() - v.min())
        if amplitude < threshold:
            continue
        rows.append({"current": float(current), "v_max": float(v.max()), "v_min": float(v.min()),
                     "amplitude": amplitude})
    return rows


# ----------------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    import csv
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path_stem, equilibrium_rows, stable_rows, unstable_rows, current_hopf, fold, working=88.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.2, 3.2))
    currents = [row["current"] for row in equilibrium_rows]
    v_star = [row["v_star"] for row in equilibrium_rows]
    stable = [row["stable"] for row in equilibrium_rows]

    # Equilibrium: solid where stable, dashed where unstable. Black throughout,
    # so the panel survives black-and-white printing.
    segment_c, segment_v, previous = [], [], None
    for c, v, s in zip(currents, v_star, stable):
        if previous is not None and s != previous:
            axis.plot(segment_c, segment_v, color="k",
                      linestyle="-" if previous else "--", linewidth=1.6)
            segment_c, segment_v = [], []
        segment_c.append(c)
        segment_v.append(v)
        previous = s
    axis.plot(segment_c, segment_v, color="k", linestyle="-" if previous else "--", linewidth=1.6)

    for key, marker in (("v_max", "o"), ("v_min", "o")):
        axis.plot([row["current"] for row in stable_rows], [row[key] for row in stable_rows],
                  color="k", linestyle="-", linewidth=2.6, alpha=0.55)
    for key in ("v_max", "v_min"):
        axis.plot([row["current"] for row in unstable_rows], [row[key] for row in unstable_rows],
                  color="k", linestyle=":", linewidth=1.6)

    axis.axvline(fold, color="k", linestyle="-.", linewidth=0.9)
    axis.axvline(current_hopf, color="k", linestyle="-.", linewidth=0.9)
    axis.axvline(working, color="k", linestyle="-", linewidth=0.9)
    axis.annotate("$I=88$", xy=(working, 35), xytext=(working - 9.0, 42),
                  arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    axis.text(fold, -95, "$I_{\\rm fold}$", ha="center", va="top", fontsize=8)
    axis.text(current_hopf, -95, "$I_{\\rm H}$", ha="center", va="top", fontsize=8)

    axis.set_xlabel("$I$")
    axis.set_ylabel("$V$")
    axis.set_xlim(min(currents), max(currents))
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(str(path_stem) + suffix, dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="bifurcation_analysis")
    parser.add_argument("--current-min", type=float, default=70.0)
    parser.add_argument("--current-max", type=float, default=110.0)
    parser.add_argument("--make-figure", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v_hopf, current_hopf, omega = hopf_point()
    lyapunov = first_lyapunov_coefficient(v_hopf, current_hopf, omega)

    print(f"Hopf: V_H = {v_hopf:.6f}, I_H = {current_hopf:.6f}, omega = {omega:.8f}, "
          f"T = {2 * np.pi / omega:.4f}, l1 = {lyapunov:+.6e}")

    stable_rows, last_alive, last_state = stable_branch()
    fold_lo, fold_hi = locate_fold(last_alive, last_state)
    fold = 0.5 * (fold_lo + fold_hi)
    print(f"Fold of limit cycles: I_fold in [{fold_lo:.5f}, {fold_hi:.5f}]")

    unstable_rows = unstable_branch(np.arange(current_hopf - 0.05, fold, -0.1))

    equilibrium_rows = []
    for current in np.arange(args.current_min, args.current_max + 1e-9, 0.1):
        for v_star in equilibrium(float(current)):
            eigenvalues = np.linalg.eigvals(jacobian(v_star))
            equilibrium_rows.append({
                "current": float(current),
                "v_star": float(v_star),
                "re_lambda": float(np.max(eigenvalues.real)),
                "im_lambda": float(np.max(np.abs(eigenvalues.imag))),
                "stable": bool(np.max(eigenvalues.real) < 0.0),
            })

    table = []
    for current in WORKING_CURRENTS:
        v_star = equilibrium(current)[0]
        eigenvalues = np.linalg.eigvals(jacobian(v_star))
        re, im = float(eigenvalues[0].real), float(abs(eigenvalues[0].imag))
        table.append({
            "current": current,
            "v_star": v_star,
            "re_lambda": re,
            "im_lambda": im,
            "damped_period": 2 * np.pi / im if im > 0 else float("nan"),
            "distance_to_hopf": current_hopf - current,
            "relative_distance_to_hopf_percent": 100.0 * (current_hopf - current) / current_hopf,
            "distance_to_fold": fold - current,
            "regime": "monostable excitable" if current < fold else "bistable (rest + cycle)",
        })
        print(f"I = {current:5.1f}: V* = {v_star:9.4f}, lambda = {re:+.6f} +- {im:.6f} i, "
              f"I_H - I = {current_hopf - current:6.3f}, {table[-1]['regime']}")

    summary = {
        "parameters": {"C": CAP, "g_Ca": G_CA, "g_K": G_K, "g_L": G_L, "V_Ca": V_CA, "V_K": V_K,
                       "V_L": V_L, "V1": V1, "V2": V2, "V3": V3, "V4": V4, "phi": PHI},
        "hopf": {"v": v_hopf, "current": current_hopf, "omega": omega,
                 "period": 2 * np.pi / omega, "first_lyapunov_coefficient": lyapunov,
                 "criticality": "subcritical" if lyapunov > 0 else "supercritical"},
        "fold_of_limit_cycles": {"current_lower": fold_lo, "current_upper": fold_hi, "current": fold},
        "bistable_window": [fold, current_hopf],
        "working_point": {
            "current": 88.0,
            "distance_to_hopf": current_hopf - 88.0,
            "relative_distance_to_hopf_percent": 100.0 * (current_hopf - 88.0) / current_hopf,
            "distance_to_fold": fold - 88.0,
        },
    }
    (output_dir / "hopf_distance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "working_point_table.csv", table)
    write_csv(output_dir / "branch_equilibrium.csv", equilibrium_rows)
    write_csv(output_dir / "branch_limit_cycles.csv",
              [dict(row, branch="stable") for row in stable_rows]
              + [dict(row, branch="unstable") for row in unstable_rows])

    if args.make_figure:
        make_figure(output_dir / "fig_bifurcation", equilibrium_rows, stable_rows, unstable_rows,
                    current_hopf, fold)

    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
