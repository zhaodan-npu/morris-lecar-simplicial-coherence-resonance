from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite


def load_aggregated(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_dv(row):
    if row.get("noise_intensity_dv", "") not in ("", "None", None):
        return float(row["noise_intensity_dv"])
    return noise_intensity_dv(float(row["noise_scale"]), float(row["alpha"]))


def lookup(rows, alpha: float, noise_intensity_dv: float, label: str, metric: str):
    for row in rows:
        if abs(float(row["alpha"]) - alpha) < 1e-12 and abs(row_dv(row) - noise_intensity_dv) < 1e-12 and row["label"] == label:
            return float(row[metric]) if row[metric] not in ("", "None") else np.nan
    return np.nan


def noise_intensity_dv(noise_scale: float, alpha: float, capacitance: float = 20.0) -> float:
    return float((noise_scale / capacitance) ** alpha)


def apply_prl_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax):
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def save_figure(fig, pdf_path: Path, png_path: Path):
    try:
        fig.savefig(pdf_path)
    except PermissionError:
        pdf_path = pdf_path.with_name(f"{pdf_path.stem}_colorbar{pdf_path.suffix}")
        fig.savefig(pdf_path)
    try:
        fig.savefig(png_path, dpi=600)
    except PermissionError:
        png_path = png_path.with_name(f"{png_path.stem}_colorbar{png_path.suffix}")
        fig.savefig(png_path, dpi=600)
    plt.close(fig)


def case_label(letter: str, alpha: float, noise_intensity_dv: float) -> str:
    return rf"({letter}) $\alpha={alpha}$, $D_v={noise_intensity_dv:.2g}$"


def simulate_node_panel(params, network, alpha: float, scale: float, seed: int):
    n_steps = int(params.t_total / params.dt)
    keep_every = max(1, int(0.5 / params.dt))
    n = params.n_nodes

    init_rng = np.random.default_rng(seed + base.stable_seed("panel_init", alpha, scale))
    noise_rng = np.random.default_rng(seed + base.stable_seed("panel_noise", alpha, scale))
    v = -60.0 + 2.0 * init_rng.standard_normal((1, n))
    w = base.w_inf(v, params)
    prev_v = v.copy()
    operator = network.pairwise_operator + params.sigma_2 * network.l2

    last_spike = np.full(n, np.nan)
    isi = [[] for _ in range(n)]
    spike_count = np.zeros(n, dtype=int)
    times = []
    snapshots = []
    snapshot_v = None

    for step in range(n_steps):
        coupling = v @ operator.T
        i_ca = params.g_ca * base.m_inf(v, params) * (v - params.v_ca)
        i_k = params.g_k * w * (v - params.v_k)
        i_l = params.g_l * (v - params.v_l)
        dv = (params.current - i_ca - i_k - i_l + coupling) / params.capacitance
        dw = params.phi * (base.w_inf(v, params) - w) / np.maximum(base.tau_w(v, params), 1e-6)
        jump = base.levy_increment(scale, alpha, params.beta, params.dt, (1, n), noise_rng) / params.capacitance
        v = np.clip(v + params.dt * dv + jump, -200.0, 200.0)
        w = np.clip(w + params.dt * dw, -0.5, 1.5)

        t = (step + 1) * params.dt
        if t > params.t_warmup:
            crossing = (prev_v[0] < params.spike_threshold) & (v[0] >= params.spike_threshold)
            nodes = np.where(crossing)[0]
            for node in nodes.tolist():
                if np.isfinite(last_spike[node]):
                    isi[node].append(t - last_spike[node])
                last_spike[node] = t
                spike_count[node] += 1

            if step % keep_every == 0:
                times.append(t - params.t_warmup)
                snapshots.append(v[0].copy())

            if snapshot_v is None and t >= params.t_warmup + 0.5 * (params.t_total - params.t_warmup):
                snapshot_v = v[0].copy()

        prev_v = v.copy()

    if snapshot_v is None:
        snapshot_v = v[0].copy()

    per_node_cv = np.full(n, np.nan)
    for idx, intervals in enumerate(isi):
        if len(intervals) >= 3:
            arr = np.asarray(intervals)
            per_node_cv[idx] = float(np.std(arr) / (np.mean(arr) + 1e-12))

    window = params.t_total - params.t_warmup
    frequency = spike_count / (window / 1000.0)
    return np.asarray(times), np.asarray(snapshots), snapshot_v, per_node_cv, frequency


def main():
    parser = argparse.ArgumentParser(description="Reference-style alpha/D_v panel: time-node map, V_i snapshot, node CV.")
    parser.add_argument("--result-root", default="paper_suite_true_simplicial_array_prl_N100_Dv_refined")
    parser.add_argument("--output-dir", default="paper_suite_true_simplicial_array_prl_N100_Dv_refined/formal_figures")
    parser.add_argument("--trace-total", type=float, default=1400.0)
    parser.add_argument("--trace-warmup", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    rows = load_aggregated(result_root / "candidate39_main" / "aggregated_metrics.csv")
    config = suite.PaperSuiteConfig(
        n_nodes=100,
        n_networks=1,
        n_trials_per_network=1,
        t_total=args.trace_total,
        t_warmup=args.trace_warmup,
        base_seed=args.seed,
    )
    params = suite.candidate39_params(config)
    network = base.build_simplicial_network(params, np.random.default_rng(params.base_seed + base.stable_seed("paper_network", 0)))

    cases = [
        (1.5, 0.12, "weak-noise"),
        (1.5, 0.25, "low-CV window"),
        (1.5, 0.75, "stronger-noise"),
        (2.0, 0.25, "low-CV window"),
        (2.0, 1.00, "stronger-noise"),
    ]

    apply_prl_style()

    fig, axes = plt.subplots(3, len(cases), figsize=(13.0, 5.7), gridspec_kw={"height_ratios": [1.05, 0.72, 0.72]})
    fig.subplots_adjust(left=0.055, right=0.905, bottom=0.105, top=0.965, hspace=0.23, wspace=0.18)
    letters = "abcde"
    im = None
    x = np.arange(1, params.n_nodes + 1)

    for col, (alpha, noise_intensity_dv, note) in enumerate(cases):
        scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
        times, voltage_frames, snapshot_v, per_node_cv, frequency = simulate_node_panel(params, network, alpha, scale, args.seed)
        prod_ho_cv = lookup(rows, alpha, noise_intensity_dv, "higher_order", "isi_cv_mean")
        prod_ho_spike = lookup(rows, alpha, noise_intensity_dv, "higher_order", "spike_count_total_mean")

        ax = axes[0, col]
        im = ax.imshow(
            voltage_frames,
            aspect="auto",
            origin="lower",
            extent=[1, params.n_nodes, times[0], times[-1]],
            cmap="turbo",
            vmin=-45,
            vmax=35,
            interpolation="nearest",
        )
        if col == 0:
            ax.set_ylabel(r"$t$")
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        style_axis(ax)

        ax = axes[1, col]
        ax.plot(x, snapshot_v, ".", color="#c62828", ms=2.8)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_ylim(-50, 35)
        if col == 0:
            ax.set_ylabel(r"$V_i$")
        else:
            ax.set_yticks([])
        ax.set_xticks([])
        style_axis(ax)

        ax = axes[2, col]
        finite = np.isfinite(per_node_cv)
        ax.plot(x[finite], per_node_cv[finite], ".", color="black", ms=2.4)
        ax.axhline(prod_ho_cv, color="#c62828", lw=1.3)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel(r"$i$")
        if col == 0:
            ax.set_ylabel(r"$CV_i$")
        else:
            ax.set_yticks([])
        ax.text(
            0.04,
            0.88,
            rf"$\overline{{CV}}={prod_ho_cv:.2f}$" + "\n" + rf"$N_{{\rm sp}}={prod_ho_spike/1000:.1f}\times 10^3$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
        )
        style_axis(ax)

    cax = fig.add_axes([0.92, 0.72, 0.014, 0.18])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$V$")
    cb.ax.tick_params(direction="in", width=1.0, length=4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig_alpha_noise_cv_reference_style.png"
    pdf = out_dir / "fig_alpha_noise_cv_reference_style.pdf"
    save_figure(fig, pdf, png)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

    for col, (alpha, noise_intensity_dv, note) in enumerate(cases):
        scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
        times, voltage_frames, snapshot_v, per_node_cv, frequency = simulate_node_panel(params, network, alpha, scale, args.seed)
        prod_ho_cv = lookup(rows, alpha, noise_intensity_dv, "higher_order", "isi_cv_mean")
        prod_ho_spike = lookup(rows, alpha, noise_intensity_dv, "higher_order", "spike_count_total_mean")

        subfig, subaxes = plt.subplots(
            3,
            1,
            figsize=(3.0, 5.2),
            gridspec_kw={"height_ratios": [1.05, 0.72, 0.72]},
        )
        subfig.subplots_adjust(left=0.23, right=0.84, top=0.96, bottom=0.10, hspace=0.26)

        ax = subaxes[0]
        sub_im = ax.imshow(
            voltage_frames,
            aspect="auto",
            origin="lower",
            extent=[1, params.n_nodes, times[0], times[-1]],
            cmap="turbo",
            vmin=-45,
            vmax=35,
            interpolation="nearest",
        )
        ax.set_ylabel(r"$t$")
        ax.set_xticks([])
        style_axis(ax)
        sub_cax = subfig.add_axes([0.88, 0.72, 0.035, 0.20])
        sub_cb = subfig.colorbar(sub_im, cax=sub_cax)
        sub_cb.set_label(r"$V$")
        sub_cb.ax.tick_params(direction="in", width=1.0, length=4)

        ax = subaxes[1]
        ax.plot(x, snapshot_v, ".", color="#c62828", ms=2.8)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_ylim(-50, 35)
        ax.set_ylabel(r"$V_i$")
        ax.set_xticks([])
        style_axis(ax)

        ax = subaxes[2]
        finite = np.isfinite(per_node_cv)
        ax.plot(x[finite], per_node_cv[finite], ".", color="black", ms=2.4)
        ax.axhline(prod_ho_cv, color="#c62828", lw=1.3)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel(r"$i$")
        ax.set_ylabel(r"$CV_i$")
        ax.text(
            0.05,
            0.88,
            rf"$\overline{{CV}}={prod_ho_cv:.2f}$" + "\n" + rf"$N_{{\rm sp}}={prod_ho_spike/1000:.1f}\times 10^3$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
        )
        style_axis(ax)

        sub_pdf = out_dir / f"fig_alpha_noise_cv_reference_style_{letters[col]}.pdf"
        sub_png = out_dir / f"fig_alpha_noise_cv_reference_style_{letters[col]}.png"
        save_figure(subfig, sub_pdf, sub_png)
        print(f"Wrote {sub_pdf}")
        print(f"Wrote {sub_png}")

        single_specs = [
            ("top", (3.1, 2.0)),
            ("middle", (3.1, 1.7)),
            ("bottom", (3.1, 1.7)),
        ]
        for panel_name, fig_size in single_specs:
            one_fig, one_ax = plt.subplots(1, 1, figsize=fig_size)
            one_fig.subplots_adjust(left=0.24, right=0.96, top=0.94, bottom=0.22)

            if panel_name == "top":
                one_im = one_ax.imshow(
                    voltage_frames,
                    aspect="auto",
                    origin="lower",
                    extent=[1, params.n_nodes, times[0], times[-1]],
                    cmap="turbo",
                    vmin=-45,
                    vmax=35,
                    interpolation="nearest",
                )
                one_ax.set_ylabel(r"$t$")
                one_ax.set_xlabel(r"$i$")
                one_fig.subplots_adjust(left=0.24, right=0.82, top=0.94, bottom=0.22)
                one_cax = one_fig.add_axes([0.86, 0.34, 0.035, 0.45])
                one_cb = one_fig.colorbar(one_im, cax=one_cax)
                one_cb.set_label(r"$V$")
                one_cb.ax.tick_params(direction="in", width=1.0, length=4)
            elif panel_name == "middle":
                one_ax.plot(x, snapshot_v, ".", color="#c62828", ms=2.8)
                one_ax.axhline(0, color="#777777", lw=0.8)
                one_ax.set_ylim(-50, 35)
                one_ax.set_ylabel(r"$V_i$")
                one_ax.set_xlabel(r"$i$")
            else:
                finite = np.isfinite(per_node_cv)
                one_ax.plot(x[finite], per_node_cv[finite], ".", color="black", ms=2.4)
                one_ax.axhline(prod_ho_cv, color="#c62828", lw=1.3)
                one_ax.set_ylim(0, 1.25)
                one_ax.set_xlabel(r"$i$")
                one_ax.set_ylabel(r"$CV_i$")
                one_ax.text(
                    0.05,
                    0.88,
                    rf"$\overline{{CV}}={prod_ho_cv:.2f}$" + "\n" + rf"$N_{{\rm sp}}={prod_ho_spike/1000:.1f}\times 10^3$",
                    transform=one_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="#333333",
                )
            style_axis(one_ax)

            single_pdf = out_dir / f"fig_alpha_noise_cv_reference_style_{letters[col]}_{panel_name}.pdf"
            single_png = out_dir / f"fig_alpha_noise_cv_reference_style_{letters[col]}_{panel_name}.png"
            save_figure(one_fig, single_pdf, single_png)
            print(f"Wrote {single_pdf}")
            print(f"Wrote {single_png}")


if __name__ == "__main__":
    main()
