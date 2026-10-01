from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_sigma2(path: Path):
    return float(path.name.split("_", 1)[1])


def row_dv(row):
    if row.get("noise_intensity_dv", "") not in ("", "None", None):
        return float(row["noise_intensity_dv"])
    return noise_intensity_dv(float(row["noise_scale"]), float(row["alpha"]))


def value(rows, alpha: float, noise_intensity_dv: float, label: str, metric: str):
    for row in rows:
        if (
            abs(float(row["alpha"]) - alpha) < 1e-12
            and abs(row_dv(row) - noise_intensity_dv) < 1e-12
            and row["label"] == label
        ):
            raw = row.get(metric, "")
            return np.nan if raw in ("", "None") else float(raw)
    return np.nan


def noise_intensity_dv(noise_scale: float, alpha: float, capacitance: float = 20.0) -> float:
    return float((noise_scale / capacitance) ** alpha)


def edges_from_centers(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def log_edges_from_centers(centers):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] / np.sqrt(centers[1] / centers[0])
    edges[-1] = centers[-1] * np.sqrt(centers[-1] / centers[-2])
    return edges


def collect(root: Path, min_spikes: float):
    sigma_dirs = sorted([p for p in root.glob("sigma2_*") if (p / "aggregated_metrics.csv").exists()], key=parse_sigma2)
    if not sigma_dirs:
        raise FileNotFoundError(f"No merged sigma2_* directories with aggregated_metrics.csv found under {root}")

    sigma2_values = [parse_sigma2(p) for p in sigma_dirs]
    first_rows = read_csv(sigma_dirs[0] / "aggregated_metrics.csv")
    noise_intensity_dv_values = sorted({row_dv(row) for row in first_rows})
    alphas = sorted({float(row["alpha"]) for row in first_rows})

    data = {}
    summary_rows = []
    for alpha in alphas:
        cv_control = np.full((len(sigma2_values), len(noise_intensity_dv_values)), np.nan)
        cv_ho = np.full_like(cv_control, np.nan)
        delta = np.full_like(cv_control, np.nan)
        spike_ho = np.full_like(cv_control, np.nan)

        for si, sigma_dir in enumerate(sigma_dirs):
            rows = read_csv(sigma_dir / "aggregated_metrics.csv")
            for ni, noise_intensity_dv in enumerate(noise_intensity_dv_values):
                c = value(rows, alpha, noise_intensity_dv, "control", "isi_cv_mean")
                h = value(rows, alpha, noise_intensity_dv, "higher_order", "isi_cv_mean")
                sp = value(rows, alpha, noise_intensity_dv, "higher_order", "spike_count_total_mean")
                cv_control[si, ni] = c
                spike_ho[si, ni] = sp
                if np.isfinite(h) and np.isfinite(sp) and sp >= min_spikes:
                    cv_ho[si, ni] = h
                if np.isfinite(c) and np.isfinite(h) and np.isfinite(sp) and sp >= min_spikes:
                    delta[si, ni] = c - h

            valid = np.isfinite(cv_ho[si]) & np.isfinite(spike_ho[si]) & (spike_ho[si] >= min_spikes)
            if valid.any():
                valid_indices = np.where(valid)[0]
                best_idx = valid_indices[np.nanargmin(cv_ho[si, valid])]
                best_dv = noise_intensity_dv_values[best_idx]
                best_ho = cv_ho[si, best_idx]
                best_control_at_same_noise = cv_control[si, best_idx]
            else:
                best_dv = np.nan
                best_ho = np.nan
                best_control_at_same_noise = np.nan

            valid_delta = np.isfinite(delta[si])
            if valid_delta.any():
                ridge_idx = np.where(valid_delta)[0][np.nanargmax(delta[si, valid_delta])]
                ridge_dv = noise_intensity_dv_values[ridge_idx]
                ridge_delta = delta[si, ridge_idx]
            else:
                ridge_dv = np.nan
                ridge_delta = np.nan

            summary_rows.append(
                {
                    "alpha": alpha,
                    "sigma_2": sigma2_values[si],
                    "best_higher_order_Dv": best_dv,
                    "best_higher_order_cv": best_ho,
                    "control_cv_at_ho_best_Dv": best_control_at_same_noise,
                    "max_delta_cv_Dv": ridge_dv,
                    "max_delta_cv": ridge_delta,
                }
            )

        data[alpha] = {
            "cv_control": cv_control,
            "cv_higher_order": cv_ho,
            "delta_cv": delta,
            "spike_higher_order": spike_ho,
        }

    return sigma2_values, noise_intensity_dv_values, alphas, data, summary_rows


def write_summary_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sparse_tick_positions(values: list[float], keep: set[float]):
    positions = []
    labels = []
    for idx, value in enumerate(values):
        if value in keep:
            positions.append(idx + 0.5)
            labels.append(f"{value:g}" if value >= 1 else f"{value:.2f}")
    return positions, labels


def sparse_value_ticks(values, keep_indices: set[int]):
    positions = []
    labels = []
    for idx, value in enumerate(values):
        if idx in keep_indices:
            positions.append(value)
            labels.append(f"{value:.2g}")
    return positions, labels


def dv_ticks(values):
    values = list(values)
    keep_indices = {0, 6, 11, 17, len(values) - 1}
    positions = []
    labels = []
    for idx, value in enumerate(values):
        if idx in keep_indices:
            positions.append(value)
            if value < 0.1:
                labels.append(f"{value:.2f}")
            elif value < 1:
                labels.append(f"{value:.2g}")
            else:
                labels.append(f"{value:g}")
    return positions, labels


def apply_prl_style(plt):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def save_prl_panel(
    plt,
    out_dir: Path,
    name: str,
    matrix: np.ndarray,
    cmap: str,
    vmin,
    vmax,
    x_edges,
    y_edges,
    x_tick_pos,
    x_tick_labels,
    y_tick_pos,
    y_tick_labels,
    colorbar_label: str,
    ridge_x=None,
    ridge_y=None,
    ridge_style="w.-",
):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.9), constrained_layout=True)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#eeeeee")
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    if ridge_x is not None and ridge_y is not None:
        ax.plot(ridge_x, ridge_y, ridge_style, lw=2.0, ms=5.0, markeredgewidth=1.0)
    ax.set_xscale("log")
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(r"$\sigma_2$")
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.03)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(which="both", direction="in", width=1.3, length=5)
    cbar.outline.set_linewidth(1.2)

    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


def main():
    parser = argparse.ArgumentParser(description="Plot sigma2-D_v CV phase diagram from merged chunks.")
    parser.add_argument("--output-root", default="paper_suite_sigma2_Dv_phase_refined_N100")
    parser.add_argument("--min-spikes", type=float, default=100.0)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    apply_prl_style(plt)

    root = Path(args.output_root)
    sigma2_values, noise_intensity_dv_values, alphas, data, summary_rows = collect(root, args.min_spikes)

    out_dir = root / "phase_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(out_dir / "sigma2_noise_phase_summary.csv", summary_rows)
    (out_dir / "sigma2_noise_phase_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    y_edges = np.arange(len(sigma2_values) + 1)
    y_centers = np.arange(len(sigma2_values)) + 0.5
    y_tick_pos, y_tick_labels = sparse_tick_positions(sigma2_values, {0.0, 0.04, 0.08, 0.12, 0.16})
    prl_dir = out_dir / "prl_panels"
    prl_dir.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        x_edges = log_edges_from_centers(noise_intensity_dv_values)
        x_tick_pos, x_tick_labels = dv_ticks(noise_intensity_dv_values)
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
        panels = [
            ("Higher-order CV", data[alpha]["cv_higher_order"], "viridis_r", 0.2, 1.1),
            ("Control - higher-order CV", data[alpha]["delta_cv"], "coolwarm", -0.1, 0.5),
            ("Higher-order spike count", data[alpha]["spike_higher_order"], "magma", None, None),
        ]
        for ax, (title, matrix, cmap, vmin, vmax) in zip(axes, panels):
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad("#eeeeee")
            im = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(matrix), cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="flat")
            ax.set_title(title)
            ax.set_xticks(x_tick_pos, x_tick_labels)
            ax.set_yticks(y_tick_pos, y_tick_labels)
            ax.set_xlabel(r"$D_v$")
            ax.set_ylabel(r"$\sigma_2$")
            ax.set_xscale("log")
            fig.colorbar(im, ax=ax)

        summary_alpha = [row for row in summary_rows if abs(row["alpha"] - alpha) < 1e-12]
        ridge_x = [row["best_higher_order_Dv"] for row in summary_alpha]
        ridge_y = list(y_centers)
        axes[0].plot(ridge_x, ridge_y, "w.-", lw=1.5, ms=5)
        axes[1].plot(ridge_x, ridge_y, "k.-", lw=1.5, ms=5)

        fig.suptitle(rf"$\alpha={alpha}$: $\sigma_2$-$D_v$ phase diagram")
        png = out_dir / f"sigma2_noise_phase_alpha_{alpha:g}_legacy_composite.png"
        pdf = out_dir / f"sigma2_noise_phase_alpha_{alpha:g}_legacy_composite.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

        alpha_tag = f"{alpha:g}".replace(".", "p")
        save_prl_panel(
            plt,
            out_dir,
            f"sigma2_noise_phase_alpha_{alpha:g}",
            data[alpha]["cv_higher_order"],
            "viridis_r",
            0.2,
            1.1,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            "CV",
            ridge_x=ridge_x,
            ridge_y=ridge_y,
            ridge_style="w.-",
        )
        save_prl_panel(
            plt,
            out_dir,
            f"sigma2_noise_phase_alpha_{alpha:g}_cv_prlstyle",
            data[alpha]["cv_higher_order"],
            "viridis_r",
            0.2,
            1.1,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            "CV",
            ridge_x=ridge_x,
            ridge_y=ridge_y,
            ridge_style="w.-",
        )
        save_prl_panel(
            plt,
            out_dir,
            f"sigma2_noise_phase_alpha_{alpha:g}_delta_cv_prlstyle",
            data[alpha]["delta_cv"],
            "coolwarm",
            -0.1,
            0.5,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            r"$\Delta CV$",
            ridge_x=ridge_x,
            ridge_y=ridge_y,
            ridge_style="k.-",
        )
        save_prl_panel(
            plt,
            out_dir,
            f"sigma2_noise_phase_alpha_{alpha:g}_spikes_prlstyle",
            data[alpha]["spike_higher_order"],
            "magma",
            None,
            None,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            "Spike count",
        )

        save_prl_panel(
            plt,
            prl_dir,
            f"sigma2_noise_alpha_{alpha_tag}_cv",
            data[alpha]["cv_higher_order"],
            "viridis_r",
            0.2,
            1.1,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            "CV",
            ridge_x=ridge_x,
            ridge_y=ridge_y,
            ridge_style="w.-",
        )
        save_prl_panel(
            plt,
            prl_dir,
            f"sigma2_noise_alpha_{alpha_tag}_delta_cv",
            data[alpha]["delta_cv"],
            "coolwarm",
            -0.1,
            0.5,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            r"$\Delta CV$",
            ridge_x=ridge_x,
            ridge_y=ridge_y,
            ridge_style="k.-",
        )
        save_prl_panel(
            plt,
            prl_dir,
            f"sigma2_noise_alpha_{alpha_tag}_spikes",
            data[alpha]["spike_higher_order"],
            "magma",
            None,
            None,
            x_edges,
            y_edges,
            x_tick_pos,
            x_tick_labels,
            y_tick_pos,
            y_tick_labels,
            "Spike count",
        )

    print(f"Wrote summary to {out_dir}")


if __name__ == "__main__":
    main()
