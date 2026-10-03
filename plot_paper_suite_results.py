from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def render_metrics_plot(path: Path, aggregated_rows: list[dict], title: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    color_map = {"control": "#455a64", "higher_order": "#c62828"}
    alphas = sorted({row["alpha"] for row in aggregated_rows})

    for col_idx, alpha in enumerate(alphas[:2]):
        subset_alpha = [row for row in aggregated_rows if row["alpha"] == alpha]
        for label in ("control", "higher_order"):
            rows = [row for row in subset_alpha if row["label"] == label]
            rows.sort(key=lambda row: row.get("noise_intensity_dv", (row["noise_scale"] / 20.0) ** row["alpha"]))
            x = [row.get("noise_intensity_dv", (row["noise_scale"] / 20.0) ** row["alpha"]) for row in rows]
            cv = [np.nan if row["isi_cv_mean"] is None else row["isi_cv_mean"] for row in rows]
            cv_err = [0.0 if row["isi_cv_std"] is None else row["isi_cv_std"] for row in rows]
            spikes = [row["spike_count_total_mean"] for row in rows]
            spike_err = [0.0 if row["spike_count_total_std"] is None else row["spike_count_total_std"] for row in rows]

            axes[0, col_idx].errorbar(x, cv, yerr=cv_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)
            axes[1, col_idx].errorbar(x, spikes, yerr=spike_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)

        axes[0, col_idx].set_title(f"alpha = {alpha}")
        axes[0, col_idx].set_ylabel("ISI CV")
        axes[1, col_idx].set_ylabel("Spike count")
        axes[1, col_idx].set_xlabel(r"$D_v$")
        axes[0, col_idx].grid(alpha=0.3)
        axes[1, col_idx].grid(alpha=0.3)
        axes[0, col_idx].legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render local plots from cluster CSV outputs.")
    parser.add_argument("output_root", nargs="?", default="paper_suite_true_simplicial_array_prl")
    args = parser.parse_args()

    root = Path(args.output_root)
    csv_paths = sorted(root.glob("*/aggregated_metrics.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No aggregated_metrics.csv files found under {root}")

    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        out_path = csv_path.parent / "metrics.png"
        render_metrics_plot(out_path, rows, csv_path.parent.name)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
