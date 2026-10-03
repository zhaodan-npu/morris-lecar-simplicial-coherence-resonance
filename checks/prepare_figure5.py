"""Cross-check Figure 5 summary values against source and restyled PDF paths."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined"
SOURCE = ROOT / "Coherence_Resonance.pdf"
SUMMARY = SUITE / "merged_summary.json"
PANELS = {
    "a": ("cv", .8, "/Fm15", "fig2a_sigma2_sweep_cv_all_alpha.pdf"),
    "b": ("dv", 1.2, "/Fm16", "fig2b_sigma2_sweep_Dv_all_alpha.pdf"),
}
COLORS = {"control": np.array([54, 69, 79]) / 255,
          "higher_order": np.array([198, 40, 40]) / 255}
GRID = np.arange(17) / 100


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    paths = [SOURCE, SUMMARY, ROOT / "plot_formal_prl_figures.py"]
    paths.extend(SUITE / "formal_figures" / spec[3] for spec in PANELS.values())
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths]


def extract_curves(stream, reader, ymax):
    matrix, color, clip, dash = np.eye(3), None, None, []
    stack, path, curves = [], [], {key: [] for key in COLORS}
    pending, curved = None, False
    for args, op in ContentStream(stream, reader).operations:
        if op == b"q":
            stack.append((matrix.copy(), color, clip, dash))
        elif op == b"Q":
            matrix, color, clip, dash = stack.pop()
        elif op == b"cm":
            a, b, c, d, e, f = map(float, args)
            matrix = matrix @ np.array([[a, c, e], [b, d, f], [0, 0, 1.]])
        elif op == b"RG":
            color = np.array(args, dtype=float)
        elif op == b"G":
            color = np.repeat(float(args[0]), 3)
        elif op == b"d":
            dash = list(args[0])
        elif op == b"re":
            x, y, w, h = map(float, args)
            lo, hi = matrix @ [x, y, 1], matrix @ [x + w, y + h, 1]
            pending = (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])
        elif op in (b"W", b"W*"):
            clip = pending
        elif op in (b"m", b"l"):
            if op == b"m":
                path, curved = [], False
            path.append((matrix @ [float(args[0]), float(args[1]), 1])[:2])
        elif op in (b"c", b"v", b"y", b"h"):
            curved = True
        elif op in (b"S", b"B", b"B*", b"f", b"f*", b"n"):
            label = next((key for key, rgb in COLORS.items()
                          if color is not None and np.allclose(color, rgb, atol=1e-7)), None)
            if op == b"S" and label and clip and len(path) == 17 and not curved:
                left, bottom, width, height = clip
                xy = (np.array(path) - [left, bottom]) / [width, height]
                xy[:, 0] = -.005 + .17 * xy[:, 0]
                xy[:, 1] *= ymax
                np.testing.assert_allclose(xy[:, 0], GRID, rtol=0, atol=1e-9)
                assert bool(dash) == (label == "control")
                curves[label].append(xy[:, 1])
            path, curved = [], False
    assert all(len(group) == 2 for group in curves.values())
    # Both source and output draw alpha=1.5 before alpha=2.0.
    return {(label, alpha): curve for label, group in curves.items()
            for alpha, curve in zip((1.5, 2.0), group)}


def expected(rows, label, alpha, metric):
    group = sorted((r for r in rows if r["label"] == label and float(r["alpha"]) == alpha),
                   key=lambda r: float(r["sigma2"]))
    assert len(group) == 17
    np.testing.assert_allclose([float(r["sigma2"]) for r in group], GRID, rtol=0, atol=1e-12)
    return np.array([float(r[metric]) for r in group])


def prepare():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for sigma in GRID:
        name = f"sigma2_{sigma:.2f}"
        item = summary[name]
        assert item["parameters"]["sigma_2"] == sigma
        for alpha in (1.5, 2.0):
            for label in COLORS:
                best = item["best_by_alpha"][str(alpha)][label]
                rows.append({"experiment": name, "sigma2": sigma, "alpha": alpha, "label": label,
                             "cv": best["best_cv_value"], "dv": best["best_noise_intensity_dv"],
                             "mean_spike_count_at_optimum": best["best_spike_count"]})
    reader = PdfReader(SOURCE)
    forms = reader.pages[10]["/Resources"]["/XObject"]
    checks = []
    for panel, (metric, ymax, form_name, original_name) in PANELS.items():
        form = forms[form_name].get_object()
        original = PdfReader(SUITE / "formal_figures" / original_name)
        assert form.get_data() == original.pages[0].get_contents().get_data()
        curves = extract_curves(form, reader, ymax)
        max_error = 0.
        for (label, alpha), curve in curves.items():
            values = expected(rows, label, alpha, metric)
            np.testing.assert_allclose(curve, values, rtol=0, atol=ymax * 1e-8)
            max_error = max(max_error, float(np.max(np.abs(curve - values))))
        checks.append({"panel": panel, "n_values": 68, "source_form": form_name,
                       "source_form_sha256": hashlib.sha256(form.get_data()).hexdigest(),
                       "max_abs_error_source_pdf": max_error})
    (HERE / "analysis").mkdir(parents=True, exist_ok=True)
    with (HERE / "analysis" / "plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "analysis" / "manifest.json").write_text(json.dumps(
        {"inputs": inputs(), "panels": checks, "n_values": 136,
         "selection": "Use the original merged_summary best_by_alpha values without reselecting optima.",
         "error_bars_in_original": False, "simulations_run": False}, indent=2), encoding="utf-8")
    print("All 136 plotted source values match the original summary; original PDF forms match exactly.")


def verify():
    manifest = json.loads((HERE / "analysis" / "manifest.json").read_text())
    assert manifest["inputs"] == inputs()
    with (HERE / "analysis" / "plot_data.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results = []
    for panel, (metric, ymax, _, _) in PANELS.items():
        path = HERE / f"fig5{panel}.pdf"
        reader = PdfReader(path)
        assert len(reader.pages) == 1
        page = reader.pages[0]
        np.testing.assert_allclose([float(page.mediabox.width), float(page.mediabox.height)],
                                   [5.6 * 72, 4.2 * 72], rtol=0, atol=1e-7)
        curves = extract_curves(page.get_contents(), reader, ymax)
        max_error = 0.
        for (label, alpha), curve in curves.items():
            values = expected(rows, label, alpha, metric)
            np.testing.assert_allclose(curve, values, rtol=0, atol=ymax * 1e-8)
            max_error = max(max_error, float(np.max(np.abs(curve - values))))
        fonts = []
        for obj in page["/Resources"]["/Font"].values():
            font = obj.get_object()
            name = str(font["/BaseFont"])
            assert "TimesNewRoman" in name, name
            for descendant in font["/DescendantFonts"]:
                desc = descendant.get_object()["/FontDescriptor"]
                assert "/FontFile2" in desc
            fonts.append(name)
        text = page.extract_text()
        assert "Pairwise" in text and "Simplicial" in text
        assert not any(f"({letter})" in text for letter in "ab")
        results.append({"file": path.name, "sha256": sha(path), "values_checked": 68,
                        "max_abs_error_output_pdf": max_error, "embedded_fonts": fonts})
    (HERE / "analysis" / "final_verification.json").write_text(json.dumps(
        {"inputs_unchanged": True, "panels": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else verify()
