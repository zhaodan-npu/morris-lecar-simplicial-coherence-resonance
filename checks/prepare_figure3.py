"""Read the manuscript's vector curves; cross-check CSV data and final PDFs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader, PageObject
from pypdf.generic import ContentStream, NameObject

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "Coherence_Resonance.pdf"
CSV = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined" / "candidate39_main" / "aggregated_metrics.csv"
COLORS = {"control": np.array([54, 69, 79]) / 255,
          "higher_order": np.array([198, 40, 40]) / 255}
GRID = np.array([.04, .06, .09, .12, .16, .2, .25, .3, .4, .56, .75, 1,
                 1.25, 1.56, 1.89, 2.25, 3.06, 4, 6.25, 9])
SPECS = [
    ("a", "/Fm7", 1.5, "isi_cv", 2.1, 1.),
    ("b", "/Fm8", 2., "isi_cv", 2.1, 1.),
    ("c", "/Fm9", 1.5, "spike_count_total", 60000., 1000.),
    ("d", "/Fm10", 2., "spike_count_total", 60000., 1000.),
    ("e", "/Fm11", 1.5, "tcor", 1.4, 1.),
    ("f", "/Fm12", 2., "tcor", 4.2, 1.),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes():
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in (SOURCE, CSV)]


def extract(stream, reader, y_max):
    matrix, color, clip = np.eye(3), None, None
    stack, path, curves, bars, verticals = [], [], {}, {}, {}
    curved = False
    pending = None
    for args, op in ContentStream(stream, reader).operations:
        if op == b"q":
            stack.append((matrix.copy(), color, clip))
        elif op == b"Q":
            matrix, color, clip = stack.pop()
        elif op == b"cm":
            a, b, c, d, e, f = map(float, args)
            matrix = matrix @ np.array([[a, c, e], [b, d, f], [0, 0, 1.]])
        elif op == b"RG":
            color = np.array(args, dtype=float)
        elif op == b"G":
            color = np.repeat(float(args[0]), 3)
        elif op == b"re":
            x, y, w, h = map(float, args)
            lo = matrix @ [x, y, 1.]
            hi = matrix @ [x + w, y + h, 1.]
            pending = (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])
        elif op in (b"W", b"W*"):
            clip = pending
        elif op in (b"m", b"l"):
            if op == b"m":
                path = []
                curved = False
            path.append((matrix @ [float(args[0]), float(args[1]), 1.])[:2])
        elif op in (b"c", b"v", b"y", b"h"):
            curved = True
        elif op in (b"S", b"B", b"B*", b"f", b"f*", b"n"):
            label = next((key for key, rgb in COLORS.items()
                          if color is not None and np.allclose(color, rgb, atol=1e-7)), None)
            if op == b"S" and label and clip and path and not curved:
                xy = np.array(path)
                left, bottom, width, height = clip
                xy = (xy - [left, bottom]) / [width, height]
                xy[:, 0] = .035 * (10.5 / .035) ** xy[:, 0]
                xy[:, 1] *= y_max
                if len(xy) == 20:
                    assert label not in curves
                    np.testing.assert_allclose(xy[:, 0], GRID, rtol=2e-7)
                    curves[label] = xy
                elif len(xy) == 2 and abs(xy[0, 0] - xy[1, 0]) < 1e-8:
                    if abs(xy[1, 1] - xy[0, 1]) > .999 * y_max:
                        verticals[label] = float(xy[0, 0])
                    else:
                        bars.setdefault(label, []).append([xy[0, 0], xy[:, 1].min(), xy[:, 1].max()])
            path, curved = [], False
    result = {}
    for label in COLORS:
        xy = curves[label]
        eb = np.array(bars[label])
        assert eb.shape == (20, 3), (label, eb.shape)
        np.testing.assert_allclose(eb[:, 0], GRID, rtol=2e-7)
        tolerance = y_max * 2e-8
        assert np.all(eb[:, 1] <= xy[:, 1] + tolerance)
        assert np.all(eb[:, 2] >= xy[:, 1] - tolerance)
        result[label] = {"mean": xy[:, 1], "lower": eb[:, 1], "upper": eb[:, 2],
                         "optimal_dv": verticals[label]}
    return result


def verify_axis_labels(form):
    page = PageObject.create_blank_page(width=374.4, height=280.8)
    page[NameObject("/Resources")] = form["/Resources"]
    page[NameObject("/Contents")] = form
    labels = []
    def visitor(text, cm, tm, font, size):
        token = text.strip()
        if cm[:4] == [1., 0., 0., 1.] and cm[4] < 70 and cm[5] > 40:
            try:
                labels.append((float(token), cm[5]))
            except ValueError:
                pass
    page.extract_text(visitor_text=visitor)
    return labels


def prepare():
    (HERE / "analysis").mkdir(parents=True, exist_ok=True)
    reader = PdfReader(SOURCE)
    forms = reader.pages[8]["/Resources"]["/XObject"]
    with CSV.open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    rows, specs = [], []
    for letter, form_name, alpha, metric, limit, scale in SPECS:
        form = forms[form_name].get_object()
        graphic = extract(form, reader, limit)
        if metric == "tcor":
            labels = verify_axis_labels(form)
            values, positions = np.array(labels).T
            slope = np.polyfit(values, positions, 1)[0]
            np.testing.assert_allclose(219.024 / slope, limit, atol=1e-9)
        optima = {}
        max_csv_error = 0.
        for label in COLORS:
            g = graphic[label]
            group = sorted((r for r in original if float(r["alpha"]) == alpha and r["label"] == label),
                           key=lambda r: float(r["noise_intensity_dv"]))
            assert len(group) == 20
            eligible = [r for r in group if float(r["spike_count_total_mean"]) >= 1000]
            best = min(eligible, key=lambda r: float(r["isi_cv_mean"]))
            optimum = float(best["noise_intensity_dv"])
            np.testing.assert_allclose(g["optimal_dv"], optimum, rtol=2e-7)
            optima[label] = {"dv": optimum, "cv": float(best["isi_cv_mean"])}
            if metric == "tcor":
                mean, lower, upper = g["mean"], g["lower"], g["upper"]
                source_kind = "manuscript_pdf_vector_coordinates"
                error_kind = "original_plotted_error_bar_endpoints"
            else:
                mean = np.array([float(r[f"{metric}_mean"]) for r in group])
                sem = np.array([float(r[f"{metric}_std"]) / np.sqrt(float(r["total_realizations"]))
                                for r in group])
                lower, upper = mean - sem, mean + sem
                for key, expected in (("mean", mean), ("lower", lower), ("upper", upper)):
                    error = float(np.max(np.abs(g[key] - expected)))
                    max_csv_error = max(error, max_csv_error)
                    np.testing.assert_allclose(g[key], expected, atol=limit * 1e-8, rtol=0)
                source_kind = "original_aggregated_metrics_csv"
                error_kind = "original_std_over_sqrt_total_realizations"
            for dv, y, lo, hi in zip(GRID, mean, lower, upper):
                rows.append({"panel": letter, "alpha": alpha, "metric": metric,
                             "label": label, "D_v": dv, "mean": y,
                             "lower": lo, "upper": hi, "display_divisor": scale,
                             "source_kind": source_kind, "error_kind": error_kind})
        specs.append({"panel": letter, "alpha": alpha, "metric": metric,
                      "y_max": limit, "display_divisor": scale, "optima": optima,
                      "source_form": form_name, "max_csv_graphic_abs_error": max_csv_error,
                      "source_form_sha256": hashlib.sha256(form.get_data()).hexdigest()})
    with (HERE / "analysis" / "plot_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "analysis" / "manifest.json").write_text(json.dumps(
        {"inputs": input_hashes(), "panels": specs, "n_means": len(rows),
         "tcor_csv_not_available": True, "no_simulations_run": True}, indent=2), encoding="utf-8")
    print("Prepared 240 means and 240 original error bars; CSV panels match source PDF.")


def verify():
    manifest = json.loads((HERE / "analysis" / "manifest.json").read_text())
    assert manifest["inputs"] == input_hashes()
    with (HERE / "analysis" / "plot_data.csv").open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    results = []
    for spec in manifest["panels"]:
        path = HERE / f"fig3{spec['panel']}.pdf"
        reader = PdfReader(path)
        assert len(reader.pages) == 1
        page = reader.pages[0]
        np.testing.assert_allclose([float(page.mediabox.width), float(page.mediabox.height)],
                                   np.array([5.6, 4.2]) * 72, atol=1e-5)
        limit = spec["y_max"] / spec["display_divisor"]
        graphic = extract(page.get_contents(), reader, limit)
        max_error = 0.
        for label in COLORS:
            group = [r for r in data if r["panel"] == spec["panel"] and r["label"] == label]
            for key in ("mean", "lower", "upper"):
                expected = np.array([float(r[key]) for r in group]) / spec["display_divisor"]
                max_error = max(max_error, float(np.max(np.abs(graphic[label][key] - expected))))
                np.testing.assert_allclose(graphic[label][key], expected, rtol=0, atol=limit * 2e-8)
            np.testing.assert_allclose(graphic[label]["optimal_dv"], spec["optima"][label]["dv"], rtol=2e-7)
        fonts = []
        for ref in page["/Resources"]["/Font"].values():
            font = ref.get_object()
            name = str(font["/BaseFont"])
            assert "TimesNewRoman" in name, name
            for descendant in font.get("/DescendantFonts", [font]):
                descriptor = descendant.get_object()["/FontDescriptor"].get_object()
                assert any(k in descriptor for k in ("/FontFile", "/FontFile2", "/FontFile3"))
            fonts.append(name)
        text = page.extract_text()
        assert not any(f"({letter})" in text for letter in "abcdef")
        results.append({"file": path.name, "sha256": sha(path), "fonts_embedded": fonts,
                        "max_graphic_abs_error_display_units": max_error,
                        "means_checked": 40, "error_bars_checked": 40})
    (HERE / "analysis" / "final_verification.json").write_text(json.dumps(
        {"inputs_unchanged": True, "panels": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else verify()
