"""Draw the construction using the manuscript's actual network generator."""

import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import morris_lecar_paired_alpha_compare_simplicial as model

FIG = HERE / "figures"
OUT = HERE / "analysis"
PARAMS = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined/candidate39_main/chunks/network_000/parameters.json"
SEED = 36216
POSITIONS = np.array([
    [0.20, 2.80], [2.10, 2.80], [3.50, 0.40],
    [1.30, 0.90], [1.40, 2.80], [0.80, 1.65],
    [2.10, 0.40], [3.50, 2.80], [2.80, 1.65],
])
EDGE = "#454b53"
CLIQUE = "#2166ac"
FACE_EDGE = "#c62828"
FACE_FILL = "#f4c9cb"
NODE_RADIUS = .135


def orient(a, b, c):
    ab, ac = b - a, c - a
    return ab[0] * ac[1] - ab[1] * ac[0]


def validate_geometry(edges, cliques):
    for e, f in itertools.combinations(edges, 2):
        if set(e) & set(f):
            continue
        a, b = POSITIONS[list(e)]
        c, d = POSITIONS[list(f)]
        crossing = orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0
        assert not crossing, (e, f)
    for clique in cliques:
        a, b, c = POSITIONS[list(clique)]
        assert abs(orient(a, b, c)) > .1
        for node, point in enumerate(POSITIONS):
            if node in clique:
                continue
            signs = [orient(a, b, point), orient(b, c, point), orient(c, a, point)]
            assert not (all(s > 0 for s in signs) or all(s < 0 for s in signs))
    for a, b in itertools.combinations(POSITIONS, 2):
        assert np.linalg.norm(a - b) > 2 * NODE_RADIUS


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert original["n_nodes"] == 100
    assert original["p_edge"] == .15 and original["p_fill_simplex"] == .5
    illustrative = dict(original, n_nodes=9)
    p = model.PairedSimplicialParams(**illustrative)
    net = model.build_simplicial_network(p, np.random.default_rng(SEED))
    graph = nx.from_numpy_array(net.a1)
    edges = sorted(tuple(map(int, e)) for e in graph.edges())
    cliques = sorted(tuple(map(int, c)) for c in net.cliques3)
    filled = sorted(tuple(map(int, c)) for c in net.simplices2)
    nx_cliques = sorted(tuple(sorted(c)) for c in nx.enumerate_all_cliques(graph) if len(c) == 3)
    assert cliques == nx_cliques == [(0, 4, 5), (1, 7, 8), (2, 6, 8)]
    assert filled == [(1, 7, 8), (2, 6, 8)]
    assert len(edges) == 11 and nx.is_connected(graph)
    assert all(tuple(sorted(e)) in edges for c in filled for e in itertools.combinations(c, 2))
    validate_geometry(edges, cliques)
    font = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    stems = ["construction_01_er", "construction_02_cliques", "construction_03_simplices"]
    panels = []
    for stage, stem in enumerate(stems):
        fig, ax = plt.subplots(figsize=(3.6, 3.3))
        fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
        ax.set_aspect("equal")
        ax.set_xlim(-.08, 3.78)
        ax.set_ylim(.10, 3.10)
        ax.axis("off")
        visible_faces = filled if stage == 2 else []
        for clique in visible_faces:
            ax.add_patch(Polygon(POSITIONS[list(clique)], closed=True, facecolor=FACE_FILL,
                                 edgecolor="none", zorder=0))
        for i, j in edges:
            ax.plot(POSITIONS[[i, j], 0], POSITIONS[[i, j], 1], color=EDGE, lw=2.0, zorder=1)
        if stage >= 1:
            for clique in cliques:
                color = FACE_EDGE if stage == 2 and clique in filled else CLIQUE
                ax.add_patch(Polygon(POSITIONS[list(clique)], closed=True, fill=False,
                                     edgecolor=color, linewidth=2.4, zorder=2))
        for node, (x, y) in enumerate(POSITIONS):
            ax.add_patch(Circle((x, y), NODE_RADIUS, facecolor="white", edgecolor="#222222",
                                linewidth=1.4, zorder=3))
            ax.text(x, y, str(node + 1), ha="center", va="center", fontsize=16, zorder=4)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert 0 < box.x0 < box.x1 < fig.bbox.x1 and 0 < box.y0 < box.y1 < fig.bbox.y1
        for suffix in ("pdf", "eps", "png"):
            fig.savefig(FIG / f"{stem}.{suffix}", dpi=600, facecolor="white")
        plt.close(fig)
        panels.append({"stem": stem, "edges": edges, "filled_faces": visible_faces})
    manifest = {
        "scope": "Reviewer construction schematic only; no dynamical simulation or irreducibility test",
        "generator": "morris_lecar_paired_alpha_compare_simplicial.build_simplicial_network",
        "generator_sha256": hashlib.sha256((ROOT / "morris_lecar_paired_alpha_compare_simplicial.py").read_bytes()).hexdigest(),
        "source_parameters": str(PARAMS.relative_to(ROOT)),
        "source_parameters_sha256": hashlib.sha256(PARAMS.read_bytes()).hexdigest(),
        "simulation_n_nodes": 100, "illustrative_n_nodes": 9, "p_edge": .15, "p_fill": .5,
        "rng": "numpy.default_rng", "seed": SEED,
        "selection_note": "Illustrative realization selected for clarity, not a representative density or filling fraction",
        "display_node_labels": "source zero-based index plus one",
        "edges": edges, "cliques": cliques, "filled": filled,
        "unfilled": [c for c in cliques if c not in filled],
        "positions": POSITIONS.tolist(), "layout_crossings": 0, "panel_letters": False,
        "same_skeleton_in_all_panels": True, "font": font, "panels": panels,
    }
    (OUT / "construction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("illustrative_n_nodes", "p_edge", "p_fill", "seed", "edges", "cliques", "filled", "unfilled", "same_skeleton_in_all_panels")}, indent=2))


if __name__ == "__main__":
    main()
