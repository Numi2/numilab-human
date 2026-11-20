"""Choose disjoint interior edge stars from exact limiting-cell witnesses."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skull-contact-036"))
from rebuild import HERE, determinant, digest, quality, table
from refine_clear_banks import confined_bisect


def reference_points(data, frame):
    rows = table(HERE / "compiled-reference/shell-mass.csv.nodes.csv")
    original = np.array([[float(row[f"{frame}_{axis}_m"]) for axis in "xyz"] for row in rows])
    midpoint = original[data["edge_parent"]].mean(axis=1)
    a, b = data["interior_edge_parent"]
    t = float(data["interior_edge_fraction"])
    interior = (1. - t) * original[a] + t * original[b]
    appended = np.vstack((midpoint, interior))
    if frame == "current":
        appended = appended.astype(np.float32).astype(np.float64)
    return np.vstack((original, appended))


def score_edge(edge, positions, old, orientation):
    a, b = edge
    added = len(positions[0])
    local, lineage = confined_bisect(old, np.arange(len(old)), a, b, added)
    assert len(local) == 2 * len(old)
    floors = [float(quality(p, old).min()) for p in positions]
    volumes = [determinant(p, old) for p in positions]
    whole_shape = 0.
    winner = None
    for fraction in np.linspace(.20, .80, 25):
        gains, ratios, fixed = [], [], None
        for k, points in enumerate(positions):
            if k in (0, 1):
                center = (1. - fraction) * points[a] + fraction * points[b]
                if k == 0:
                    center = center.astype(np.float32).astype(np.float64)
                    fixed = [center]
                else:
                    fixed.append(center)
            else:
                center = fixed[k - 2]
            trial = np.vstack((points, center))
            new_volume = determinant(trial, local)
            if new_volume.min() <= 1.001 * orientation[k % 2]:
                break
            if not np.isclose(new_volume.sum(), volumes[k].sum(), rtol=2e-5, atol=1e-19):
                break
            gains.append(float(quality(trial, local).min()) / floors[k])
            ratios.append(float(new_volume.min() / volumes[k].min()))
        else:
            score = min(gains)
            if score > whole_shape:
                whole_shape, winner = score, (float(fraction), gains, ratios)
    return winner


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.source / "receipt.json").read_text())
    archive = args.source / "rebuild.npz"
    assert digest(archive) == receipt["candidate_sha256"]
    data = dict(np.load(archive, allow_pickle=False))
    cells, material = data["elements"], data["material"]
    reference_world = reference_points(data, "current")
    reference_rest = reference_points(data, "rest")
    positions = (reference_world, reference_rest, data["world"].astype(np.float64), data["rest"])
    orientation = [determinant(p, cells).min() for p in positions[:2]]
    samples = Counter()
    for step in receipt["fit"]:
        for event in step["geometric_witnesses"]:
            for frame in event["frames"].values():
                for cell, parent, _ in frame["worst_children"]:
                    assert data["parent"][cell] == parent
                    samples[cell] += 1
    assert samples
    boundary_edges = {edge for face in data["surface"].tolist()
                      for edge in combinations(sorted(face), 2)}
    edge_witnesses = Counter()
    for row, visits in samples.items():
        for edge in combinations(sorted(cells[row].tolist()), 2):
            if edge not in boundary_edges and max(edge) < len(reference_world) - 1:
                edge_witnesses[edge] += visits
    incident = defaultdict(list)
    for row, tet in enumerate(cells.tolist()):
        for edge in combinations(sorted(tet), 2):
            if edge in edge_witnesses:
                incident[edge].append(row)
    trials = []
    for edge, visits in edge_witnesses.most_common(120):
        rows = incident[edge]
        if len(rows) < 3 or len(np.unique(material[rows])) != 1:
            continue
        result = score_edge(edge, positions, cells[rows], orientation)
        if result is None or min(result[1]) <= 1.02:
            continue
        fraction, gains, ratios = result
        trials.append({"edge": edge, "fraction": fraction, "rows": rows, "witnesses": visits,
                       "parent_rows": sorted(set(map(int, data["parent"][rows]))),
                       "shape_gains_base_world_base_rest_fit_world_fit_rest": gains,
                       "minimum_det_ratios": ratios})
    used = set()
    retained = []
    for cavity in sorted(trials, key=lambda x: (-x["witnesses"], -min(x["shape_gains_base_world_base_rest_fit_world_fit_rest"]),
                                                 x["edge"])):
        if used.isdisjoint(cavity["rows"]):
            used.update(cavity["rows"])
            retained.append(cavity)
    covered = set(samples).intersection(used)
    args.output.write_text(json.dumps({
        "checkpoint_sha256": receipt["candidate_sha256"],
        "examined_edges": min(120, len(edge_witnesses)), "quality_improving_edges": len(trials),
        "total_witnesses": sum(samples.values()), "covered_witnesses": sum(samples[i] for i in covered),
        "total_limiting_children": len(samples), "covered_limiting_children": len(covered),
        "retained_cavities": retained,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
