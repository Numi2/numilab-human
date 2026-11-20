"""Backtrack retained side-specific proposals against the simultaneous exact geometry."""
from collections import Counter
import json

import numpy as np

from audit import closed_surface, digest
from audit_bed_layer_surfaces import owner
from check_candidate import targets
from extension import determinant, quality
from fit_muscle_interface import MUSCLES, exact_pairs, prepare_target


def candidates(asset_path, proposed_path, obstacles):
    asset = np.load(asset_path)
    with np.load(proposed_path) as proposed:
        for key in ("cells", "source_faces", "source_vertex_ids"):
            assert np.array_equal(asset[key], proposed[key])
        move = proposed["nodes"].astype(float) - asset["nodes"]
    base, cells = asset["nodes"], asset["cells"]
    lookup = {tuple(p): i for i, p in enumerate(base)}
    indices = np.array([lookup[tuple(p)] for p in asset["source_nodes"]])
    faces = indices[asset["source_faces"]]
    box = (np.minimum(base, base + move).min(axis=0) - 1e-7,
           np.maximum(base, base + move).max(axis=0) + 1e-7)
    prepared = {k: (ids, prepare_target(v, ids)) for k, v in obstacles.items()
                if (ids := targets(v, box)).size}
    baseline = exact_pairs(base, faces, prepared)
    frozen = np.unique(faces[sorted({i for group, i, _ in baseline if group == "bones"})])
    assert np.all(move[frozen] == 0.)
    d0 = determinant(base, cells)
    assert d0.min() > 0.
    original_records, _ = owner.float32_surface_records(base, faces)
    changed = set(np.flatnonzero((move[faces] != 0.).any(axis=(1, 2))).tolist())
    self_baseline = set(map(tuple, owner._audit_self_pairs_touching_faces(original_records, changed)["triangle_pairs"]))
    runs, passing = [], {}
    for n in range(17):
        p = (base + (n / 16.) * move).astype(np.float32)
        det = determinant(p.astype(float), cells)
        q = quality(p.astype(float), cells, det)
        v = float(abs(det.sum()/d0.sum() - 1.))
        pairs = exact_pairs(p, faces, prepared)
        outside = {p for p in pairs - baseline if p[0] != "layers"}
        surface, _ = owner.float32_surface_records(p, faces)
        self_crossings = set(map(tuple, owner._audit_self_pairs_touching_faces(surface, changed)["triangle_pairs"]))
        violations = self_crossings - self_baseline
        row = dict(numerator=n, layer_pairs=sum(p[0] == "layers" for p in pairs),
                   layer_pairs_added=sum(p[0] == "layers" for p in pairs-baseline),
                   new_outside_pairs=dict(Counter(p[0] for p in outside)), new_self_pairs=len(violations),
                   determinant_ratio=float(min(det / d0)), minimum_quality=float(q.min()),
                   absolute_relative_volume_change=v, moving_nodes=int(np.sum((p != base).any(axis=1))),
                   maximum_displacement_m=float(np.max(abs(p-base))))
        runs.append(row)
        if not outside and not violations and det.min() > 0. and q.min() >= 1e-5 and v <= 1e-6:
            assert closed_surface(p, faces)
            passing[n] = (p, surface, pairs)
    assert 0 in passing
    return asset, indices, faces, runs, passing


def check(prior, proposals, geometry, roles, out):
    built = {side: candidates(prior / f"gluteal-{side}.npz", proposals / f"gluteal-{side}.npz", geometry)
             for side in MUSCLES}
    right, left = (built[i] for i in MUSCLES)
    control = set(map(tuple, owner._audit_pair_prepared_first(
        owner._prepare_surface_aabb(right[-1][0][1]), left[-1][0][1])["triangle_pairs"]))
    combination_checks = []
    best = None
    for a, (_, r, _) in right[-1].items():
        prepared = owner._prepare_surface_aabb(r)
        for b, (_, l, _) in left[-1].items():
            mutual = set(map(tuple, owner._audit_pair_prepared_first(prepared, l)["triangle_pairs"]))
            added = mutual - control
            count = right[-2][a]["layer_pairs"] + left[-2][b]["layer_pairs"]
            combination_checks.append(dict(fractions_over_16=[a,b], layer_pairs=count,
                                           mutual_pairs=len(mutual), new_mutual_pairs=len(added)))
            if not added:
                key = (count, right[-2][a]["maximum_displacement_m"] + left[-2][b]["maximum_displacement_m"])
                if best is None or key < best[0]:
                    best = (key, (a,b))
    assert best is not None
    selected = {}
    for side, a in zip(MUSCLES, best[1], strict=True):
        asset, indices, faces, runs, passing = built[side]
        points, _, pairs = passing[a]
        output = out / f"gluteal-{side}.npz"
        np.savez_compressed(output, nodes=points, cells=asset["cells"], source_nodes=points[indices],
                            source_faces=asset["source_faces"], source_vertex_ids=asset["source_vertex_ids"])
        asset.close()
        selected[str(side)] = dict(metrics=runs[a], layer_counts=dict(Counter(roles[j] for k, _, j in pairs if k == "layers")),
                                   retained_layer_pairs=sorted((i,j) for k,i,j in pairs if k == "layers"), geometry_sha256=digest(output))
    return dict(proposal_sha256={str(i):digest(proposals / f"gluteal-{i}.npz") for i in MUSCLES},
                baseline_mutual_gluteal_pairs=len(control), fractions={str(i):built[i][-2] for i in MUSCLES},
                bilateral_checks=combination_checks, selected=selected)
