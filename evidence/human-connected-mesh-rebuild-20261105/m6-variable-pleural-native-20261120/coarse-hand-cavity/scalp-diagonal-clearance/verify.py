"""Independently certify the edited shell at two pinned native poses."""
from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from audit import digest, rows
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from audit_bed_motion_shell import terminal_positions
from audit_boundary_bones import audit as audit_bones
from cavity import LABEL, cell_shape, determinant, oriented_boundary
from map_mass import regional_context


def connectivity(path):
    return np.array([[int(r[f"parent_node{i}"]) for i in range(4)] for r in rows(path)])


def canonical_witnesses(report):
    return {(role, tuple(face), body, bone, tri)
            for role, _, face, body, bone, tri in report["witnesses"]}


def endpoint_difference(before, after, edited_before, edited_after):
    old, new = map(canonical_witnesses, (before, after))
    retained = {w for w in old if (w[0], w[1]) not in edited_before}
    assert retained == {w for w in new if (w[0], w[1]) not in edited_after}
    removed, added = old - new, new - old
    bank = lambda keys: Counter(f"{role}/{body}/{bone}" for role, _, body, bone, _ in keys)
    assert not (bank(added) - bank(removed))
    return {"baseline_pairs": len(old), "edited_pairs": len(new),
            "net_pairs_removed": len(old) - len(new),
            "removed_by_role_and_bank": dict(bank(removed)),
            "added_by_role_and_bank": dict(bank(added))}


def verify(args):
    install_for_float32()
    tets = args.run / "shell-mass.csv.tets.csv"
    nodes = args.run / "shell-mass.csv.nodes.csv"
    labels = args.run / "shell-mass.csv.labels.csv"
    log = args.run / "stdout.log"
    motion = args.run / "native/resting-body-motion-partition.csv"
    witness = args.run / "bone-endpoints.json"
    baseline = json.loads(witness.read_text())
    for path in (tets, nodes, log, motion, args.bones):
        assert baseline["input_hashes"][path.name] == digest(path)
    report_path = args.candidate / "geometry.json"
    proposal = json.loads(report_path.read_text())
    for path in (nodes, tets, labels):
        assert proposal["source_sha256"][path.name] == digest(path)

    compiler, pools, _, original_source = regional_context(args.original, args.shared)
    assert len(pools) == 628
    reference_labels = rows(args.original / labels.name)
    material_labels = rows(labels)
    assert [tuple(r[k] for k in LABEL) for r in reference_labels] == [
            tuple(r[k] for k in LABEL) for r in material_labels]
    dataset = args.candidate / "candidate.npz"
    with np.load(dataset) as data:
        source, target, weight, stored_mass = (data[k] for k in
                     ("parent_cell", "target_cell", "mass_kg", "cell_mass_kg"))
        old, cells, changed = (data[k] for k in ("original_cells", "cells", "changed_rows"))
    assert len(original_source) == len(old) == len(cells)
    np.testing.assert_array_equal(old, connectivity(tets))
    np.testing.assert_array_equal(changed, np.flatnonzero(np.any(old != cells, axis=1)))
    assert len({tuple(sorted(row)) for row in cells}) == len(cells)
    oriented_boundary(cells)
    assert np.all(weight > 0.)
    np.testing.assert_array_equal(compiler.group_for_parent[source], compiler.group_for_parent[target])
    source_balance = np.bincount(source, weights=weight, minlength=len(cells)) - compiler.source_kg
    assert np.max(np.abs(source_balance)) < 1e-12
    mapped = np.bincount(target, weights=weight, minlength=len(cells))
    np.testing.assert_array_equal(mapped, stored_mass)

    node_rows = rows(nodes)
    rest = np.array([[r[f"rest_{a}_m"] for a in "xyz"] for r in node_rows], float)
    start = np.array([[r[f"current_{a}_m"] for a in "xyz"] for r in node_rows], np.float32)
    end = terminal_positions(log, start)
    route = np.linalg.norm(compiler.source_centroid[source] - rest[cells].mean(axis=1)[target], axis=1)
    max_route = float(route.max())
    changed_route = float(route[np.isin(target, changed)].max())
    assert max_route <= .075
    density = mapped * 6. / determinant(rest, cells)
    lo, hi = compiler.density_limits.T
    assert np.all(density >= lo-1.e-5) and np.all(density <= hi+1.e-5)
    nodal = np.bincount(cells.ravel(), weights=np.repeat(mapped / 4., 4), minlength=len(rest))
    assert np.all(nodal > 0.) and abs(nodal.sum()-mapped.sum()) < 1.e-10

    new_rows = rows(tets)
    before = role_surfaces(new_rows)
    for row, cell in zip(new_rows, cells, strict=True):
        for i, node in enumerate(cell):
            row[f"parent_node{i}"] = int(node)
    after = role_surfaces(new_rows)
    exact_faces = lambda table: {(role, tuple(face)) for role, group in table.items() for face in group}
    before_keys, after_keys = map(exact_faces, (before, after))
    expected = lambda key: {(role, tuple(sorted(face))) for option in proposal["selected"]
                         for i, face in enumerate(option[key]) for role in [("outer", "interface", "deep")[i//2]]}
    assert before_keys - after_keys == expected("before")
    assert after_keys - before_keys == expected("after")
    assert len(before_keys) == len(after_keys)

    certificate = {}
    ordered = sorted(after_keys)
    edited_faces = {i for i, face in enumerate(ordered) if face not in before_keys}
    for name, xyz in (("reference", rest), ("initial", start), ("terminal", end)):
        x = xyz.astype(float)
        det = determinant(x, cells)
        q = cell_shape(x, cells, det)
        assert (det > 0.).all() and (q >= 1.e-5).all()
        # The retained native poses already passed an all-shell audit. Test
        # every new face against the complete shell, including the other layers.
        entry = {"min_signed_determinant_m3": float(det.min()), "min_shape": float(q.min())}
        if name != "reference":
            shell = owner.float32_surface_records(xyz, [f for _, f in ordered])[0]
            crossing = owner._audit_self_pairs_touching_faces(shell, edited_faces)["triangle_pairs"]
            entry["new_shell_crossing_pairs"] = len(crossing)
            if crossing:
                entry["crossing_witnesses"] = [[ordered[a], ordered[b]] for a, b in sorted(crossing)]
        certificate[name] = entry

    staged = args.output / "pose-input"
    staged.mkdir(parents=True)
    (staged / "native").mkdir()
    for path in (nodes, log, motion):
        copy = staged / path.relative_to(args.run)
        copy.symlink_to(path.resolve())
    with (staged / tets.name).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0]))
        writer.writeheader()
        writer.writerows(new_rows)
    raw = audit_bones(staged, args.bones, baseline["accepted_endpoint_step"])
    bone_results = {}
    for phase in ("initial", "terminal"):
        result = endpoint_difference(baseline[phase], raw[phase], before_keys-after_keys, after_keys-before_keys)
        assert result["net_pairs_removed"] == proposal["selected_pair_gain"][len(bone_results)]
        bone_results[phase] = result
    payload = {"scope": "Authoring candidate at fixed accepted native poses. No integration or motion claim.",
               "changed_tetrahedra": len(changed), "changed_faces": len(edited_faces),
               "bone_triangles": raw["bone_faces"], "bone": bone_results, "quality": certificate,
               "mapped_mass_kg": float(mapped.sum()), "source_mass_error_kg": float(np.abs(source_balance).max()),
               "maximum_changed_route_m": changed_route, "maximum_route_m": max_route,
               "source_pools": len(pools), "volume_cells": len(cells),
               "material_density_kg_m3": {str(m): [float(density[compiler.material == m].min()),
                                                   float(density[compiler.material == m].max())] for m in (15, 16)},
               "input_sha256": {p.name: digest(p) for p in (nodes, tets, labels, witness, dataset, report_path)}}
    for name, content in (("verification.json", payload), ("bone-witnesses.json", raw)):
        with (args.output / name).open("x") as f:
            json.dump(content, f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps(payload, indent=2))
    return all(not record.get("new_shell_crossing_pairs") for record in certificate.values())


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for key in ("run", "bones", "original", "shared", "candidate", "output"):
        p.add_argument(key, type=Path)
    raise SystemExit(0 if verify(p.parse_args()) else 1)
