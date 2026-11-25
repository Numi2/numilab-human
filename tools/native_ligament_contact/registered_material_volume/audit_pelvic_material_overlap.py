"""Measure registered dermal and adipose mass inside the two pelvic cortical solids."""

import argparse
import json
from pathlib import Path

import numpy as np
from igl.copyleft import cgal

from audit_registered_cortical_occupancy import banks, digest, read_material_capture
from measure_registered_overlap_mass import overlap_mass


def signed_volume(points, faces):
    if not len(faces):
        return 0.0
    t = points[faces] - points[0]
    return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6


def patch_boundary(cells):
    oriented = cells[:, [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]].reshape(-1, 3)
    _, first, count = np.unique(
        np.sort(oriented, axis=1), axis=0, return_index=True, return_counts=True
    )
    if np.max(count) > 2:
        raise ValueError("Selected cell face has more than two owners")
    return oriented[first[count == 1]]


def oriented_solid_faces(points, faces):
    edges = faces[:, [(0, 1), (1, 2), (2, 0)]].reshape(-1, 2)
    _, index, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(count != 2) or np.any(
        np.bincount(index, weights=np.sign(edges[:, 1] - edges[:, 0])) != 0
    ):
        raise ValueError("Pelvic bank is not a closed edge-oriented boundary")
    return faces if signed_volume(points, faces) > 0 else faces[:, ::-1].copy()


def audit(run, output):
    cfg = json.loads((run / "launch.json").read_bytes())
    env = cfg["environment"]
    pre = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    elements, reference, labels = (
        Path(env[pre + k]) for k in ("TETGEN_ELEMENTS", "TETGEN_NODE", "CHILD_LABELS")
    )
    raw = np.loadtxt(elements, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    lab = np.loadtxt(labels, delimiter=",", skiprows=1, usecols=(1,), dtype=np.int64)
    if set(np.unique(lab)) != {15, 16} or not np.array_equal(raw[:, 4], lab):
        raise ValueError("Unmatched source material labels")
    source = Path(cfg["argv"][3])
    x, poses, cells, tags, _, obs = read_material_capture(
        run, source, reference, elements, labels, raw[:, :4] - 1, lab
    )
    mass = np.empty(len(raw))
    sources = {}
    for material, name in ((15, "SKIN"), (16, "FAT")):
        path = Path(
            env[
                "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"
                + name
                + "_CHILD_MASS_CSV"
            ]
        )
        if cfg["pins"].get(str(path)) != digest(path):
            raise ValueError("Changed material mass allocation")
        mass[lab == material] = np.loadtxt(
            path, skiprows=1, delimiter=",", usecols=(2,)
        )
        sources[name] = digest(path)
    if "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ" in env:
        with np.load(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"]) as change:
            mass = np.r_[
                mass[change["retained_input_cell_rows"]], change["replacement_mass_kg"]
            ]
    if len(mass) != len(cells) or not np.isfinite(mass).all() or mass.min() <= 0:
        raise ValueError("Missing accepted cell mass")
    source_banks = banks(source)
    manifest = source.with_suffix(".manifest.json")
    pelvis_ids = {
        a["core_body_index"]
        for a in json.loads(manifest.read_bytes())["source"]["anchors"]
        if a["myosim_body"] == "pelvis"
    }
    if len(pelvis_ids) != 1:
        raise ValueError("Pelvic owner is ambiguous")
    body = pelvis_ids.pop()
    rot, translation = poses[body]
    solids = []
    for (donor, bank), (f, vertices, _) in source_banks.items():
        if donor != body:
            continue
        v, ix = np.unique(vertices, axis=0, return_inverse=True)
        v = rot.apply(v) + translation
        solids.append((bank, v, oriented_solid_faces(v, ix[f])))
    if len(solids) != 2:
        raise ValueError("Expected both registered pelvic banks")
    _, v0, f0 = solids[0]
    _, v1, f1 = solids[1]
    union, face, _ = cgal.mesh_boolean(v0, f0, v1, f1, "intersect")
    if len(face) and abs(signed_volume(union, face)) > 1e-18:
        raise ValueError("Pelvic volumes overlap")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    owned = {}
    for material in (15, 16):
        rows = np.flatnonzero(tags == material)
        tet = x[cells[rows]]
        low = tet.min(1)
        high = tet.max(1)
        for bank, v, f in solids:
            local = rows[((low <= v.max(0)) & (high >= v.min(0))).all(1)]
            if not len(local):
                reports.append(
                    {
                        "bank": bank,
                        "material": material,
                        "cells": 0,
                        "volume_ml": 0.0,
                        "mass_g": 0.0,
                    }
                )
                continue
            z = cells[local]
            surface = patch_boundary(z)
            nodes, idx = np.unique(surface, return_inverse=True)
            clipped, cf, _ = cgal.mesh_boolean(
                x[nodes], idx.reshape(-1, 3), v, f, "intersect"
            )
            volume = signed_volume(clipped, cf)
            ids, share, kg, bary = overlap_mass(x, z, mass[local], v, f, volume)
            ids = local[ids]
            key = f"{bank}_{material}"
            if len(ids):
                owned.update(
                    {
                        key + "_cell_rows": ids,
                        key + "_node_rows": cells[ids],
                        key + "_volume_fraction": share,
                        key + "_mass_kg": kg,
                        key + "_parent_barycentric": bary,
                    }
                )
            reports.append(
                {
                    "bank": bank,
                    "material": material,
                    "cells": len(ids),
                    "volume_ml": volume * 1e6,
                    "mass_g": kg.sum() * 1000,
                }
            )
    dest = output / "pelvic-cell-mass.npz"
    np.savez_compressed(dest, **owned)
    result = {
        "native_launch_sha256": digest(run / "launch.json"),
        "accepted_receipt_sha256": obs["receipt_sha256"],
        "physical_nodes_sha256": obs["accepted_node_sha256"],
        "anatomy_sha256": digest(source),
        "anatomical_identity_sha256": digest(manifest),
        "mass_allocation_sha256": sources,
        "cell_ledger_sha256": digest(dest),
        "regions": reports,
    }
    (output / "pelvic-material.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accepted_run", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = audit(args.accepted_run.resolve(), args.output_directory.resolve())
    print(
        "Pelvic overlap",
        sum(r["volume_ml"] for r in result["regions"]),
        "mL,",
        sum(r["mass_g"] for r in result["regions"]),
        "g",
    )
