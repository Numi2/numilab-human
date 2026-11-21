"""Measure bed-normal load path engagement in the repaired posterior anatomy."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from audit import BONE, contacts, digest, rows
from audit_bed_layer_surfaces import role_surfaces
from screen_gluteal_columns import first_chord

MARGIN = 1.e-7
CLOSURE_M = (.0005, .001, .002, .005, .010, .020)


def material(path):
    source = path.read_text()
    assert "energy = neo_hookean(mu, lambda);" in source
    mu, lam = (1000. * float(re.search(
        rf"parameter {name} : kPa = ([\d.]+)", source)[1]) for name in ("mu", "lambda"))
    return {
        "shear_pa": mu,
        "lame_lambda_pa": lam,
        "unconfined_axial_initial_tangent_pa": mu * (3. * lam + 2. * mu) / (lam + mu),
        "confined_axial_initial_tangent_pa": lam + 2. * mu,
    }


def capacity(routes, parameters):
    totals = []
    for closure in CLOSURE_M:
        active = [r for r in routes if r["clearance_m"] < closure]
        delta = np.array([closure - r["clearance_m"] for r in active])
        length = np.array([r["muscle_path_length_m"] for r in active])
        area = np.array([r["projected_area_m2"] for r in active])
        ratio = delta / length
        # The instantaneous tangent provides sensitivity only. Extrapolating
        # it through large strains would confound the source inventory.
        keep = ratio <= .15
        gain = float(area[keep] @ ratio[keep])
        totals.append({
            "closure_m": closure, "engaged_routes": len(active),
            "projected_area_sum_m2": float(area.sum()),
            "routes_within_15_percent_strain": int(keep.sum()),
            "routes_excluded_above_15_percent_strain": int((~keep).sum()),
            "low_strain_unconfined_load_n": parameters["unconfined_axial_initial_tangent_pa"] * gain,
            "low_strain_confined_load_n": parameters["confined_axial_initial_tangent_pa"] * gain,
        })
    return totals


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("run", "bone", "partition", "crossings", "material", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--shell-fit", type=Path)
    a = parser.parse_args()
    shell_nodes = a.run / "shell-mass.csv.nodes.csv"
    shell_cells = a.run / "shell-mass.csv.tets.csv"
    mesh = np.array([[float(r[f"current_{axis}_m"]) for axis in "xyz"]
                     for r in rows(shell_nodes)], dtype=np.float32).astype(float)
    if a.shell_fit is not None:
        with np.load(a.shell_fit) as fitted:
            data = fitted["nodes_current"].astype(float)
        assert mesh.shape == data.shape and np.array_equal(mesh[:, :2], data[:, :2])
        assert np.array_equal(mesh[:2 * (len(mesh) // 3)], data[:2 * (len(mesh) // 3)])
        mesh = data
    n = len(mesh) // 3
    assert len(mesh) == 3 * n
    groups = role_surfaces(rows(shell_cells))
    faces = np.asarray(groups["deep"])
    offset = len(groups["outer"]) + len(groups["interface"])
    crossing_data = json.loads(a.crossings.read_text())
    crossed = {int(row[1]) for row in crossing_data["initial"]["witnesses"]}
    bones = np.memmap(a.bone, mode="r", dtype=BONE)
    deep = mesh[faces]
    outer = mesh[faces % n]
    area = .5 * np.abs(np.cross(deep[:, 1] - deep[:, 0], deep[:, 2] - deep[:, 0])[:, 2])
    close_to_bed = outer[:, :, 2].max(axis=1) < mesh[:n, 2].min() + .050
    eligible = np.flatnonzero(close_to_bed & (area > 0.))
    origins = deep[eligible].mean(axis=1)
    candidates, hashes = {}, {}
    for side in (29, 30):
        mesh_file = a.partition / f"gluteal-{side}.npz"
        with np.load(mesh_file) as shape:
            p, f = shape["source_nodes"], shape["source_faces"]
        candidates[side] = p[f].astype(float)
        hashes[side] = digest(mesh_file)
    hull_lo = np.minimum(*(p.min(axis=(0, 1)) for p in candidates.values()))
    hull_hi = np.maximum(*(p.max(axis=(0, 1)) for p in candidates.values()))
    indices = np.flatnonzero(np.all((origins[:, :2] >= hull_lo[:2]) &
                                   (origins[:, :2] <= hull_hi[:2]), axis=1))
    locations = origins[indices]
    bone_hits = contacts(locations[:, :2], bones["vertices"])
    hits = {side: contacts(locations[:, :2], p) for side, p in candidates.items()}
    normals = {side: np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
               for side, p in candidates.items()}
    rejections = Counter()
    routes = []
    for local, index in enumerate(indices):
        face = int(eligible[index])
        if face + offset in crossed:
            rejections["deep_face_crosses_bone"] += 1
            continue
        start = locations[local, 2]
        chord_bank = {}
        for side in (29, 30):
            intersections = [(i, z) for i, z in hits[side][local] if z > start + MARGIN]
            chord = first_chord(intersections, normals[side], start, MARGIN)
            if chord is not None:
                chord_bank[side] = chord
        if not chord_bank:
            rejections["no_valid_gluteal_chord"] += 1
            continue
        side, ((enter, entry_z), (exit, exit_z)) = min(chord_bank.items(),
              key=lambda item: item[1][0][1])
        donor = next(((i, z) for i, z in bone_hits[local] if z > start + MARGIN), None)
        if donor is None or donor[1] < exit_z + MARGIN:
            rejections["no_distinct_bone_beyond_chord"] += 1
            continue
        bank, donor_z = donor
        src = bones[bank]
        anatomical_donors = {1, 128, (131 if side == 29 else 145)}
        if int(src["body"]) not in anatomical_donors:
            rejections["other_bone_is_first"] += 1
            continue
        opposite = 30 if side == 29 else 29
        if any(entry_z - MARGIN < z < donor_z - MARGIN for _, z in hits[opposite][local]):
            rejections["opposing_muscle_blocks_column"] += 1
            continue
        face_id = tuple(map(int, faces[face]))
        distal_gap = donor_z - exit_z
        proximal_gap = entry_z - start
        routes.append({
            "side": side, "fat_face": face + offset, "fat_nodes": face_id,
            "entry_face": int(enter), "exit_face": int(exit),
            "bone_body": int(src["body"]), "bone_bank": int(src["bone"]),
            "bone_face": int(src["triangle"]),
            "point_m": locations[local].tolist(),
            "fat_muscle_gap_m": proximal_gap, "muscle_bone_gap_m": distal_gap,
            "clearance_m": proximal_gap + distal_gap,
            "muscle_path_length_m": exit_z - entry_z,
            "projected_area_m2": float(area[face]),
        })
    assert len(routes) + sum(rejections.values()) == len(indices)
    params = material(a.material)
    result = {
        "scope": ("Centroid rays at distinct deep-fat faces. Pure bed-normal relative motion "
                  "with fixed skeletal donors, before tissue or joints redistribute loads. "
                  "Areas can overlap in projection. Stiffness is a low-strain sensitivity "
                  "using the existing uncalibrated passive muscle material, ignoring other "
                  "compliances and adjacent muscle obstructions. No runtime rows are installed."),
        "input_sha256": {p.name: digest(p) for p in (shell_nodes, shell_cells, a.bone, a.crossings, a.material)},
        "gluteal_mesh_sha256": hashes,
        "shell_fit_sha256": digest(a.shell_fit) if a.shell_fit else None,
        "reference_material": params,
        "tested_faces": len(indices),
        "eligible_5_cm_outer_band_faces": len(eligible),
        "excluded": rejections,
        "sacral_routes": sum(r["bone_body"] == 1 for r in routes),
        "routes": routes,
        "survey": capacity(routes, params),
    }
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"{len(routes)} routes from {len(indices)} faces, {result['sacral_routes']} reach sacrum")
    for row in result["survey"]:
        print(row)


if __name__ == "__main__":
    main()
