"""Transport each native arm as one rigid chain to the thoracic head reference at bed entrance."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from bone_overlap import intersections
from compare_registration import prepare
from fast_skin_intersection_audit import install_for_float32
from numilab_human.capsular_material_registration import (
    emitted_body_vertices,
    material_bank,
    read_material_banks,
)
from scipy.spatial.transform import Rotation
from screen_material_transport import inputs as layer_inputs
from shoulder_inputs import digest

from shared_thoracic_reference import fit


def descendants(path, start):
    raw = path.read_bytes()
    if raw[:8] != b"NHRIGID2":
        raise ValueError("Unsupported rigid owner")
    nb, nj = struct.unpack_from("<2I", raw, 20)
    edges = [
        struct.unpack_from("<2I", raw, 224 + 160 * nb + 144 * i) for i in range(nj)
    ]
    if (
        nj != nb - 1
        or len({child for _, child in edges}) != nj
        or any(max(a, b) >= nb for a, b in edges)
    ):
        raise ValueError("Rigid body tree is inconsistent")
    result = {start}
    while True:
        updated = result | {child for parent, child in edges if parent in result}
        if updated == result:
            return result
        result = updated


def run(
    launch,
    registration,
    motion,
    archive,
    scene,
    loader,
    receipt,
    bones_path,
    torso,
    head,
    chain,
    spacing,
    out,
):
    _, _, _, _, _, _, atlas = prepare(launch, registration, motion, archive)
    _, layers, _, _, pins = layer_inputs(scene, loader, receipt, bones_path)
    cfg = json.loads((scene / "launch.json").read_text())
    rigid = Path(cfg["argv"][1])
    poses = {
        e["body_index"]: np.array(e["position_m"] + e["quaternion_xyzw"])
        for e in json.loads(receipt.read_text())["accepted_body_poses"]
    }
    data = read_material_banks(bones_path)
    local = {}
    tris = {}
    old = {}
    for sid, r in data[2].items():
        _, r, attrs, f = material_bank(data, sid, r[0])
        local[sid] = emitted_body_vertices(attrs, r)
        pose = poses[r[0]]
        old[sid] = pose[:3] + Rotation.from_quat(pose[3:]).apply(local[sid])
        tris[sid] = f
    chest = json.loads(torso.read_text())
    t = poses[data[2][27][0]]
    chest_rotation = Rotation.from_quat(t[3:]) * Rotation.from_quat(chest["rot_xyzw"])
    normal = np.array(chest["bed_normal"])
    targets = {
        r["side"]: r["variants"][0] for r in json.loads(head.read_text())["rows"]
    }
    wrists = {
        r["side"]: next(s["wrist_target_m"] for s in r["samples"] if s["step"] == -4)
        for r in json.loads(chain.read_text())["rows"]
    }
    shifts = {
        r["side"]: np.array(r["translation_atlas_axes_m"])
        for r in json.loads(spacing.read_text())["rows"]
    }
    group = {}
    new = dict(old)
    moved = []
    for side, scapula, humerus in [("l", 11, 13), ("r", 10, 12)]:
        center = np.array(targets[side]["atlas_center_m"])
        dest = np.array(targets[side]["native_entrance_target_m"])
        arm = descendants(rigid, data[2][humerus][0])
        for sid, ids, name in [
            (scapula, [scapula], side + "_scapula"),
            (
                humerus,
                [k for k, row in data[2].items() if row[0] in arm],
                side + "_arm",
            ),
        ]:
            raw = atlas[sid][2]
            prior, scale, offset = fit(raw, old[sid], np.ones(len(raw)))
            error = np.max(
                np.linalg.norm(scale * prior.apply(raw) + offset - old[sid], axis=1)
            )
            if error > 2e-6:
                raise ValueError(f"Rigid material map differs on bank {sid}")
            previous_center = prior.apply(center) * scale + offset
            rot = chest_rotation * prior.inv()

            dest = np.asarray(targets[side]["native_entrance_target_m"]) + (
                chest_rotation.apply(shifts[side]) if sid == humerus else np.zeros(3)
            )
            for member in ids:
                if member in group:
                    raise ValueError("Limb ownership overlaps")
                group[member] = name
                new[member] = dest + rot.apply(old[member] - previous_center)
            report = {
                "group": name,
                "banks": sorted(ids),
                "rotation_xyzw": rot.as_quat().tolist(),
                "pivot_before_m": previous_center.tolist(),
                "pivot_after_m": dest.tolist(),
                "native_material_map_error_m": error,
                "bank_scale": scale,
            }
            if sid == humerus:
                wrist = np.array(wrists[side])
                report.update(
                    wrist_displacement_m=(
                        (dest + rot.apply(wrist - previous_center)) - wrist
                    ).tolist(),
                    wrist_bed_gap_m=float(
                        (dest + rot.apply(wrist - previous_center)) @ normal
                        - chest["bed_offset_m"]
                    ),
                )
            moved.append(report)
    install_for_float32()
    records = []
    for label, positions in (("entrance", old), ("free_arm", new)):
        bounds = {s: (p.min(0), p.max(0)) for s, p in positions.items()}
        pairs = []
        layer_pairs = []
        minima = {}
        for s in sorted(group):
            p, f = positions[s], tris[s]
            g = group[s]
            z = float((p @ normal - chest["bed_offset_m"]).min())
            if g not in minima or z < minima[g][0]:
                minima[g] = [z, s]
            for k, ref in positions.items():
                if k == s or (k in group and (k < s or group[k] == g)):
                    continue
                if np.any(bounds[k][0] > bounds[s][1]) or np.any(
                    bounds[k][1] < bounds[s][0]
                ):
                    continue
                hits = intersections(p, f, ref, tris[k])
                if hits:
                    pairs.append([s, k, len(hits), hits[0]])
            for k in (1, 0xFFFFFD11, 0xFFFFFD12):
                y, faces = layers[51007, k]
                hits = intersections(p, f, y, faces)
                if hits:
                    layer_pairs.append([s, k, len(hits), hits[0]])
        records.append(
            {
                "state": label,
                "bed_minimum_m_and_bank": minima,
                "checked_moved_banks": len(group),
                "between_groups_triangle_pairs": pairs,
                "skin_layer_triangle_pairs": layer_pairs,
            }
        )
    out.write_text(
        json.dumps(
            {
                "rigid_sha256": digest(rigid),
                "bones_sha256": data[1],
                "captured": pins,
                "motion": "single captured entrance, all native descendants of each humerus follow its rigid transform",
                "moved": moved,
                "checks": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
