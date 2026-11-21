"""Fit one rest frame from the registered rib cage and test the adjoining atlas shoulder assemblies."""

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32
from numilab_human.capsular_material_registration import (
    emitted_body_vertices,
    material_bank,
    read_material_banks,
)
from numilab_human.model import _bodyparts_obj_triangles
from scipy.spatial.transform import Rotation as R
from screen_material_transport import inputs as layer_inputs
from shoulder_inputs import digest


def fit(x, y, w):
    ax = np.average(x, axis=0, weights=w)
    ay = np.average(y, axis=0, weights=w)
    X = x - ax
    Y = y - ay
    u, s, vt = np.linalg.svd((X * w[:, None]).T @ Y)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    rot = R.from_matrix(((u * sign) @ vt).T)
    scale = (s @ sign) / np.dot(w, np.sum(X * X, axis=1))
    return rot, scale, ay - scale * rot.apply(ax)


def run(atlas, scene, loader, receipt, bone, centers, sites, out):
    _state, banks, _clavs, _, pins = layer_inputs(scene, loader, receipt, bone)
    init = {
        r["body_index"]: np.asarray(r["position_m"] + r["quaternion_xyzw"])
        for r in json.loads(receipt.read_text())["accepted_body_poses"]
    }
    data = read_material_banks(bone)
    manifest, pin, records, _verts, _idx = data
    raw = {}
    existing = {}
    equal = {}
    torso = records[27][0]
    samples = []
    ids = [
        s
        for s in records
        if s in (8, 9, 10, 11, 12, 13, 27) or (36 <= s <= 59 and records[s][0] == torso)
    ]
    archive = manifest["source"]["bodyparts"]["archives"][0]
    if digest(atlas) != archive["sha256"]:
        raise ValueError("Atlas source changed")
    with zipfile.ZipFile(atlas) as z:
        members = {Path(n).name: n for n in z.namelist() if n.endswith(".obj")}
        for s in ids:
            b = records[s][0]
            anchor, r, attrs, faces = material_bank(data, s, b)
            n = members[anchor["member_id"] + ".obj"]
            content = z.read(n)
            if hashlib.sha256(content).hexdigest() != anchor["member_sha256"]:
                raise ValueError(n)
            v, f = _bodyparts_obj_triangles(content, n)
            raw[s] = np.asarray(v) * 0.001, np.asarray(f)
            existing[s] = emitted_body_vertices(attrs, r), faces
            equal[s] = bool(np.array_equal(f, faces) and len(v) == len(attrs))
            if b != torso or not equal[s]:
                continue
            # The individual bank fits have the same aggregate weight.
            ids1 = np.unique(np.linspace(0, len(v) - 1, 600).astype(int))
            samples.append((s, raw[s][0][ids1], existing[s][0][ids1]))
    x = np.vstack([r[1] for r in samples])
    y = np.vstack([r[2] for r in samples])
    w = np.repeat(1.0, len(x))
    scatter = []
    for _ in range(4):
        rot, scale, t = fit(x, y, w)
        err = np.linalg.norm(rot.apply(x) * scale + t - y, axis=1)
        median = np.median(err)
        w = np.minimum(1.0, max(0.001, 1.5 * median) / np.maximum(err, 1e-12))
    for s, rx, ey in samples:
        mismatch = np.linalg.norm(rot.apply(rx) * scale + t - ey, axis=1)
        scatter.append([s, float(np.sqrt(np.mean(mismatch**2))), float(mismatch.max())])

    def atlas_to_entrance(x):
        p = init[torso]
        return p[:3] + R.from_quat(p[3:]).apply(rot.apply(x) * scale + t)

    target = {}
    for s in ids:
        v, f = raw[s]
        target[s] = (atlas_to_entrance(v).astype("f4"), f)
    plane = re.findall(
        r"^resting_fem_skin_bed_source=enabled .*?normal=([^ ]+) plane_offset_m=([^ ]+)",
        (scene / "stdout.log").read_text(),
        flags=re.MULTILINE,
    )
    if len(set(plane)) != 1:
        raise ValueError("Native bed plane is missing or ambiguous")
    normal = np.asarray([float(x) for x in plane[0][0].split(",")])
    offset = float(plane[0][1])
    if abs(np.linalg.norm(normal) - 1) > 1e-5:
        raise ValueError("Native bed normal differs from a unit vector")
    install_for_float32()
    audits = []
    for s in (8, 9, 10, 11, 12, 13):
        v, f = target[s]
        local = []
        for k, (y, g) in target.items():
            if s == k or (k in (8, 9, 10, 11, 12, 13) and k < s):
                continue
            if np.any(y.min(0) > v.max(0)) or np.any(v.min(0) > y.max(0)):
                continue
            p = intersections(v, f, y, g)
            if len(p):
                local.append([k, len(p)])
        layers = []
        for k in (1, 0xFFFFFD11, 0xFFFFFD12):
            y, g = banks[51007, k]
            pairs = intersections(v, f, y, g)
            layers.append([k, len(pairs)])
        audits.append(
            {
                "bank": s,
                "bed_gap_m": float((v.astype(float) @ normal - offset).min()),
                "shared_bony_neighbors": local,
                "native_frozen_layers": layers,
            }
        )
    # Frame mismatch for two independent estimates of each native shared pivot
    # in raw-atlas coordinates. Fit these points using donor/atlas material correspondence.
    named = {r["side"]: r for r in json.loads(sites.read_text())["rows"]}
    joint = []
    for rr in json.loads(centers.read_text())["rows"]:
        side = rr["side"]
        mapped = []
        ends = []
        for e, a in zip(rr["centers"], named[int(side == "r")]["attachments"]):
            s = e["bank"]
            vr = raw[s][0]
            vl = existing[s][0]
            if not equal[s]:
                raise ValueError("Raw shoulder topology changed")
            rr0, ss, tt = fit(vr, vl, np.ones(len(vr)))
            maxerr = np.max(np.linalg.norm(rr0.apply(vr) * ss + tt - vl, axis=1))
            if maxerr > 2e-6:
                raise ValueError("A material bank has a nonrigid prior map")
            atlas_p = rr0.inv().apply(np.asarray(e["atlas_center_body_m"]) - tt) / ss
            mapped.append(atlas_to_entrance(atlas_p))
            ends.append(
                atlas_to_entrance(a["barycentric"] @ vr[raw[s][1][a["triangle"]]])
            )
        p = init[rr["centers"][1]["owner"]]
        native = p[:3] + R.from_quat(p[3:]).apply(
            rr["centers"][1]["donor_center_body_m"]
        )
        joint.append(
            {
                "side": side,
                "independent_atlas_center_difference_m": np.linalg.norm(
                    mapped[1] - mapped[0]
                ),
                "atlas_center_offsets_from_native_m": [
                    np.linalg.norm(v - native) for v in mapped
                ],
                "span_m": np.linalg.norm(ends[1] - ends[0]),
            }
        )
    out.write_text(
        json.dumps(
            {
                "bone_sha256": pin,
                "skin_pose_sha256": pins["pose"],
                "fitted_banks": len(scatter),
                "excluded_unmatched_torso_ids": [
                    s for s in ids if records[s][0] == torso and not equal[s]
                ],
                "bed_normal": normal.tolist(),
                "bed_offset_m": offset,
                "rot_xyzw": rot.as_quat().tolist(),
                "scale": scale,
                "translation_m": t.tolist(),
                "per_bank_rms_max_m": scatter,
                "entrance": audits,
                "joints": joint,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
