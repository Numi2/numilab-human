"""Install the all-neighbor fit through the directly owned NHTISS bank with its native receipt."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from assemble_material import frames
from compile_connected_joint import digest
from prepare_shoulder import read
from registration_stages import parts
from repair_registered_surface import per_pose, state, volume
from scipy.spatial.transform import Rotation


def emit(w, case, fit, out):
    args = read(case / "launch.json")["argv"]
    src = Path(args[args.index("--soft-tissue-payload") + 1])
    receipt = Path(args[args.index("--resting-anatomy-receipt") + 1])
    check = read(fit / "fit.json")
    doc = read(receipt)
    owner = doc["passive_tissue_parent_bindings"]
    if (
        digest(src) != check["source_sha256"]
        or owner["child_payload_sha256"] != check["source_sha256"]
    ):
        raise ValueError("Stale native source")
    if not check["history"] or any(
        v > check["original_mm3"][k] + 1e-5 for k, v in check["candidate_mm3"].items()
    ):
        raise ValueError("Failed volumetric comparison")
    if any(s["stable_id"] == 83 for s in owner["surfaces"]) or any(
        t[1] == 83
        for key in ("surfaces", "anatomy_surfaces")
        for b in owner[key]
        for row in b["vertices"]
        for t in row
    ):
        raise ValueError("Inherited geometry requires repair of its original source")
    if any(digest(Path(p["path"])) != p["sha256"] for p in owner["source_payloads"]):
        raise ValueError("Changed shared source")
    poses = frames(case)
    raw = src.read_bytes()
    dst = bytearray(raw)
    vertices, wt, b, faces, aff = parts(src, 83)
    x, idx, inv = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    candidate = np.load(fit / "candidate.npz")
    p = candidate["source"]
    F = np.ascontiguousarray(inv[faces])
    pins = np.load(w / "registered/bank-83.npz")["pins"]
    if not np.array_equal(candidate["source_index"], inv) or not np.array_equal(
        x[pins], p[pins]
    ):
        raise ValueError("Invalid source or attachment")
    magic, abi, nr, nb, nv, ni, _ = struct.unpack_from("<8s6I", raw)
    records = np.frombuffer(raw, "<u4", nr * 8, 64).reshape(-1, 8)
    rr = records[records[:, 6] == 83]
    if len(rr) != 1 or int(rr[0, 3]) != len(vertices):
        raise ValueError("Ambiguous bank")
    start = 64 + 32 * nr + 36 * nb + 56 * int(rr[0, 2])
    live = np.frombuffer(dst, "<f4", 14 * len(vertices), start).reshape(-1, 14)
    if not np.array_equal(live[:, :3], x[inv]):
        raise ValueError("Coordinate mismatch")

    def normals(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        v0 = np.zeros_like(v)
        for row in F.T:
            np.add.at(v0, row, n)
        q = v0[inv]
        return q / np.linalg.norm(q, axis=1)[:, None]

    u, v = normals(x), normals(p)
    h = np.column_stack((np.cross(u, v), 1 + (u * v).sum(1)))
    if not np.all(np.isfinite(h)) or np.linalg.norm(h, axis=1).min() < 0.01:
        raise ValueError("Reversed surface normal")
    live[:, :3] = p[inv]
    live[:, 3:6] = Rotation.from_quat(h).apply(live[:, 3:6])
    old = np.frombuffer(raw, "u1")
    new = np.frombuffer(dst, "u1")
    different = np.flatnonzero(old != new)
    if (
        different.min() < start
        or different.max() >= start + 56 * len(vertices)
        or ((different - start) % 56 >= 24).any()
    ):
        raise ValueError("Changed tissue ownership")
    counts = [len(t) for t in per_pose(p, wt[idx], F, aff, b, poses)]
    oldframes = [x] + [
        state(x, wt[idx], aff, b, q).astype("f4").astype(float) for q in poses
    ]
    newframes = [p] + [
        state(p, wt[idx], aff, b, q).astype("f4").astype(float) for q in poses
    ]
    ratio = [
        abs(volume(a, F) / volume(b0, F) - 1) for a, b0 in zip(newframes, oldframes)
    ]
    bed = read(case / "support-check.json")
    gap = (newframes[-1] @ bed["bed_normal"] - bed["source_plane_offset_m"]).min()
    if any(counts) or max(ratio) > 0.01 or gap < 0:
        raise ValueError(
            ("Failed native pose, volume, or bed check", counts, ratio, float(gap))
        )
    out.mkdir(exist_ok=True)
    asset = out / src.name
    asset.write_bytes(dst)
    owner["child_payload_sha256"] = digest(asset)
    rec = out / receipt.name
    rec.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    np.savez_compressed(
        out / "patch.npz", idx=different, old=old[different], new=new[different]
    )
    result = dict(
        original=digest(src),
        candidate=digest(asset),
        original_receipt=digest(receipt),
        receipt=digest(rec),
        checks={
            "83": dict(
                self_pairs=counts,
                relative_volume_error=ratio,
                bed_clearance_mm=float(gap * 1000),
                pinned_tendon_source_points=len(pins),
            )
        },
        neighbors=digest(fit / "fit.json"),
        parent_surfaces_unchanged=len(owner["surfaces"]),
        parent_payloads_unchanged=len(owner["source_payloads"]),
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result["checks"], flush=True)


if __name__ == "__main__":
    emit(*(Path(a).resolve() for a in sys.argv[1:]))
