"""Relate terminal envelope folds to the exact NHTISS4 source influences and accepted native frame."""

import json, struct, sys, runpy, subprocess, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from material_context import sha
from compare_hand_interfaces import query


def transfer(x, w, rot, shift, scale):
    transforms = np.stack([r.apply(x) * s + t for r, t, s in zip(rot, shift, scale)])
    return np.einsum("vbi, vb -> vi", transforms.transpose(1, 0, 2), w)


def audit(run, reader, exe, out):
    argv = json.loads((run / "launch.json").read_text())
    args = argv["argv"]
    asset = Path(args[args.index("--soft-tissue-payload") + 1])
    meta = json.loads(asset.with_suffix(".manifest.json").read_text())
    src = asset.read_bytes()
    ident = hashlib.sha256(src).hexdigest()
    if ident != argv["pins"][str(asset)] or ident != meta["payload"]["sha256"]:
        raise ValueError("NHTISS4 identity mismatch")
    magic, abi, nr, nb, nv, ni, _ = struct.unpack_from("<8s6I", src)
    if (magic, abi) != (b"NHTISS4\0", 5):
        raise ValueError("Unsupported envelope")
    records = np.frombuffer(src, "<u4", 8 * nr, 64).reshape(-1, 8)
    vp = 64 + 32 * nr + 36 * nb
    ip = vp + 56 * nv
    admission = json.loads((run / "admission.json").read_text())
    step = admission["accepted_steps"]
    receipt = json.loads(
        (run / f"native/accepted-geometry/step-{step}.receipt.json").read_text()
    )
    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-qdc", str(run / "terminal.mrvpack.zst")]
    )
    if (
        hashlib.sha256(raw).hexdigest() != receipt["pack_file_sha256"]
        or sha(run / "terminal.mrvpack.zst") != admission["terminal_archive_sha256"]
    ):
        raise ValueError("Pack identity mismatch")
    poses = {r["body_index"]: r for r in receipt["accepted_body_poses"]}
    ends = {
        sid: (x, t)
        for (sem, sid), _, x, t in runpy.run_path(str(reader))["primitives"](raw, poses)
        if sem == 51005 and sid in (137, 138)
    }
    result = {}
    for sid, (y, f) in ends.items():
        matching = records[records[:, 6] == sid]
        if len(matching) != 1:
            raise ValueError("Invalid extensor identity")
        bi, count, vi, n, i0, ic, *_ = map(int, matching[0])
        srcfaces = np.frombuffer(src, "<u4", ic, ip + 4 * i0).reshape(-1, 3) - vi
        used = np.unique(srcfaces)
        if not np.array_equal(np.searchsorted(used, srcfaces), f):
            raise ValueError("Envelope topology differs")
        rows = np.frombuffer(src, "<f4", 14 * n, vp + 56 * vi).reshape(-1, 14)[used]
        x = rows[:, :3].astype(float)
        index = rows[:, 6:10].copy().view("<u4")
        weights = rows[:, 10:14].astype(float)
        w = np.zeros((len(x), count))
        v, j = np.nonzero(weights > 0)
        np.add.at(w, (v, index[v, j]), weights[v, j])
        if np.max(abs(w.sum(1) - 1)) > 1e-5:
            raise ValueError("Unnormalized native influences")
        rots = []
        shifts = []
        scales = []
        body = []
        for i in range(count):
            b, *q = struct.unpack_from("<I8f", src, 64 + 32 * nr + 36 * (bi + i))
            B = poses[b]
            rb = Rotation.from_quat(B["quaternion_xyzw"])
            body.append(b)
            rots.append(rb * Rotation.from_quat(q[3:7]))
            shifts.append(rb.apply(q[:3]) + B["position_m"])
            scales.append(q[7])
        reconstructed = transfer(x, w, rots, shifts, scales)
        err = np.linalg.norm(reconstructed - y, axis=1)
        surface = lambda points: query(exe, points, f, np.ones(len(f), bool))
        original = surface(x)
        actual = surface(y)
        bodylabels = [
            v["myosim_body"]
            for v in next(
                s for s in meta["source"]["surfaces"] if s["stable_id"] == sid
            )["body_bindings"]
        ]
        np.argmax(w, axis=1)
        affected = np.unique(np.array(sorted(actual), dtype=int).ravel())
        owners = {}
        for face in affected:
            own = bodylabels[np.argmax(w[f[face]].mean(0))]
            owners[own] = owners.get(own, 0) + 1
        info = dict(
            maximum_reconstruction_discrepancy_m=float(err.max()),
            rms_reconstruction_discrepancy_m=float(np.mean(err**2) ** 0.5),
            source_self_pairs=len(original),
            native_self_pairs=len(actual),
            unique_affected_faces=len(affected),
            dominant_body_for_affected_faces=owners,
            body_ids=body,
            body_names=bodylabels,
        )
        if err.max() > 2e-6:
            raise ValueError("Registration owner mismatch: " + str(info))
        np.savez_compressed(
            out / f"extensor-{sid}.npz",
            x=x,
            y=y,
            f=f,
            weights=w,
            rot=np.array([r.as_quat() for r in rots]),
            shift=shifts,
            scale=scales,
            body=body,
            source_rows=used,
        )
        result[sid] = info
        print(sid, info, flush=True)
    (out / "extensor-transfer.json").write_text(
        json.dumps(
            dict(source=ident, pack=receipt["pack_file_sha256"], banks=result), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    audit(*(Path(v).resolve() for v in sys.argv[1:]))
