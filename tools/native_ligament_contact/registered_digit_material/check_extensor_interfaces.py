"""Compare both re-bound envelopes to every nearby native structure, including all physical skin/fat layers."""

import collections, hashlib, json, runpy, subprocess, sys
from pathlib import Path
import numpy as np
from compare_hand_interfaces import query
from material_context import sha


def run(root, native, reader, exe):
    a = json.loads((native / "admission.json").read_text())
    rc = json.loads(
        (
            native / f"native/accepted-geometry/step-{a['accepted_steps']}.receipt.json"
        ).read_text()
    )
    poses = {r["body_index"]: r for r in rc["accepted_body_poses"]}
    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-qdc", str(native / "terminal.mrvpack.zst")]
    )
    if (
        hashlib.sha256(raw).hexdigest() != rc["pack_file_sha256"]
        or sha(native / "terminal.mrvpack.zst") != a["terminal_archive_sha256"]
    ):
        raise ValueError("Unmatched native frame")
    auth = json.loads((root / "extensor-transfer.json").read_text())
    if auth["pack"] != rc["pack_file_sha256"]:
        raise ValueError("Different registration frame")
    banks = (137, 138)
    versions = ["6", "12", "24"]
    allx = []
    allf = []
    keys = []
    change = []
    offset = 0
    bounds = []
    trials = [[] for _ in versions]
    for sid in banks:
        src = np.load(root / f"extensor-{sid}.npz")
        v = src["y"]
        f = src["f"]
        e = [np.load(root / f"branch-{sid}-{n}.npz")["x"] for n in versions]
        envelope = np.concatenate([v, *e])
        bounds.append((envelope.min(0), envelope.max(0)))
        for i, vv in enumerate(e):
            trials[i].append(vv)
        allx.append(v)
        allf.append(f + offset)
        keys.extend([f"51005/{sid}"] * len(f))
        change.extend([True] * len(f))
        offset += len(v)

    def append(name, v, f):
        nonlocal offset
        if not any(
            np.all(v.min(0) <= hi) and np.all(v.max(0) >= lo) for lo, hi in bounds
        ):
            return
        lo = v[f].min(1)
        hi = v[f].max(1)
        keep = np.zeros(len(f), bool)
        for a, b in bounds:
            keep |= np.all(lo <= b, 1) & np.all(hi >= a, 1)
        if not keep.any():
            return
        ix = np.unique(f[keep])
        allx.append(v[ix])
        allf.append(np.searchsorted(ix, f[keep]) + offset)
        offset += len(ix)
        for t in trials:
            t.append(v[ix])
        keys.extend([name] * int(keep.sum()))
        change.extend([False] * int(keep.sum()))

    layers = np.load(root / "physical.npz")
    for layer in "OID":
        append(layer, layers[layer + "v"], layers[layer + "f"])
    for (category, sid), _, v, f in runpy.run_path(str(reader))["primitives"](
        raw, poses
    ):
        if category != 51007 and (category != 51005 or sid not in banks):
            append(f"{category}/{sid}", v, f)
    faces = np.concatenate(allf)
    changed = np.array(change)

    def split(pairs):
        return dict(
            collections.Counter(" ".join(sorted((keys[a], keys[b]))) for a, b in pairs)
        )

    old = query(exe, np.concatenate(allx), faces, changed)
    result = dict(baseline=split(old), trials={})
    for width, points in zip(versions, trials):
        current = query(exe, np.concatenate(points), faces, changed)
        result["trials"][width] = dict(
            total=len(current),
            new=len(current - old),
            cleared=len(old - current),
            by_bank=split(current),
            new_by_bank=split(current - old),
            cleared_by_bank=split(old - current),
        )
        print(
            width,
            len(old),
            len(current),
            len(current - old),
            len(old - current),
            flush=True,
        )
    result.update(
        pack_sha256=rc["pack_file_sha256"],
        face_count=len(faces),
        predicate_sha256=sha(exe),
        trial_widths_mm=list(map(float, versions)),
    )
    (root / "branch-interfaces.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))
