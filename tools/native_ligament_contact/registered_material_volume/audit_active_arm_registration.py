"""Locate the dominant arm surface defects before and after registration at a native endpoint."""

import hashlib, json, runpy, struct, subprocess, sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

run, impl, output = map(Path, sys.argv[1:])
ad = json.loads((run / "admission.json").read_text())
n = ad["accepted_steps"]
receipt = json.loads(
    (run / "native/accepted-geometry" / f"step-{n}.receipt.json").read_text()
)
job = json.loads((run / "launch.json").read_text())
args = job["argv"]
asset = Path(args[args.index("--soft-tissue-payload") + 1])
descriptor = asset.with_suffix(".manifest.json")
manifest = json.loads(descriptor.read_text())
physical = asset.read_bytes()
if sha(asset) != job["pins"][str(asset)] or sha(asset) != manifest["payload"]["sha256"]:
    raise ValueError("Unmatched muscle source")
raw, abi, nr, nb, nv, ni, _ = struct.unpack_from("<8s6I", physical)
assert raw == b"NHTISS4\0" and abi == 5
v0 = 64 + 32 * nr + 36 * nb
i0 = v0 + 56 * nv
assert len(physical) == i0 + ni * 4
rec = np.frombuffer(physical, "<u4", 8 * nr, 64).reshape(-1, 8)
xyz = np.frombuffer(physical, "<f4", 14 * nv, v0).reshape(-1, 14)[:, :3]
indices = np.frombuffer(physical, "<u4", ni, i0)
archive = run / "terminal.mrvpack.zst"
assert sha(archive) == ad["terminal_archive_sha256"]
frame = subprocess.check_output(["/opt/homebrew/bin/zstd", "-q", "-dc", str(archive)])
assert (
    hashlib.sha256(frame).hexdigest()
    == receipt["pack_file_sha256"]
    == ad["terminal_pack_sha256"]
)
sys.path.insert(0, str(impl.parent / "src"))
aud = runpy.run_path(str(impl))
poses = {r["body_index"]: r for r in receipt["accepted_body_poses"]}
geo = json.loads((run / "envelope/accepted-envelope.json").read_text())
assert geo["pack_sha256"] == receipt["pack_file_sha256"]
selected = dict(
    sorted(
        [(i, n) for s, i, _, n in geo["affected_banks"] if s == 51005],
        key=lambda k: -k[1],
    )[:4]
)
labels = {r["stable_id"]: r for r in manifest["source"]["surfaces"]}


def exact(points, faces):
    records = aud["clearance"]._exact_surface_records(points, faces)
    tree = aud["intersections"]._prepare_surface_aabb(records)
    result = aud["intersections"]._audit_pair_prepared_first(
        tree, records, same_surface=True
    )
    return set(map(tuple, result["triangle_pairs"]))


result = []
for (semantic, bank), _, posed, tri in aud["primitives"](frame, poses):
    if semantic != 51005 or bank not in selected:
        continue
    _, _, first, count, start, width, _, _ = map(int, rec[rec[:, 6] == bank].reshape(8))
    face = (indices[start : start + width] - first).reshape(-1, 3)
    used = np.unique(face)
    assert int(used.max()) < count and np.array_equal(tri, np.searchsorted(used, face))
    source = np.ascontiguousarray(xyz[first : first + count][used])
    posed = np.ascontiguousarray(posed)
    # The same physical corner must coincide in both poses before counting it as a duplicate seam.
    _, reps, inv = np.unique(
        np.c_[source, posed], axis=0, return_index=True, return_inverse=True
    )
    t = inv[tri]
    assert np.array_equal(source[reps][t], source[tri]) and np.array_equal(
        posed[reps][t], posed[tri]
    )
    sa = exact(source[reps], t)
    sb = exact(posed[reps], t)
    result.append(
        dict(
            bank_id=bank,
            structure=labels[bank]["label"],
            triangles=len(t),
            coincident_aliases=len(source) - len(reps),
            source_self_pairs=len(sa),
            native_self_pairs=len(sb),
            new_self_pairs=len(sb - sa),
            cleared_self_pairs=len(sa - sb),
            external_skin_pairs=selected[bank],
            new_self_pair_witnesses=sorted(sb - sa),
        )
    )
assert len(result) == 4
output.write_text(
    json.dumps(
        dict(
            scope="Source NHTISS4 positions and the accepted pose on identical indexed faces",
            source_sha256=sha(asset),
            source_manifest_sha256=sha(descriptor),
            accepted_pack_sha256=receipt["pack_file_sha256"],
            accepted_step=n,
            structures=result,
        ),
        indent=2,
    )
    + "\n"
)
for r in result:
    print(
        r["structure"],
        r["source_self_pairs"],
        r["native_self_pairs"],
        r["external_skin_pairs"],
        flush=True,
    )
