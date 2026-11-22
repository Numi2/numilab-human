"""Compare actual accepted material/anatomy pairs at the same native physical time."""

from pathlib import Path
import csv, json, runpy, subprocess, hashlib, sys
import numpy as np
from compare_hand_interfaces import query
from material_context import sha


def measure(native, reference, patch, reader, exe):
    job = json.loads((native / "launch.json").read_text())
    a = json.loads((native / "admission.json").read_text())
    step = a["accepted_steps"]
    rc = json.loads(
        (native / f"native/accepted-geometry/step-{step}.receipt.json").read_text()
    )
    if (
        a["exit_code"]
        or step != rc["accepted_step"]
        or a["terminal_pack_sha256"] != rc["pack_file_sha256"]
        or sha(native / "terminal.mrvpack.zst") != a["terminal_archive_sha256"]
    ):
        raise ValueError("Unmatched physical endpoint")
    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-qdc", str(native / "terminal.mrvpack.zst")]
    )
    if hashlib.sha256(raw).hexdigest() != rc["pack_file_sha256"]:
        raise ValueError("Mismatch during accepted pack recovery")
    poses = {r["body_index"]: r for r in rc["accepted_body_poses"]}
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    env = job["environment"]

    def pin(k, h):
        p = Path(env[prefix + k])
        if sha(p) != env[prefix + h]:
            raise ValueError("Unbound input " + k)
        return p

    data = np.load(reference / "physical.npz")
    mapfile = pin("COMBINED_NODE_MAP_CSV", "COMBINED_NODE_MAP_SHA256")
    offsets = {key: np.full(len(data[key + "rows"]), -1, dtype=int) for key in "OID"}
    with mapfile.open() as f:
        for row in csv.DictReader(f):
            key = row["registered_surface_layer"]
            if key in offsets:
                vid = int(row["registered_surface_vertex_row"])
                oid = int(row["joint_node_id"]) - 1
                if offsets[key][vid] not in (-1, oid):
                    raise ValueError("Multivalued layer transfer")
                offsets[key][vid] = oid
    xyz = np.loadtxt(native / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    faces = {}
    point = {}
    mutable = {}
    active = patch["nodes"]
    boxes = []
    for key in "OID":
        if not np.array_equal(offsets[key], data[key + "rows"]):
            raise ValueError("Different material connectivity")
        file = pin("SURFACE_" + key + "_FACES_CSV", "SURFACE_" + key + "_FACES_SHA256")
        t = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype="i4")
        if not np.array_equal(t, data[key + "f"]):
            raise ValueError("Changed facet identities")
        x = xyz[offsets[key]]
        keep = np.any(np.isin(offsets[key][t], active), axis=1)
        vertices = x[t[keep]].reshape(-1, 3)
        for side in (-1, 1):
            q = vertices[side * vertices[:, 0] > 0]
            if len(q):
                boxes.append((q.min(0), q.max(0)))
        faces[key] = t
        point[key] = x
        mutable[key] = keep
    if not boxes:
        raise ValueError("No physical boundary moved")
    allx = []
    allf = []
    edited = []
    label = []
    start = 0

    def add(key, x, t, m):
        nonlocal start
        if not any(np.all(x.min(0) <= b) and np.all(x.max(0) >= a) for a, b in boxes):
            return
        tri = x[t]
        lo = tri.min(1)
        hi = tri.max(1)
        relevant = np.zeros(len(t), bool)
        for a, b in boxes:
            relevant |= np.all(lo <= b, axis=1) & np.all(hi >= a, axis=1)
        if np.any(m & ~relevant):
            raise ValueError("Dropped a physical facet")
        rows = np.flatnonzero(relevant)
        ids = np.unique(t[rows])
        allx.append(x[ids])
        allf.append(start + np.searchsorted(ids, t[rows]))
        edited.append(m[rows])
        start += len(ids)
        label.extend((key, int(i)) for i in rows)

    for key in "OID":
        add(key, point[key], faces[key], mutable[key])
    parity = None
    for (sem, bank), _, x, t in runpy.run_path(str(reader))["primitives"](raw, poses):
        if sem == 51007:
            if bank == 1:
                # Viewer uses the indexed subset of the compiled O boundary.
                used = np.unique(faces["O"])
                ref_faces = np.searchsorted(used, faces["O"])
                if not np.array_equal(ref_faces, t):
                    raise ValueError("Presented O topology mismatch")
                parity = float(
                    np.linalg.norm(x.astype(float) - point["O"][used], axis=1).max()
                )
            continue
        add(f"{sem}/{bank}", x, t, np.zeros(len(t), bool))
    if parity is None or parity > 2e-7:
        raise ValueError(("Accepted display differs from physics", parity))
    pairs = query(
        exe, np.concatenate(allx), np.concatenate(allf), np.concatenate(edited)
    )
    return {tuple(sorted((label[i], label[j]))) for i, j in pairs}, dict(
        step=step,
        time_s=rc["accepted_time_s"],
        pack=rc["pack_file_sha256"],
        accepted_material=sha(native / "physical-accepted.node"),
        maximum_render_error_m=parity,
        changed_face_count=int(np.concatenate(edited).sum()),
    )


def main(old, new, root, case, reader, exe):
    q = np.load(case / "field.npz")
    before, original = measure(old, root, q, reader, exe)
    after, repaired = measure(new, root, q, reader, exe)
    if original["step"] != repaired["step"] or original["time_s"] != repaired["time_s"]:
        raise ValueError("Physical horizons differ")
    targets = json.loads((case / "interfaces.json").read_text())["cleared_witnesses"]
    desired = {tuple(sorted(((a, b), (c, d)))) for a, b, c, d in targets}
    result = dict(
        source=original,
        repaired=repaired,
        before=len(before),
        after=len(after),
        cleared=sorted(before - after),
        new=sorted(after - before),
        targeted_cleared=len(desired & (before - after)),
        targeted_pairs=len(desired),
        scope="Every changed boundary face against all nearby accepted anatomical banks",
    )
    (new / "matched-digital-interfaces.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        result["before"],
        result["after"],
        len(result["new"]),
        result["targeted_cleared"],
        flush=True,
    )


if __name__ == "__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
