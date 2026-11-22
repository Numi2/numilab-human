"""Audit every moved boundary face against itself, other material layers and all nearby native banks."""

import collections, csv, hashlib, json, runpy, sys, subprocess
from pathlib import Path
import numpy as np
from compare_hand_interfaces import query
from material_context import sha


def run(root, run, case, impl, exe, *, prepare=False):
    job = json.loads((run / "launch.json").read_text())
    terminal = json.loads((run / "admission.json").read_text())
    n = terminal["accepted_steps"]
    capture = json.loads(
        (run / f"native/accepted-geometry/step-{n}.receipt.json").read_text()
    )
    assert sha(run / "terminal.mrvpack.zst") == terminal["terminal_archive_sha256"]
    raw = subprocess.check_output(
        ["/opt/homebrew/bin/zstd", "-qdc", str(run / "terminal.mrvpack.zst")]
    )
    assert (
        hashlib.sha256(raw).hexdigest()
        == capture["pack_file_sha256"]
        == json.loads((root / "geometry.json").read_text())["identities"]["native_pack"]
    )
    api = runpy.run_path(str(impl))
    poses = {r["body_index"]: r for r in capture["accepted_body_poses"]}
    e = job["environment"]
    pref = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    x = np.loadtxt(run / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    patch = np.load(case / "field.npz")
    if (
        sha(run / "physical-accepted.node")
        != json.loads((run / "release-skin-parity.json").read_text())[
            "accepted_nodes_sha256"
        ]
    ):
        raise ValueError("Incorrect material instance")
    data = np.load(root / "physical.npz")
    nodes = patch["nodes"]
    dx = np.zeros(len(x))
    dx[nodes] = patch["delta_x_m"]
    active = np.flatnonzero(dx != 0)
    spans = []
    if prepare:
        from fit_dorsal_material import make_field

        active = np.flatnonzero(
            make_field(x, json.loads((root / "geometry.json").read_text())) != 0
        )
    eligible = np.isin(np.arange(len(x)), active)
    for mask in (x[active, 0] < 0, x[active, 0] >= 0):
        region = x[active[mask]]
        other = region.copy()
        other[:, 0] += dx[active[mask]]
        both = np.vstack([region, other])
        spans.append((both.min(0) - 0.003, both.max(0) + 0.003))
    maps = {r: np.full(len(data[r + "v"]), -1, int) for r in "OID"}
    mp = Path(e[pref + "COMBINED_NODE_MAP_CSV"])
    assert sha(mp) == e[pref + "COMBINED_NODE_MAP_SHA256"]
    with mp.open() as f:
        for row in csv.DictReader(f):
            k = row["registered_surface_layer"]
            if k not in maps:
                continue
            i = int(row["registered_surface_vertex_row"])
            v = int(row["joint_node_id"]) - 1
            if maps[k][i] not in (-1, v):
                raise ValueError("Ambiguous layer")
            maps[k][i] = v
    parts = []
    ys = []
    ts = []
    ms = []
    labels = []
    source_ids = []
    node_ids = []
    nn = 0

    def append(key, v, w, f, m, volume_rows=None):
        nonlocal nn
        va = v[f]
        wa = w[f]
        lo = np.minimum(va.min(1), wa.min(1))
        hi = np.maximum(va.max(1), wa.max(1))
        keep = np.zeros(len(f), bool)
        for l, h in spans:
            keep |= np.all(lo <= h, axis=1) & np.all(hi >= l, axis=1)
        if np.any(m & ~keep):
            raise ValueError("Dropped moved facet")
        if not np.any(keep):
            return
        ids = np.unique(f[keep])
        parts.append(v[ids])
        ys.append(w[ids])
        node_ids.append(
            np.full(len(ids), -1, dtype="i4")
            if volume_rows is None
            else volume_rows[ids]
        )
        ts.append(nn + np.searchsorted(ids, f[keep]))
        ms.append(m[keep])
        nn += len(ids)
        labels.extend([key] * int(keep.sum()))
        source_ids.extend(np.flatnonzero(keep).tolist())

    for layer in "OID":
        rows = maps[layer]
        assert min(rows) >= 0 and np.array_equal(x[rows], data[layer + "v"])
        f = data[layer + "f"]
        v = x[rows].copy()
        w = v.copy()
        w[:, 0] += dx[rows]
        append(layer, v, w, f, np.any(eligible[rows][f], axis=1), rows)
    for (sem, sid), _, v, f in api["primitives"](raw, poses):
        if sem == 51007:
            continue
        if not any(np.all(v.min(0) <= h) and np.all(v.max(0) >= l) for l, h in spans):
            continue
        append(f"{sem}/{sid}", v, v, f, np.zeros(len(f), bool))
    before, after = np.concatenate(parts), np.concatenate(ys)
    f = np.concatenate(ts)
    changed = np.concatenate(ms)
    old = query(exe, before, f, changed)
    if prepare:
        keys, groups = np.unique(labels, return_inverse=True)
        np.savez_compressed(
            root / "contact-domain.npz",
            x=before,
            f=f.astype("i4"),
            changed=changed,
            gid=groups.astype("i4"),
            keys=keys,
            material=np.concatenate(node_ids).astype("i4"),
            source_face=np.array(source_ids, dtype="i4"),
            baseline=np.array(sorted(old), dtype="i4"),
            accepted_material_sha256=sha(run / "physical-accepted.node"),
            pack_sha256=capture["pack_file_sha256"],
        )
        return
    new = query(exe, after, f, changed)

    def pairs(rows):
        return collections.Counter(
            tuple(sorted((labels[a], labels[b]))) for a, b in rows
        )

    def counters(rows):
        return {" ".join(k): int(v) for k, v in sorted(pairs(rows).items())}

    def summarize(rows):
        s = collections.Counter()
        for (a, b), v in pairs(rows).items():
            k = tuple(sorted((a.split("/")[0], b.split("/")[0])))
            s[" ".join(k)] += v
        return dict(s)

    def witness(rows):
        return [
            [labels[a], source_ids[a], labels[b], source_ids[b]]
            for a, b in sorted(rows)
        ]

    report = dict(
        native_pack=capture["pack_file_sha256"],
        accepted_nodes=sha(run / "physical-accepted.node"),
        field_sha256=sha(case / "field.npz"),
        scope="counterfactual displacement on accepted material nodes against every nearby native bank, complete for moved boundary triangles",
        considered_faces=len(f),
        moved_faces=int(changed.sum()),
        before=len(old),
        after=len(new),
        new=len(new - old),
        cleared=len(old - new),
        before_by_kind=summarize(old),
        after_by_kind=summarize(new),
        new_by_kind=summarize(new - old),
        cleared_by_kind=summarize(old - new),
        new_by_bank=counters(new - old),
        cleared_by_bank=counters(old - new),
        new_witnesses=witness(new - old),
        cleared_witnesses=witness(old - new),
    )
    (case / "interfaces.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n"
    )
    print(
        report["before"],
        report["after"],
        report["new"],
        report["cleared"],
        report["new_by_kind"],
        report["cleared_by_kind"],
        flush=True,
    )


if __name__ == "__main__":
    run(
        *(Path(v).resolve() for v in sys.argv[1:6]),
        prepare=sys.argv[6:] == ["--prepare"],
    )
