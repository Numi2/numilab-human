"""Fit local material vertices against both native snapshots; keep the outer boundary fixed."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha


def metric(v):
    b, c, d = (v[..., i, :] - v[..., 0, :] for i in (1, 2, 3))
    det = np.sum(b * np.cross(c, d), axis=-1)
    longest = np.maximum.reduce(
        [np.sum(edge * edge, axis=-1) for edge in (b, c, d, c - b, d - b, d - c)]
    )
    return det / 6.0, det / (longest * np.sqrt(longest))


def coords(path):
    with path.open() as src:
        n, dim, attributes, markers = map(int, next(src).split())
        if (dim, attributes, markers) != (3, 0, 0):
            raise ValueError("Expected registered TetGen XYZ rows")
        rows = np.loadtxt(src, dtype="f8")
    if (
        rows.shape != (n, 4)
        or not np.array_equal(rows[:, 0], np.arange(1, n + 1))
        or not np.isfinite(rows).all()
    ):
        raise ValueError("Invalid native coordinates or node order in " + str(path))
    return np.ascontiguousarray(rows[:, 1:])


def main():
    ap = argparse.ArgumentParser()
    for key in ("source", "reference", "accepted", "witness_log", "flow", "output"):
        ap.add_argument(key, type=Path)
    ap.add_argument("--allow-density-routing", action="store_true")
    ap.add_argument(
        "--motion",
        nargs=2,
        type=Path,
        action="append",
        default=[],
        metavar=("REFERENCE", "ACCEPTED"),
    )
    ap.add_argument("--scale", default=1.0, type=float)
    ap.add_argument("--seed-patch", type=Path)
    ap.add_argument("--freeze-boundaries", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    mesh = a.source / "mesh"
    original = coords(mesh / "mesh.node")
    rest = coords(a.reference)
    endpoint = coords(a.accepted)
    # The row match is a mechanical input, so any reindexing must be resolved before fitting.
    if not np.array_equal(original, rest):
        raise ValueError(
            ("native/authoring reference mismatch", float(np.max(abs(original - rest))))
        )
    with (mesh / "mesh.ele").open() as fh:
        n = int(next(fh).split()[0])
        elements = np.loadtxt(fh, max_rows=n, usecols=(1, 2, 3, 4, 5), dtype="i4")
    t, role = elements[:, :4] - 1, elements[:, 4]
    witness = {}
    for s in a.witness_log.read_text().splitlines():
        if s.startswith("registered_cell_shape_witness "):
            d = dict(p.split("=") for p in s.split()[1:])
            i = int(d["local"])
            ids = [int(d["node" + str(k)]) for k in range(4)]
            if int(d["material"]) != role[i] or not np.array_equal(t[i], ids):
                raise ValueError(("Native connectivity mismatch", i, ids, t[i]))
            witness[i] = d
    seeds = np.array(sorted(witness), dtype="i4")
    if not len(seeds) and not (a.motion or a.scale != 1.0 or a.seed_patch):
        raise ValueError("No authenticated failing cell")
    mass = np.load(a.flow)["element_mass_kg"]
    lo = np.where(role == 15, 900.0, 850.0)
    hi = np.where(role == 15, 1100.0, 950.0)
    x = original.copy()
    du = endpoint - rest
    motions = [du] + [coords(b) - coords(k) for k, b in a.motion]
    if not 1.0 <= a.scale <= 4:
        raise ValueError("Invalid motion scale")
    if a.scale != 1.0:
        motions += [v * a.scale for v in motions]
    if a.seed_patch:
        with np.load(a.seed_patch) as patch:
            x[patch["nodes"]] = patch["x"]
    if a.seed_patch or a.scale != 1.0 or a.motion:
        unsafe = set()
        for i in range(0, len(t), 30000):
            rows = t[i : i + 30000]
            v = x[rows]
            bad = np.zeros(len(v), dtype=bool)
            for samples in [v] + [v + m[rows] for m in motions]:
                vol, shape = metric(samples)
                bad |= (vol <= 1e-18) | (shape < 1e-5)
            unsafe.update((np.flatnonzero(bad) + i).tolist())
        seeds = np.asarray(sorted(unsafe), dtype="i4")
    before_local = x.copy()
    seedset = set(map(int, seeds))
    free = np.unique(t[seeds])
    region = {int(i): "" for i in free}
    with (a.source / "material/registration/combined-node-map.csv").open() as src:
        for row in csv.DictReader(src):
            node = int(row["joint_node_id"]) - 1
            if node in region and row["registered_surface_layer"]:
                region[node] = row["registered_surface_layer"]
    frozen = set(np.load(a.seed_patch)["nodes"].tolist()) if a.seed_patch else set()
    groups = {
        i: np.flatnonzero((t == i).any(1))
        for i in free
        if int(i) not in frozen
        and (not region[int(i)] if a.freeze_boundaries else region[int(i)] != "O")
    }
    all_cells = (
        np.unique(np.concatenate(list(groups.values())))
        if groups
        else np.empty(0, dtype="i4")
    )

    # Use measured displacement as a local counterfactual only. A fresh native solve decides promotion.
    def measure(y, rows):
        v0, q0 = metric(y)
        margin = np.minimum(v0 / 1e-18, q0 / 1e-5)
        for m in motions:
            v1, q1 = metric(y + m[t[rows]])
            margin = np.minimum(margin, np.minimum(v1 / 1e-18, q1 / 1e-5))
        return margin, v0

    def scores(y, rows):
        s, v = measure(y, rows)
        target = np.where(np.isin(rows, seeds), 1.02, 1.0)
        return np.maximum(target - s, 0), s, v

    before, ss, _ = scores(x[t[all_cells]], all_cells)
    if set(all_cells[ss < 1]) != seedset:
        raise ValueError(
            (
                "Failing cells without a permitted movable incident node",
                sorted(seedset - set(all_cells[ss < 1])),
            )
        )
    steps = []
    lookup = {v: i for i, v in enumerate(all_cells)}
    dirs = np.array(
        [
            (x, y, z)
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
            if (x, y, z) != (0, 0, 0)
        ],
        dtype="f8",
    )
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    radii = np.geomspace(1e-9, 2e-5, 28)
    for epoch in range(5):
        improved = False
        for cell in seeds:
            old_def, _, _ = scores(x[t[all_cells]], all_cells)
            if old_def[lookup[cell]] <= 1e-12:
                continue
            best = None
            for node in t[cell]:
                if node not in groups:
                    continue
                rows = groups[node]
                corner = np.argmax(t[rows] == node, axis=1)
                pts = x[t[rows]]
                px = x[t[cell]]
                others = np.delete(px, int(np.flatnonzero(t[cell] == node)[0]), axis=0)
                norm = np.cross(others[1] - others[0], others[2] - others[0])
                norm /= np.linalg.norm(norm)
                test_dirs = np.vstack([dirs, norm, -norm])
                ps = (
                    (x[node] + (radii[:, None, None] * test_dirs).reshape(-1, 3))
                    .astype("f4")
                    .astype("f8")
                )
                ps = np.unique(ps, axis=0)
                previous = old_def[[lookup[i] for i in rows]]
                for start in range(0, len(ps), 32):
                    pos = ps[start : start + 32]
                    batch = np.broadcast_to(pts, (len(pos), *pts.shape)).copy()
                    batch[:, np.arange(len(rows)), corner, :] = pos[:, None, :]
                    deficit, margin, vol = scores(batch, rows)
                    # All cells reaching a construction floor remain above it, including adjacent material.
                    good = np.all(deficit <= previous + 1e-12, axis=1)
                    if not a.allow_density_routing:
                        density = mass[rows] / vol
                        good &= np.all(
                            (density >= lo[rows]) & (density <= hi[rows]), axis=1
                        )
                    merit = np.sum(previous) - np.sum(deficit, axis=1)
                    for i in np.flatnonzero(good & (merit > 1e-12)):
                        disp = float(np.linalg.norm(pos[i] - original[node]))
                        rank = (
                            round(float(merit[i]), 10),
                            region[int(node)] == "",
                            -disp,
                        )
                        if best is None or rank > best[0]:
                            best = (rank, int(node), pos[i].copy())
            if best is not None:
                rank, node, point = best
                x[node] = point
                steps.append(
                    dict(
                        seed=int(cell),
                        node=node,
                        role=region[node],
                        shift_m=float(np.linalg.norm(point - original[node])),
                    )
                )
                improved = True
        if not improved:
            break
    final, s, v = scores(x[t[all_cells]], all_cells)
    moved = np.flatnonzero(np.any(x != before_local, axis=1))
    density = mass[all_cells] / v
    out = a.output
    assert not any(region[int(k)] == "O" for k in moved)
    changed_rows = all_cells[np.isin(t[all_cells], moved).any(1)]
    rows_with_density_change = all_cells[
        ((density < lo[all_cells]) | (density > hi[all_cells]))
        & np.isin(all_cells, changed_rows)
    ]
    np.savez_compressed(
        out / "proposal.npz",
        nodes=moved.astype("i4"),
        x=x[moved].astype("f4"),
        cell=changed_rows.astype("i4"),
    )
    margs = {str(i): float(s[lookup[i]]) for i in seeds}
    report = dict(
        physical_source_sha256=sha(mesh / "mesh.node"),
        reference_sha256=sha(a.reference),
        accepted_sha256=sha(a.accepted),
        witness_log_sha256=sha(a.witness_log),
        proposal_sha256=sha(out / "proposal.npz"),
        failed_cells_initial=seeds.tolist(),
        failed_cells_frozen_endpoint=all_cells[s < 1].tolist(),
        seeded_floors_margin_ratio=margs,
        changed_nodes=moved.tolist(),
        changed_outer_nodes=0,
        affected_tets=len(changed_rows),
        maximum_shift_m=float(
            max(np.linalg.norm(x[moved] - original[moved], axis=1), default=0)
        ),
        fixed_mass_density_violations=rows_with_density_change.tolist(),
        steps=steps,
        requires_new_native_replay=True,
    )
    if a.motion or a.scale != 1.0 or a.seed_patch:
        report.update(
            motion_scale=a.scale,
            earlier_capture_pairs=[[sha(k), sha(b)] for k, b in a.motion],
            seed_patch_sha256=sha(a.seed_patch) if a.seed_patch else None,
            all_material_boundary_nodes_fixed=a.freeze_boundaries,
        )
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        seeds.tolist(),
        report["failed_cells_frozen_endpoint"],
        len(moved),
        report["maximum_shift_m"],
        len(rows_with_density_change),
        flush=True,
    )


if __name__ == "__main__":
    main()
