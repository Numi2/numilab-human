"""Enumerate conforming fillings of a measured material pocket using its existing vertices."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from audit_cavity_response import elastic
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from screen_layer_spanning_cavities import oriented_faces, quality
from screen_remaining_stars import load_module


def integer_points(x):
    by_axis = []
    for col in np.asarray(x).T:
        f = [float(v).as_integer_ratio() for v in col]
        base = max(d for _, d in f)
        v = [n * (base // d) for n, d in f]
        by_axis.append([a - v[0] for a in v])
    return list(zip(*by_axis))


def det(p, t):
    a, b, c = [[p[j][k] - p[t[0]][k] for k in range(3)] for j in t[1:]]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def signed_faces(t):
    for f in oriented_faces(t):
        inversions = sum(f[i] > f[j] for i, j in combinations(range(3), 2))
        yield tuple(sorted(f)), -1 if inversions % 2 else 1


def fillings(x, old, eligible):
    pts = integer_points(x)
    boundary = Counter()
    for t in old:
        if det(pts, t) <= 0:
            raise ValueError("Invalid original cell")
        for f, sgn in signed_faces(t):
            boundary[f] += sgn
    boundary = {f: s for f, s in boundary.items() if s}
    if any(abs(s) != 1 for s in boundary.values()):
        raise ValueError("Invalid original boundary")
    total = sum(det(pts, t) for t in old)
    cells = []
    faceids = []
    vol = []
    byface = defaultdict(list)
    for t in combinations(range(len(x)), 4):
        d = det(pts, t)
        if not d or abs(d) > total:
            continue
        if d < 0:
            t = (t[1], t[0], t[2], t[3])
        fs = tuple(signed_faces(t))
        if any(f in boundary and boundary[f] != s for f, s in fs) or not eligible(t):
            continue
        i = len(cells)
        cells.append(t)
        faceids.append(fs)
        vol.append(abs(d))
        for fsig in fs:
            byface[fsig].append(i)
    results = []
    work = 0

    def visit(remaining, front, used, chosen):
        nonlocal work
        work += 1
        if not front:
            if remaining == 0:
                results.append([cells[i] for i in chosen])
            return

        def choices(pair):
            return [
                i
                for i in byface[pair]
                if vol[i] <= remaining and not any(fs in used for fs in faceids[i])
            ]

        candidates = min((choices(pair) for pair in sorted(front)), key=len)
        for i in candidates:
            nf = front.copy()
            for f, s in faceids[i]:
                if (f, s) in nf:
                    nf.remove((f, s))
                else:
                    nf.add((f, -s))
            visit(remaining - vol[i], nf, used.union(faceids[i]), chosen + [i])

    visit(total, set(boundary.items()), set(), [])
    return results, work


def screen(run, accepted, report, ledger, lib, out):
    data = json.loads(report.read_text())
    x, y, t, role, pins = inputs(run)
    x = x.astype(float)
    y = y.astype(float)
    if (
        data["geometry"] != pins
        or sha(accepted / "physical-accepted.node") != data["physical_endpoint_sha256"]
    ):
        raise ValueError("Mismatched physical pocket")
    cert = json.loads((accepted / "verified-endpoint.json").read_text())
    if cert["physical_nodes_sha256"] != data["physical_endpoint_sha256"]:
        raise ValueError("Mismatched accepted frame")
    z = np.loadtxt(accepted / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    env = json.loads((run / "launch.json").read_text())["environment"]
    props, mpins = materials(
        Path(json.loads((run / "launch.json").read_text())["argv"][0]).parent
    )
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    labels = Path(env[prefix + "CHILD_LABELS"])
    if sha(labels) != env[prefix + "CHILD_LABELS_SHA256"]:
        raise ValueError("Unpinned region labels")
    lab = np.loadtxt(
        labels, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3), dtype="i4"
    )
    if not np.array_equal(lab[:, 0], np.arange(len(t))) or not np.array_equal(
        lab[:, 1], role
    ):
        raise ValueError("Source rows and installed material roles disagree")
    composition = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if (
        sha(composition) != data["mesh_sha256"]
        or sha(ledger / "current-density-closed-flow.npz") != data["mass_sha256"]
    ):
        raise ValueError("Changed mass or topology")
    with np.load(composition) as p:
        keep = p["retained_input_cell_rows"]
    exact = load_module(lib / "fixed_boundary_volume_flip_198_root.py")
    catalog = []
    passing = []
    for group in data["joined_interface_rejections"]:
        for c in group["failed_components"]:
            core = c["retained_source_rows"]
            target = {w["source_row"] for w in c["witnesses"]}
            ids = np.unique(t[core])
            adj = np.flatnonzero(np.sum(np.isin(t, ids), axis=1) >= 3)
            # Grow across existing fat faces only. Both source lineage and material are fixed.
            face0 = {f for tet in t[core] for f, _ in signed_faces(tet)}
            nbr = [
                int(j)
                for j in adj
                if j not in core
                and np.all(lab[j, 1:] == lab[core[0], 1:])
                and any(f in face0 for f, _ in signed_faces(t[j]))
            ]
            packs = [
                list(core) + list(a)
                for n in range(min(2, len(nbr)) + 1)
                for a in combinations(nbr, n)
            ]
            summary = Counter()
            best_response = None
            for patch in packs:
                if not np.isin(patch, keep).all() or not np.all(
                    lab[patch, 1:] == lab[patch[0], 1:]
                ):
                    continue
                ids = np.unique(t[patch])
                tt = np.searchsorted(ids, t[patch])
                X, Y, Z = (v[ids] for v in (x, y, z))
                if len(ids) > 10:
                    continue
                forbidden = {tuple(sorted(np.searchsorted(ids, t[j]))) for j in target}
                options, it = fillings(
                    X,
                    tt,
                    lambda tet: (
                        tuple(sorted(tet)) not in forbidden
                        and quality(X, Y, Z, np.array([tet]), exact)
                    ),
                )
                summary["searched_patches"] += 1
                summary["visited_states"] += it
                summary["geometric_fillings"] += len(options)
                for fill in options:
                    nt = np.array(fill)
                    response = []
                    for pose in (Y, Z):
                        E, F, _ = elastic(
                            X,
                            pose,
                            tt,
                            np.arange(len(ids)),
                            *props[int(role[patch[0]]) - 15],
                        )
                        e, f, _ = elastic(
                            X,
                            pose,
                            nt,
                            np.arange(len(ids)),
                            *props[int(role[patch[0]]) - 15],
                        )
                        response.append(
                            (
                                float(abs(e - E) / max(abs(E), 1e-30)),
                                float(
                                    np.linalg.norm(f - F)
                                    / max(np.linalg.norm(F), 1e-30)
                                ),
                            )
                        )
                    score = max(max(pair) for pair in response)
                    if (
                        best_response is None
                        or score < best_response["worst_relative_error"]
                    ):
                        best_response = dict(
                            worst_relative_error=score,
                            energy_and_force_by_pose=response,
                            retired=patch,
                            new=ids[nt].tolist(),
                        )
                    if score <= 0.01:
                        passing.append(
                            dict(
                                old=patch,
                                new=ids[nt].tolist(),
                                witnesses=sorted(target),
                                matched_response=response,
                            )
                        )
            catalog.append(
                dict(
                    witnesses=sorted(target),
                    original_pocket_cells=len(core),
                    counts=summary,
                    best_geometric_candidate=best_response,
                )
            )
    out.write_text(
        json.dumps(
            dict(
                source=pins,
                registered_pockets_sha256=sha(report),
                materials=mpins,
                results=catalog,
                proposals_requiring_full_carrier_mapping=passing,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    screen(*(Path(s).resolve() for s in sys.argv[1:]))
