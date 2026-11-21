"""Resolve competing anatomical contact strips under the current rigid poses."""

import json, sys
from pathlib import Path
import numpy as np
from certify_material_branch import certify, tauten, unit
from shoulder_frames import setup, at_angle


def candidates(bones, ends, attachments, sid, strips):
    solutions = []
    for strip_index, edges in enumerate(strips):
        seed = [(sid, e, 0.5) for e in edges]
        pts, ts = tauten(list(bones.values()), ends, seed)
        if np.linalg.norm(np.diff(pts, axis=0), axis=1).min() < 1e-9:
            continue
        path = [(sid, e, float(t)) for e, t in zip(edges, ts)]
        c = certify(list(bones.values()), pts, path, ts, attachments)
        if (
            c["crossings"]
            or c["inside"]
            or c["normal_cone_residual_per_unit_tension"] > 1e-6
        ):
            continue
        u = unit(np.diff(pts, axis=0))
        grad = np.vstack([-u[0], u[:-1] - u[1:], u[-1]])
        moment = np.cross(pts[1:], grad[1:]).sum(0)
        solutions.append(
            dict(
                strip=strip_index,
                coordinates=ts.tolist(),
                length_m=c["length_m"],
                moment=moment.tolist(),
                cone=c["normal_cone_residual_per_unit_tension"],
            )
        )
    return sorted(solutions, key=lambda s: (s["length_m"], s["strip"]))


def run(paths):
    row, b, ep, ids = setup(*paths[:7])
    features = json.loads(paths[7].read_text())
    from shoulder_inputs import digest

    if features["atlas_sha256"] != digest(paths[3]) or features[
        "sites_sha256"
    ] != digest(paths[4]):
        raise ValueError("Feature hints and the anatomical source differ")
    catalog = sorted(
        set(
            tuple(int(e) for e, _ in edges)
            for _, _, edges in features["rows"][1]["samples"]
        )
    )
    sid = ids[1]
    raw = [(ax, float(a)) for ax, a, _ in features["rows"][1]["samples"]]
    angles = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1, 0]
    commanded = [(0, 0)] + [(axis, angle) for axis in range(3) for angle in angles]
    if raw != commanded or len(catalog) != len(set(catalog)):
        raise ValueError("The complete prescribed angular cycle is required")
    if any(
        not chain or any(e < 0 or e >= len(b[sid].edges) for e in chain)
        for chain in catalog
    ):
        raise ValueError("Invalid source edge in material route")
    grid = sorted(set(raw) | {(1, float(a)) for a in np.linspace(-5, 0, 81)})
    states = []
    lookup = {}
    for axis, angle in grid:
        bb, ends = at_angle(b, ep, ids, axis, angle)
        sol = candidates(bb, ends, row["attachments"], sid, catalog)
        states.append(dict(axis=axis, degrees=angle, solutions=sol))
        if sol:
            lookup[axis, angle] = sol[0]
    out = paths[8]
    out.write_text(
        json.dumps(dict(catalog=catalog, states=states), separators=(",", ":")) + "\n"
    )
    print(
        "strips",
        len(catalog),
        "samples",
        len(grid),
        "qualified",
        len(lookup),
        "misses",
        [s for s in grid if s not in lookup],
        flush=True,
    )
    if all(k in lookup for k in raw):
        seq = []
        for k, a in raw:
            s = lookup[k, a]
            seq.append(
                [k, a, [[e, t] for e, t in zip(catalog[s["strip"]], s["coordinates"])]]
            )
        features["rows"][1]["samples"] = seq
        paths[9].write_text(json.dumps(features, separators=(",", ":")) + "\n")
    if len(lookup) != len(grid):
        raise ValueError("A requested shoulder pose failed physical contact")
    for prev, now in zip(states, states[1:]):
        if (
            now["axis"] != 1
            or prev["axis"] != now["axis"]
            or not (prev["solutions"] and now["solutions"])
        ):
            continue
        a, d = prev["solutions"][0], now["solutions"][0]
        if a["strip"] != d["strip"]:
            print(
                "switch",
                prev["degrees"],
                now["degrees"],
                a["strip"],
                d["strip"],
                "length change",
                d["length_m"] - a["length_m"],
                flush=True,
            )


if __name__ == "__main__":
    run(list(map(Path, sys.argv[1:])))
