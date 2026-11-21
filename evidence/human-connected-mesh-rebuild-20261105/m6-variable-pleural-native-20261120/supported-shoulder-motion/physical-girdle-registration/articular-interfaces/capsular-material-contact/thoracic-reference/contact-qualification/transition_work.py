"""Locate equal-length material routes and integrate their moment on the changing branch."""

import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from certify_material_branch import coords
from verify_articular_motion import adjacent_hits, adjacent_surfaces
from solve_retained_strips import setup, at_angle, candidates


def stations(p):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return np.column_stack(
        [np.interp(np.linspace(0, s[-1], 257), s, p[:, j]) for j in range(3)]
    )


def run(paths):
    launch, ref, motion, archive, sites, heads, spacing, torso, table, out = paths
    row, b, ep, ids = setup(*paths[:7])
    sid = ids[1]
    doc = json.loads(table.read_text())
    catalog = doc["catalog"]
    cfg = json.loads(launch.read_text())
    manifest = json.loads(
        Path(cfg["argv"][3]).with_suffix(".manifest.json").read_text()
    )
    center = next(r for r in json.loads(heads.read_text())["rows"] if r["side"] == "r")[
        "variants"
    ][0]["atlas_center_m"]
    offset = next(
        r for r in json.loads(spacing.read_text())["rows"] if r["side"] == "r"
    )["translation_atlas_axes_m"]
    scale = json.loads(torso.read_text())["scale"]
    near = {}
    for bank, (v, f) in adjacent_surfaces(
        archive, manifest["source"]["anchors"]
    ).items():
        world = (v - center) * scale - offset
        near[bank] = world, f, world.min(0), world.max(0)

    def points(axis, angle, num, value):
        bb, ends = at_angle(b, ep, ids, axis, angle)
        return coords(
            bb, ends, [(sid, e, t) for e, t in zip(catalog[num], value["coordinates"])]
        )

    def evaluate(axis, angle, num):
        bb, ends = at_angle(b, ep, ids, axis, angle)
        found = candidates(bb, ends, row["attachments"], sid, [catalog[num]])
        if not found:
            raise ValueError(f"Strip {num} lost equilibrium at {angle}")
        v = found[0]
        p = points(axis, angle, num, v)
        if adjacent_hits(p, near):
            raise ValueError("Adjacent source bone intersects material path")
        return v, p

    for sample in doc["states"]:
        best = sample["solutions"][0]
        p = points(sample["axis"], sample["degrees"], best["strip"], best)
        if adjacent_hits(p, near):
            raise ValueError("Neighbor collision in angular scan")
    seq = [p for p in doc["states"] if p["axis"] == 1 and -5 <= p["degrees"] <= 0]
    events = []
    parts = []
    start = -5.0
    active = seq[0]["solutions"][0]["strip"]
    for left, right in zip(seq, seq[1:]):
        incoming = left["solutions"][0]["strip"]
        outgoing = right["solutions"][0]["strip"]
        if incoming == outgoing:
            continue

        def difference(a):
            return (
                evaluate(1, a, incoming)[0]["length_m"]
                - evaluate(1, a, outgoing)[0]["length_m"]
            )

        root = brentq(difference, left["degrees"], right["degrees"], xtol=2e-10)
        u, p = evaluate(1, root, incoming)
        v, q = evaluate(1, root, outgoing)
        events.append(
            dict(
                axis=1,
                degrees=root,
                incoming=incoming,
                outgoing=outgoing,
                length_error_m=abs(u["length_m"] - v["length_m"]),
                centerline_gap_m=float(
                    np.linalg.norm(stations(p) - stations(q), axis=1).max()
                ),
                moment_jump_m_per_unit_tension=v["moment"][1] - u["moment"][1],
                max_cone_error=max(u["cone"], v["cone"]),
            )
        )
        parts.append((start, root, active))
        start = root
        active = outgoing
    parts.append((start, 0.0, active))
    quadrature = []
    for lower, upper, num in parts:
        gauss, gw = np.polynomial.legendre.leggauss(6)
        mid = (upper + lower) / 2
        width = (upper - lower) / 2
        moment = [evaluate(1, mid + width * q, num)[0]["moment"][1] for q in gauss]
        work = float(np.radians(width) * np.dot(gw, moment))
        delta = (
            evaluate(1, upper, num)[0]["length_m"]
            - evaluate(1, lower, num)[0]["length_m"]
        )
        quadrature.append(
            dict(
                lo_deg=lower,
                hi_deg=upper,
                strip=num,
                length_change_m=delta,
                integrated_moment_m=work,
                error_m=abs(delta - work),
            )
        )
    result = dict(
        source_bone_neighbor_count=len(near),
        sampled_right_poses_with_neighbors=len(doc["states"]),
        domain_degrees=[-5.0, 0.0],
        axis=1,
        events=events,
        integrals=quadrature,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    run(list(map(Path, sys.argv[1:])))
