"""Audit executed APL routes against cortical solids at each retained mechanical sample."""

import csv
import json
import struct
import sys
from pathlib import Path
import igl
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from forearm_route import asset, read
from rebind_ligament_material import banks


def pose_samples(case):
    table = case / "native-traces/resting-body-motion-partition.csv"
    frames = {}
    with table.open() as stream:
        for row in csv.DictReader(stream):
            body = int(row["body_index"])
            step = int(row["accepted_step"])
            position = [
                float(row[f"root_{v}_m"]) + float(row[f"com_relative_root_{v}_m"])
                for v in "xyz"
            ]
            quat = [float(row[f"orientation_{v}"]) for v in "xyzw"]
            dst = frames.setdefault(step, {})
            if body in dst:
                raise ValueError("Duplicate native body pose")
            dst[body] = (np.array(position), Rotation.from_quat(quat))
    if sorted(frames) != list(range(32, 8001, 32)):
        raise ValueError("Incomplete motion samples")
    for p in sorted((case / "native-traces/accepted-geometry").glob("*.receipt.json")):
        r = read(p)
        n = r["accepted_step"]
        if n not in frames:
            continue
        for b in r["accepted_body_poses"]:
            x, R = frames[n][b["body_index"]]
            if (
                np.linalg.norm(x - b["position_m"]) > 1e-6
                or (R.inv() * Rotation.from_quat(b["quaternion_xyzw"])).magnitude()
                > 2e-6
            ):
                raise ValueError("Native frame mismatch")
    return frames


def actuator(cfg, mi):
    raw = Path(cfg["argv"][2]).read_bytes()
    h = struct.unpack_from("<8s9I32s", raw)
    ro = 76 + 16 * h[4] + 64 * h[5]
    _, start, n, _ = struct.unpack_from("<4I", raw, ro + 16 * h[6] + 164 * mi)
    pts = []
    body = []
    for j in range(start, start + n):
        kind, idx, _, _ = struct.unpack_from("<4I", raw, ro + 16 * j)
        if kind != 1:
            raise ValueError("Expected straight radial-compartment guides")
        b, *p = struct.unpack_from("<I3f", raw, 76 + 16 * idx)
        body.append(b)
        pts.append(p)
    if len(set(body)) != 2 or n != 8:
        raise ValueError("Different route topology")
    return body, np.array(pts)


def segments_in_solid(tree, v, f, p):
    d = np.diff(p, axis=0)
    L = np.linalg.norm(d, axis=1)
    hits = tree.intersect_ray(
        v, f, np.ascontiguousarray(p[:-1]), np.ascontiguousarray(d / L[:, None])
    )
    loc = []
    segments = []
    for i, (group, dist) in enumerate(zip(hits, L)):
        if np.any(np.maximum(p[i], p[i + 1]) < v.min(0)) or np.any(
            np.minimum(p[i], p[i + 1]) > v.max(0)
        ):
            continue
        cuts = [0.0]
        for t in sorted(x[1] for x in group if 1e-10 < x[1] < dist - 1e-10):
            if t - cuts[-1] > 1e-9:
                cuts.append(t)
        cuts.append(dist)
        for a, b in zip(cuts, cuts[1:]):
            loc.append(p[i] + (a + b) * 0.5 / dist * d[i])
            segments.append((i, a / dist, b / dist, (b - a)))
    if not loc:
        return []
    inside = abs(igl.winding_number(v, f, np.ascontiguousarray(loc))) > 0.5
    return [e for e, ok in zip(segments, inside) if ok]


def run(case, out, proposal=None):
    c = read(case / "launch.json")
    frames = pose_samples(case)
    payload = Path(c["argv"][3])
    solids = []
    for (b, sid), (tri, x, _) in banks(payload).items():
        u, inv = np.unique(x, axis=0, return_inverse=True)
        f = np.ascontiguousarray(inv[tri], dtype="i8")
        u = np.ascontiguousarray(u) * 1000
        bb = igl.AABB()
        bb.init(u, f)
        solids.append((b, sid, u, f, u.min(0), u.max(0), bb))
    result = []
    for mi, ends in [(259, [16, 88]), (322, [17, 107])]:
        bodies, pts = actuator(c, mi)
        if proposal is not None:
            from registration_stages import parts

            sid = 147 if mi == 259 else 148
            fit = read(proposal / f"{sid}.json")
            material = read(proposal / "source-launch.json")
            if fit["launch"] != digest(proposal / "source-launch.json") or digest(
                Path(material["argv"][3])
            ) != digest(payload):
                raise ValueError("Preflight geometry source differs")
            bodies, pts = actuator(material, mi)
            _, _, donors, _, aff = parts(asset(material, "--soft-tissue-payload"), sid)
            pts[[0, -1]] = fit["endpoint_local_m"]
            for j, d in enumerate(fit["source_delta_mm"], start=1):
                pts[j] = np.asarray(
                    pts[j] + aff[donors.index(bodies[j])][0] @ np.asarray(d) * 0.001,
                    dtype="f4",
                ).astype(float)
        events = []
        lengths = []
        moment = []
        nc = 0
        worst = 0.0
        max_terminal = 0.0
        invalid_count = 0
        for n, poses in frames.items():
            q = np.asarray(
                [poses[b][1].apply(x) + poses[b][0] for b, x in zip(bodies, pts)]
            )
            lengths.append(np.linalg.norm(np.diff(q, axis=0), axis=1).sum() * 1000)
            # Net thumb moment per unit tension about the native thumb body origin.
            u = q[4] - q[5]
            u /= np.linalg.norm(u)
            b = bodies[5]
            moment.append(
                1000 * poses[b][1].inv().apply(np.cross(q[5] - poses[b][0], u))
            )
            cache = {}
            failed = False
            for owner, bank, x, f, lo, hi, tree in solids:
                if owner not in cache:
                    pos, rot = poses[owner]
                    cache[owner] = rot.inv().apply(q - pos) * 1000
                t = cache[owner]
                if np.any(t.max(0) < lo) or np.any(t.min(0) > hi):
                    continue
                nc += 1
                for i, a, b, mm in segments_in_solid(tree, x, f, t):
                    terminal = (bank == ends[0] and i == 0 and a == 0) or (
                        bank == ends[-1] and i == 6 and b == 1
                    )
                    if terminal and mm <= 0.001:
                        max_terminal = max(max_terminal, mm)
                        continue
                    failed = True
                    worst = max(worst, mm)
                    events.append(
                        dict(
                            step=n,
                            bank=bank,
                            segment=i,
                            start_fraction=a,
                            end_fraction=b,
                            length_mm=mm,
                        )
                    )
            invalid_count += int(failed)
        result.append(
            dict(
                muscle=mi,
                samples=len(frames),
                samples_with_penetration=invalid_count,
                total_intervals=len(events),
                tested_bank_pose_pairs=nc,
                maximum_interval_mm=worst,
                maximum_terminal_roundoff_mm=max_terminal,
                witnesses=sorted(events, key=lambda s: -s["length_mm"]),
                length_range_mm=[min(lengths), max(lengths)],
                thumb_moment_per_N_about_body_origin_mm_min=np.min(
                    moment, axis=0
                ).tolist(),
                thumb_moment_per_N_about_body_origin_mm_max=np.max(
                    moment, axis=0
                ).tolist(),
            )
        )
    out.write_text(
        json.dumps(
            dict(
                launch=digest(case / "launch.json"),
                motion_series=digest(
                    case / "native-traces/resting-body-motion-partition.csv"
                ),
                bone=digest(payload),
                muscle=digest(Path(c["argv"][2])),
                rows=result,
                proposal=None
                if proposal is None
                else [digest(proposal / f"{sid}.json") for sid in [147, 148]],
                scope="Every retained 32nd accepted step versus all AABB-overlapping registered cortical surfaces. The submicrometer endpoint convention applies only to the cortical owner at the endpoint itself. Skin requires a separately retained mesh capture.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(p).resolve() for p in sys.argv[1:]))
