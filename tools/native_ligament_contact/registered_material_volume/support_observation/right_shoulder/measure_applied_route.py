"""Reconcile actual contact rows with the GPU's reduced rigid wrenches and accepted physical frame."""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest


def get(p):
    return json.loads(p.read_bytes())


def trace(case):
    with (case / "native-traces/resting-coupled.csv").open() as f:
        return list(csv.DictReader(f))


def rows(case, i):
    path = case / f"contact-{i}.rows"
    a = np.fromfile(path, "<f4").reshape(-1, 7, 4)
    ids = a[:, 0].copy().view("<u4")
    order = np.argsort(ids[:, 0])
    return ids[order], a[order, 1:].astype(float)


def summarize(case, control, vol, baseline, out):
    execinfo = get(case / "execution.json")
    r = get(case / "launch.json")
    base = get(baseline / "route.json")
    if execinfo["exit"] != 0 or r["route_control"] != digest(control / "launch.json"):
        raise ValueError("Wrong source or incomplete coupling")
    c = trace(case)
    steps = [int(a["step"]) for a in c]
    if steps != list(range(1, len(steps) + 1)) or len(steps) > 4:
        raise ValueError("Unmatched accepted transactions")
    receipt = get(
        next((case / "native-traces/accepted-geometry").glob("*.receipt.json"))
    )
    if (
        receipt["accepted_step"] != steps[-1]
        or abs(receipt["accepted_time_s"] - float(c[-1]["time_s"])) > 1e-12
    ):
        raise ValueError("Unmatched accepted endpoint")
    counts = np.fromfile(case / "contact-counts.u32", "<u4")
    if (
        len(counts) != 4
        or (counts[: len(c)] == 0).any()
        or counts.max() > 8192
        or counts[len(c) :].any()
    ):
        raise ValueError("Overflow or incomplete source-step records")
    wrenches = (
        np.fromfile(case / "contact-wrenches.f32", "<f4")
        .astype(float)
        .reshape(4, -1, 2, 4)
    )
    bodies = len(receipt["accepted_body_poses"])
    if wrenches.shape[1] != bodies:
        raise ValueError("Different skeletal domain")
    err = (case / "stderr.log").read_text()
    log = (case / "stdout.log").read_text()
    span = re.search(
        r"registered_material_span object=\d+ element_offset=\d+ element_count=\d+ state_offset=(\d+) state_count=(\d+)",
        err,
    )
    offset, n = map(int, span.groups())
    end = np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    opening = np.loadtxt(vol / "current.node", skiprows=1, usecols=(1, 2, 3))
    if len(end) != n or len(opening) != n:
        raise ValueError("Wrong material domain")
    center = base["source_node"]
    normal = np.array(get(control / "support-check.json")["bed_normal"])
    walls = np.load(vol / "material-walls.npz")
    tags = np.zeros(n, dtype="u1")
    for k, flag in [("O", 1), ("I", 2), ("D", 4), ("C", 8)]:
        if k in walls:
            tags[np.unique(walls[k])] |= flag
    dist = np.linalg.norm(end - end[center], axis=1)
    states = []
    scapula = 34
    if base["cortex"]["10"]["nearest_sample"] != 311173:
        raise ValueError("Different registered shoulder")
    for i in range(len(c)):
        ids, z = rows(case, i)
        if (
            len(z) != counts[i]
            or len(set(ids[:, 0])) != len(z)
            or not np.isfinite(z).all()
        ):
            raise ValueError("Invalid or repeated source rows")
        point, contact, tissue, rigid, torque, origin = z.transpose(1, 0, 2)
        body = ids[:, 2]
        node = ids[:, 1].astype("i8") - offset
        inskin = (node >= 0) & (node < n)
        local = np.clip(node, 0, n - 1)
        if np.any(body >= bodies):
            raise ValueError("Unknown anatomical body")
        f, t = tissue[:, :3], rigid[:, :3]
        bforce = np.zeros((bodies, 3))
        bmom = bforce.copy()
        np.add.at(bforce, body, t)
        np.add.at(bmom, body, torque[:, :3])
        ferr = float(np.linalg.norm(bforce - wrenches[i, :, 0, :3], axis=1).max())
        merr = float(np.linalg.norm(bmom - wrenches[i, :, 1, :3], axis=1).max())
        closure = float(np.linalg.norm(f + t, axis=1).max())
        row_moment = float(
            np.linalg.norm(
                torque[:, :3] - np.cross(contact[:, :3] - origin[:, :3], t), axis=1
            ).max()
        )
        if max(ferr, closure) > 1e-5 or max(merr, row_moment) > 2e-6:
            raise ValueError(
                ("Native reaction mismatch", ferr, merr, closure, row_moment)
            )

        def metric(mask):
            return dict(
                rows=int(mask.sum()),
                resultant_on_bone_N=t[mask].sum(0).tolist(),
                bone_bed_normal_N=float(t[mask].sum(0) @ normal),
                sum_force_magnitudes_N=float(np.linalg.norm(t[mask], axis=1).sum()),
                torque_about_body_center_Nm=torque[mask, :3].sum(0).tolist()
                if len(set(body[mask])) <= 1
                else None,
            )

        shells = {}
        for rad in [23, 50, 100, 160]:
            region = inskin & (dist[local] <= rad * 0.001)
            shells[str(rad)] = dict(
                all_receivers=metric(region), scapula=metric(region & (body == scapula))
            )
        states.append(
            dict(
                source_step=i,
                rows=len(z),
                paired_force_max_N=closure,
                body_reduction_max_error_N=ferr,
                body_moment_max_error_Nm=merr,
                row_moment_max_error_Nm=row_moment,
                by_body={str(int(k)): metric(body == k) for k in np.unique(body)},
                skin_fat_all=metric(inskin),
                skin_fat_to_scapula=metric(inskin & (body == scapula)),
                within_radius_mm=shells,
                scapula_material_sample_rows=[
                    dict(
                        node=int(node[u]),
                        flags=int(ids[u, 3]),
                        wall_mask=int(tags[local[u]]),
                        distance_from_bed_node_m=float(dist[local[u]]),
                        force_N=t[u].tolist(),
                    )
                    for u in np.flatnonzero(inskin & (body == scapula))
                ],
            )
        )
    bed_rows = list(
        csv.DictReader((case / "physical-posterior-bed-history.csv").open())
    )
    bed_imp = sum(float(a["terminal_microstep_normal_impulse_Ns"]) for a in bed_rows)
    m = re.search(
        r"resting_fem_skin_bed_accepted_root_reaction normal_impulse_Ns=([^ ]+) mean_normal_force_N=([^ ]+)[^\n]+root_dt_s=([^ ]+)",
        log,
    )
    if (
        abs(bed_imp - float(m[1])) > 1e-8
        or abs(bed_imp / float(m[3]) - float(m[2])) > 1e-4
    ):
        raise ValueError("Different bed reaction clocks")
    deltas = {
        k: max(abs(float(x[k]) - float(y[k])) for x, y in zip(trace(control), c))
        for k in c[0]
    }
    result = dict(
        control=digest(control / "launch.json"),
        launch=digest(case / "launch.json"),
        source_steps=states,
        accepted_receipt=digest(
            next((case / "native-traces/accepted-geometry").glob("*.receipt.json"))
        ),
        emitted_end=digest(case / "physical-accepted.node"),
        material_endpoint_identical_to_control=digest(case / "physical-accepted.node")
        == digest(control / "physical-accepted.node"),
        source_bed_node=center,
        bed_N=float(m[2]),
        bed_active_nodes=len(bed_rows),
        bed_center_node_displacement_m=(end[center] - opening[center]).tolist(),
        scapula_closest_sample=311173,
        scapula_sample_displacement_m=(end[311173] - opening[311173]).tolist(),
        max_trace_deltas=deltas,
        integrated=next(
            a
            for a in log.splitlines()
            if a.startswith("resting_integrated_body=completed")
        ),
        cold_wall_s=execinfo["wall_s"],
        scope="Row forces assembled from the accepted source state of each step. Bed impulse belongs to the end of the accepted root. A static force is not causal evidence of bed load transmission.",
    )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        "bed",
        result["bed_N"],
        "skin/fat to scapula",
        states[-1]["skin_fat_to_scapula"],
        "50 mm",
        states[-1]["within_radius_mm"]["50"]["all_receivers"],
        flush=True,
    )


if __name__ == "__main__":
    summarize(*(Path(p).resolve() for p in sys.argv[1:]))
