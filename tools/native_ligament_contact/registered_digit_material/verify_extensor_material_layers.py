"""Compare bilateral muscle surfaces with the accepted physical dermis and fat boundaries."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from extensor_registration import sha
from solve_extensor_sites import Neighbors
from verify_native_extensor_pair import loaded


def surfaces(case, reader, muscle_ids=(137, 138)):
    _, poses, data = loaded(case, initial=False, dense=True)
    sys.path.insert(0, str(reader.parent / "src"))
    pr = runpy.run_path(str(reader))["primitives"]
    return {
        sid: (v, f)
        for (kind, sid), _, v, f in pr(data, poses)
        if (kind == 51005 and sid in muscle_ids) or (kind, sid) == (51007, 1)
    }


def boundaries(case):
    report = json.loads((case / "accepted-layers.json").read_text())
    env = json.loads((case / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    nodes = case / "physical-accepted.node"
    if sha(nodes) != report["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"]:
        raise ValueError("Physical boundary audit and trial use different nodes")
    with nodes.open() as source:
        n, dim, fields, markers = map(int, next(source).split())
        values = np.loadtxt(source)
    if (
        (dim, fields, markers) != (3, 0, 0)
        or values.shape != (n, 4)
        or not np.array_equal(values[:, 0], np.arange(1, n + 1))
    ):
        raise ValueError("Incomplete physical volume")
    xyz = np.ascontiguousarray(values[:, 1:], dtype="<f8")
    if (
        hashlib.sha256(xyz.tobytes()).hexdigest()
        != report["accepted_coordinate_f64_sha256"]
    ):
        raise ValueError("Accepted material coordinate mismatch")

    def artifact(key):
        p = Path(env[prefix + key])
        if sha(p) != report["inputs"][key]:
            raise ValueError("Physical facet provenance changed at " + key)
        return p

    groups = [[], [], []]
    roles = {
        "outer_O": 0,
        "donor_skin_OI_headwall": 0,
        "inner_I": 1,
        "inner_D": 2,
        "donor_fat_ID_headwall": 2,
    }
    with artifact("FACET_ANCESTRY_CSV").open() as source:
        for row in csv.DictReader(source):
            role = row["face_source_role"]
            if (
                role != "outer_I"
            ):  # Bonded fat/skin interface occurs in both partitions.
                groups[roles[role]].append(
                    tuple(int(row[f"joint_node{k}"]) - 1 for k in range(3))
                )
    if [len(g) for g in groups] != report["triangle_counts"]:
        raise ValueError("Physical facet count differs from the installed-volume audit")
    group_arrays = [np.asarray(g, dtype=np.int32) for g in groups]
    if any(f.min() < 0 or f.max() >= n for f in group_arrays):
        raise ValueError("Invalid material node index")
    with artifact("COMBINED_NODE_MAP_CSV").open() as source:
        order = sorted(
            (int(row["registered_surface_vertex_row"]), int(row["joint_node_id"]) - 1)
            for row in csv.DictReader(source)
            if row["registered_surface_layer"] == "O"
        )
    rows, indices = np.array(order).T
    if not np.array_equal(rows, np.arange(len(rows))) or len(np.unique(indices)) != len(
        rows
    ):
        raise ValueError("Incomplete outer boundary ancestry")
    return xyz, group_arrays, xyz[indices]


def verify(control, edited, reader, predicate, muscle_ids=(137, 138)):
    banks = [surfaces(case, reader, muscle_ids) for case in (control, edited)]
    xyz, faces, outer = boundaries(control)
    if sha(control / "physical-accepted.node") != sha(
        edited / "physical-accepted.node"
    ):
        raise ValueError("Passive registration changed physical skin/fat nodes")
    if len(set(muscle_ids)) != 2 or set(banks[0]) != {1, *muscle_ids} or set(banks[1]) != set(banks[0]):
        raise ValueError("Missing muscle or skin surface")
    presented = banks[0][1][0]
    if not np.array_equal(presented, banks[1][1][0]) or presented.shape != outer.shape:
        raise ValueError("Unmatched presented outer surface")
    offset = np.linalg.norm(presented.astype(float) - outer, axis=1)
    # Both representations are in native world metres; report any offsets explicitly.
    result = dict(
        time_s=json.loads((control / "terminal.receipt.json").read_text())[
            "accepted_time_s"
        ],
        physical_nodes=len(xyz),
        physical_node_sha256=sha(control / "physical-accepted.node"),
        presented_to_physical_outer_max_mm=1000 * float(offset.max()),
        presented_to_physical_outer_rms_mm=1000 * float(np.sqrt(np.mean(offset**2))),
        banks={},
    )
    for sid in muscle_ids:
        pts = [bank[sid][0] for bank in banks]
        if not np.array_equal(banks[0][sid][1], banks[1][sid][1]):
            raise ValueError("Changed extensor topology")
        box = np.vstack(pts)
        lo, hi = box.min(0), box.max(0)
        nearby = []
        for i, f in enumerate(faces):
            tri = xyz[f]
            possible = ((tri.max(1) >= lo) & (tri.min(1) <= hi)).all(1)
            if possible.any():
                idx, inv = np.unique(f[possible], return_inverse=True)
                nearby.append((i + 1, i, xyz[idx], inv.reshape(-1, 3), 1.0))
        scene = Neighbors(nearby, banks[0][sid][1], pts[0], predicate)
        counts = {}
        for key, pt in zip(("control", "edited"), pts):
            pairs, _ = scene.evaluate(pt)
            kind = scene.kind[pairs[:, 0]]
            counts[key] = {
                name: int(np.sum(kind == i))
                for i, name in enumerate(
                    ("self", "outer_skin", "inner_skin", "deep_fat")
                )
            }
        result["banks"][sid] = counts
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "edited", "reader", "predicate"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument("--muscle-ids", type=int, nargs=2, default=(137, 138))
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
