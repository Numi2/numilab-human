"""Measure which tissues occupy the paths inward from loaded physical bed nodes."""

import csv
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from material_context import sha
from measure_bed_tissue_path import closed, hits, union_length


def measure(case, bed, reader):
    job = json.loads((case / "launch.json").read_text())
    receipt = json.loads((case / "terminal.receipt.json").read_text())
    support = json.loads(bed.read_text())
    verified = json.loads((case / "verified-endpoint.json").read_text())
    if (
        support["source_launch"] != sha(case / "launch.json")
        or support["receipt"] != sha(case / "terminal.receipt.json")
        or verified["physical_nodes_sha256"] != sha(case / "physical-accepted.node")
    ):
        raise ValueError("Different physical load and coordinates")
    frame = subprocess.check_output(
        ["zstd", "-qdc", str(case / "accepted-terminal.mrvpack.zst")]
    )
    if (
        hashlib.sha256(frame).hexdigest() != receipt["pack_file_sha256"]
        or verified["frame_sha256"] != receipt["pack_file_sha256"]
    ):
        raise ValueError("Different native frame")
    env = job["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def pinned(key, pin):
        path = Path(env[prefix + key])
        if sha(path) != env[prefix + pin]:
            raise ValueError("Different " + key)
        return path

    facets = pinned("SURFACE_D_FACES_CSV", "SURFACE_D_FACES_SHA256")
    table = pinned("COMBINED_NODE_MAP_CSV", "COMBINED_NODE_MAP_SHA256")
    with table.open() as stream:
        deep = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(stream)
            if r["registered_surface_layer"] == "D"
        )
    ids, joined = np.array(deep).T
    if not np.array_equal(ids, np.arange(len(joined))) or len(np.unique(joined)) != len(
        joined
    ):
        raise ValueError("Unknown physical deep boundary")
    faces = np.loadtxt(facets, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=int)
    coords = np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    rows = {
        r["outer_material_node"]: {
            "node": r["outer_material_node"],
            "donor": r["donor_body"],
            "bed_N": r["endpoint_N"],
            "muscle": [],
            "bone": [],
            "ambiguous": [],
        }
        for r in support["largest_force_changes"]
        if r["endpoint_N"] > 0
    }
    samples = {i: coords[i, :2] for i in rows}
    deep_hits = hits(coords[joined], faces, samples)
    sys.path.insert(0, str(reader.parent / "src"))
    banks = runpy.run_path(str(reader))["primitives"]
    poses = {r["body_index"]: r for r in receipt["accepted_body_poses"]}
    tissue = Path(job["argv"][job["argv"].index("--soft-tissue-payload") + 1])
    names = json.loads(tissue.with_suffix(".manifest.json").read_text())
    if sha(tissue) != names["payload"]["sha256"]:
        raise ValueError("Changed tissue identity")
    labels = {
        r["stable_id"]: r.get("label", r.get("member_id"))
        for r in names["source"]["surfaces"]
        if r["layer"] == "muscle"
    }
    for (sem, sid), _, v, f in banks(frame, poses):
        if sem not in (51004, 51005):
            continue
        lo, hi = v[:, :2].min(0), v[:, :2].max(0)
        use = {i: q for i, q in samples.items() if ((lo <= q) & (q <= hi)).all()}
        if not use:
            continue
        intercept = hits(v, f, use, require_oriented=sem == 51005)
        sealed = sem != 51005 or not intercept or closed(v, f)
        for i, depths in intercept.items():
            if depths is None or not sealed:
                rows[i]["ambiguous"].append([sem, sid])
                continue
            if sem == 51004:
                rows[i]["bone"].extend((z, sid) for z in depths)
            else:
                rows[i]["muscle"].extend(
                    (lo, hi, sid) for lo, hi in zip(depths[::2], depths[1::2])
                )
    for i, r in rows.items():
        floor = coords[i, 2]
        deepz = deep_hits.get(i)
        bones = sorted((z, sid) for z, sid in r.pop("bone") if z > floor)
        chords = r.pop("muscle")
        if deepz is None or not bones or any(sem == 51004 for sem, _ in r["ambiguous"]):
            r["status"] = "ambiguous or missing material or cortex intersection"
            continue
        borders = [z for z in deepz if z > floor]
        if not borders:
            r["status"] = "missing deep boundary above loaded node"
            continue
        D = borders[0]
        B, sid = bones[0]
        r.update(
            skin_to_deep_mm=(D - floor) * 1000,
            skin_to_first_cortex_mm=(B - floor) * 1000,
            first_cortex_id=sid,
        )
        if B <= D:
            r["status"] = "cortex encountered before deep fat"
            continue
        chords = [
            (max(lo, D), min(hi, B), sid)
            for lo, hi, sid in chords
            if min(hi, B) > max(lo, D)
        ]
        r.update(
            status="measured",
            deep_to_cortex_mm=(B - D) * 1000,
            closed_muscle_union_mm=1000
            * union_length((lo, hi) for lo, hi, _ in chords),
            closed_muscles=[
                [sid, labels[sid], (hi - lo) * 1000] for lo, hi, sid in chords
            ],
        )
    return {
        "frame": receipt["pack_file_sha256"],
        "bed_digest": sha(bed),
        "physical_nodes": verified["physical_nodes_sha256"],
        "reader": sha(reader),
        "measured_rows": list(rows.values()),
        "sampled_barrier_N": sum(r["bed_N"] for r in rows.values()),
        "total_barrier_N": support["endpoint_barrier_N"],
        "scope": "Eight largest barrier-force changes, with positive endpoint loads. Distances are vertical intersections on the native accepted geometry. Intervening muscle chord lengths can overlap. They do not assign loads to muscles. Ambiguous banks are recorded.",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            measure(*(Path(v).resolve() for v in sys.argv[1:])),
            indent=2,
            allow_nan=False,
        )
    )
