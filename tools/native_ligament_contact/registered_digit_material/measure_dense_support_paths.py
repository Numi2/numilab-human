"""Measure which tissues occupy the paths inward from loaded physical bed nodes."""

import argparse
import csv
import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from material_context import sha
from measure_bed_tissue_path import closed, hits, union_length


def nearest_feature(points, faces, point, limit):
    """Prune by bounding boxes, retaining every triangle that could improve the result."""
    from verify_digit_virtual_work import point_triangle

    points = points.astype(float, copy=False)
    extent = np.maximum(np.maximum(points.min(0) - point, point - points.max(0)), 0)
    if np.dot(extent, extent) > limit**2:
        return None
    # The frame reader removes unreferenced vertices for each primitive.
    radius = min(limit, float(np.sqrt(np.min(np.sum((points - point) ** 2, axis=1)))))
    # Outward padding avoids excluding a face touching this floating-point bound.
    pad = (
        64
        * np.finfo(float).eps
        * max(1.0, abs(radius), np.max(abs(point)), np.max(abs(points)))
    )
    tri = points[faces]
    box = np.maximum(np.maximum(tri.min(1) - point, point - tri.max(1)), 0)
    ids = np.flatnonzero(np.sum(box * box, axis=1) <= (radius + pad) ** 2)
    winner = None
    for i in ids:
        t = tri[i]
        cross = np.cross(t[1] - t[0], t[2] - t[0])
        if np.any(cross != 0):
            q, _ = point_triangle(point, t)
        else:
            ends = np.roll(t, -1, axis=0)
            edges = ends - t
            length = np.sum(edges * edges, axis=1)
            projection = np.divide(
                np.sum((point - t) * edges, axis=1),
                length,
                out=np.zeros(3),
                where=length > 0,
            )
            on_edges = t + np.clip(projection, 0, 1)[:, None] * edges
            q = on_edges[np.argmin(np.sum((on_edges - point) ** 2, axis=1))]
        distance = float(np.linalg.norm(q - point))
        if distance <= limit and (winner is None or distance < winner[0]):
            winner = (distance, int(i), q.tolist())
    return winner


def measure(case, bed, reader, all_banks=False):
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
    atlas = {}
    intercostals = set()
    if all_banks:
        argv = job["argv"]
        rc = Path(argv[argv.index("--resting-anatomy-receipt") + 1])
        payload = Path(argv[argv.index("--torso-anatomy-payload") + 1])
        if sha(rc) != job["pins"][str(rc)] or sha(payload) != job["pins"][str(payload)]:
            raise ValueError("Different registered anatomy sources")
        contents = json.loads(rc.read_text())
        if contents["payload"]["sha256"] != job["pins"][str(payload)]:
            raise ValueError("Anatomical names belong to a different payload")
        atlas = contents["provenance"]["source_id_map"]
        intercostals = set(contents["functional_bindings"]["intercostal_stable_ids"])
    queries = {}
    nearest = {i: {} for i in rows}
    if all_banks:
        for i in rows:
            upper = [z for z in (deep_hits.get(i) or []) if z > coords[i, 2]]
            if upper:
                queries[i] = np.array([*samples[i], upper[0]])
    if all_banks:
        from numilab_human.capsular_material_registration import (
            material_bank,
            read_material_banks,
        )

        binary = Path(job["argv"][0])
        owner = binary.parent
        identities = json.loads((owner / "identities.json").read_text())
        header = owner / "NumiHumanLoadedSkinPoseData.inc"
        if (
            sha(header) != identities["loaded_pose_override"]["header_sha256"]
            or sha(binary) != identities["binary_sha256"]
            or sha(binary) != job["pins"][str(binary)]
        ):
            raise ValueError("Different attachment owner")
        m = re.search(
            r"constexpr Anchor anchors\[\]=\{(.*?)\};", header.read_text(), re.DOTALL
        )
        if m is None:
            raise ValueError("Missing compiled attachment table")
        anchors = [
            tuple(map(int, row))
            for row in re.findall(r"\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,", m[1])
        ]
        with pinned(
            "DEEP_ATTACHMENT_REQUESTS_CSV", "DEEP_ATTACHMENT_REQUESTS_SHA256"
        ).open() as stream:
            source_pairs = sorted(
                (int(r["stable_attachment_id"]), int(r["donor_body"]))
                for r in csv.DictReader(stream)
            )
        if (
            not anchors
            or sorted((a, c) for a, b, c in anchors) != source_pairs
            or not all(0 <= b < len(coords) for _, b, _ in anchors)
        ):
            raise ValueError("Compiled anchors differ from registered requests")
        for i, point in queries.items():
            distance, stable, node, donor = min(
                (float(np.linalg.norm(coords[b] - point)), a, b, c)
                for a, b, c in anchors
            )
            rows[i]["closest_registered_deep_attachment"] = {
                "stable": stable,
                "volume_node": node,
                "donor_body": donor,
                "distance_m": distance,
            }
        bone_payload = Path(job["argv"][3])
        if sha(bone_payload) != job["pins"][str(bone_payload)]:
            raise ValueError("Different skeletal source")
        bone_banks = read_material_banks(bone_payload)
    intervals = {i: [] for i in rows}
    unresolved = {i: [] for i in rows}
    scanned = 0
    for (sem, sid), primitive, v, f in banks(frame, poses):
        if sem == 51007 or (not all_banks and sem not in (51004, 51005)):
            continue
        scanned += 1
        if all_banks:
            kind = "cortex" if sem == 51004 else "registered_soft_surface"
            for i, point in queries.items():
                prior = nearest[i].get(kind)
                result = nearest_feature(
                    v, f, point, prior["distance_m"] if prior else np.inf
                )
                if result and (prior is None or result[0] < prior["distance_m"]):
                    distance, triangle, world = result
                    name = (
                        atlas.get(str(sid), {}).get("name")
                        if sem == 51010
                        else labels.get(sid)
                        if sem == 51005
                        else None
                    )
                    nearest[i][kind] = {
                        "semantic": sem,
                        "stable_id": sid,
                        "primitive": primitive,
                        "triangle": triangle,
                        "distance_m": distance,
                        "world_m": world,
                        "source_name": name,
                    }
        lo, hi = v[:, :2].min(0), v[:, :2].max(0)
        use = {i: q for i, q in samples.items() if ((lo <= q) & (q <= hi)).all()}
        if not use:
            continue
        intercept = hits(v, f, use, require_oriented=sem != 51004)
        sealed = sem == 51004 or not intercept or closed(v, f)
        for i, depths in intercept.items():
            if depths is None or not sealed:
                if sem in (51004, 51005):
                    rows[i]["ambiguous"].append([sem, sid])
                if all_banks and sem != 51004:
                    unresolved[i].append(
                        (
                            float(v[:, 2].min()),
                            float(v[:, 2].max()),
                            sem,
                            sid,
                            primitive,
                        )
                    )
                continue
            if sem == 51004:
                rows[i]["bone"].extend((z, sid) for z in depths)
            elif all_banks:
                for low, high in zip(depths[::2], depths[1::2]):
                    intervals[i].append((low, high, sem, sid, primitive))
            if sem == 51005:
                rows[i]["muscle"].extend(
                    (lo, hi, sid) for lo, hi in zip(depths[::2], depths[1::2])
                )
    if all_banks and scanned != verified["checked_banks"]:
        raise ValueError("Column analysis and whole-body audit use different anatomical banks")
    for i, r in rows.items():
        if all_banks and i in queries:
            r["deep_query_world_m"] = queries[i].tolist()
            r["nearest_registered_surfaces"] = nearest[i]
            if "cortex" in nearest[i]:
                c = nearest[i]["cortex"]
                sid = c["stable_id"]
                record = material_bank(bone_banks, sid, int(bone_banks[2][sid][0]))[0]
                c.update(
                    myosim_body=record["myosim_body"],
                    source_member=record["member_id"],
                    source_sha256=record["member_sha256"],
                )

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
        if all_banks:
            spans = [
                (max(lo, D), min(hi, B), sem, sid, prim)
                for lo, hi, sem, sid, prim in intervals[i]
                if lo < B and hi > D
            ]
            r["registered_solid_chords"] = [
                {
                    "semantic": sem,
                    "stable_id": sid,
                    "primitive": prim,
                    "start_from_deep_mm": 1e3 * (lo - D),
                    "end_from_deep_mm": 1e3 * (hi - D),
                }
                for lo, hi, sem, sid, prim in sorted(spans)
            ]
            for entry in r["registered_solid_chords"]:
                sid = entry["stable_id"]
                if entry["semantic"] == 51010 and str(sid) in atlas:
                    record = atlas[str(sid)]
                    entry.update(
                        source_name=record["name"],
                        source_member=record["source_member"],
                        source_sha256=record["source_sha256"],
                        thoracic_intercostal_binding=sid in intercostals,
                    )
                elif entry["semantic"] == 51005 and sid in labels:
                    entry["source_name"] = labels[sid]
            r["registered_solid_union_mm"] = 1e3 * union_length(
                (lo, hi) for lo, hi, *_ in spans
            )
            r["unresolved_solid_candidates"] = [
                {"semantic": sem, "stable_id": sid, "primitive": prim}
                for lo, hi, sem, sid, prim in unresolved[i]
                if lo <= B and hi >= D
            ]
    return {
        **(
            {
                "registered_banks_scanned": scanned,
                "all_bank_scope": "Closed oriented solids strictly between the deep-fat boundary and first cortex on these vertical rays. Unsealed and ambiguous sources are listed where their height limits overlap. A geometric interval alone does not transmit a mechanical force.",
            }
            if all_banks
            else {}
        ),
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
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("case", "bed", "reader"):
        parser.add_argument(key, type=lambda val: Path(val).resolve())
    parser.add_argument("--all-banks", action="store_true")
    print(json.dumps(measure(**vars(parser.parse_args())), indent=2, allow_nan=False))
