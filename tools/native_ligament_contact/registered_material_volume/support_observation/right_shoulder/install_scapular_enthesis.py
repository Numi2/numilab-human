"""Register a route-private cortical attachment without changing same-body segment length or force law."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest
from register_deltoid_origins import pack_envelope


def build(case, fit, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    args = cfg["argv"]
    mp = Path(args[2])
    tp = Path(args[args.index("--tendon-payload") + 1])
    tm = tp.with_suffix(".manifest.json")
    r = json.loads((fit / "candidates.json").read_bytes())
    side = json.loads((fit / "facing.json").read_bytes())
    if side["source"] != digest(fit / "candidates.json") or any(
        digest(p) != r[k]
        for p, k in [(mp, "muscle"), (tp, "tendon"), (tm, "tendon_manifest")]
    ):
        raise ValueError("Source revision mismatch")
    options = [
        q
        for q in side["rows"]
        if all(q["proven_source_sites"]) and q["max_cortical_gap_m"] <= 1e-6
    ]
    if not options:
        raise ValueError("No source-verified scapular attachment")
    chosen = options[0]["proposal"]
    proposal = r["proposals"][chosen]
    e = proposal["envelope"]
    p = np.array(proposal["point"], dtype="<f4").astype(float)
    old = np.array(r["old_site"])
    via = np.array(r["via"])
    source = bytearray(mp.read_bytes())
    t = bytearray(tp.read_bytes())
    m = json.loads(tm.read_bytes())
    off = 76 + 16 * r["source_site_id"]
    original = struct.unpack_from("<I3f", source, off)
    if (
        original[0] != 34
        or not np.array_equal(original[1:], old)
        or e["body_index"] != 34
        or e["bone_stable_id"] != 10
    ):
        raise ValueError("Wrong material endpoint")
    struct.pack_into("<I3f", source, off, 34, *p)
    delta = abs(np.linalg.norm(p - via) - np.linalg.norm(old - via))
    if delta > 1e-8 or proposal["cortical_gap_m"] > 1e-6:
        raise ValueError("Route length or cortical discrepancy")
    eidx = r["endpoint_index"]
    row = m["endpoints"][eidx]
    header = list(struct.unpack_from("<8s10I32s32s32s", t))
    ep = list(struct.unpack_from("<8I8f", t, 144 + 64 * eidx))
    if (
        header[0] != b"NHTEND3\0"
        or header[-2].hex() != r["muscle"]
        or ep[:6]
        != [row["muscle_index"], 1, r["route_index"], r["source_site_id"], 34, 0]
        or row["admission_reason"] != "surface_distance_exceeds_gate"
    ):
        raise ValueError("Wrong previous tendon binding")
    last = header[6]
    header[6] += 1
    struct.pack_into(
        "<8I8f",
        t,
        144 + 64 * eidx,
        *ep[:5],
        2,
        last,
        10,
        *p,
        e["surface_distance_m"],
        e["sampled_total_force_amplification"],
        e["patch_radius_m"],
        e["moment_residual_m"],
        0.0,
    )
    t.extend(pack_envelope(e, 34))
    out.mkdir(exist_ok=True)
    binary = out / mp.name
    binary.write_bytes(source)
    header[-2] = bytes.fromhex(digest(binary))
    struct.pack_into("<8s10I32s32s32s", t, 0, *header)
    target = out / tp.name
    target.write_bytes(t)
    row.update(
        prior_source_local_point_m=old.tolist(),
        source_local_point_m=p.tolist(),
        resolved_local_point_m=p.tolist(),
        endpoint_migration_m=0.0,
        attachment_mode="registered_bone_distributed_envelope",
        admission_reason="admitted",
        surface={
            k: v
            for k, v in e.items()
            if k not in ("force_maps", "resolved_local_point_m")
        },
    )
    m["payload"].update(sha256=digest(target), bytes=len(t))
    m["source"]["myosim_muscle_payload_sha256"] = digest(binary)
    cov = m["coverage"]
    cov["registered_bone_distributed_envelope_count"] = header[6] - cov.get(
        "source_component_mechanics_surface_enthesis_envelope_count", 0
    )
    cov["distributed_surface_envelope_count"] = header[6]
    cov["source_site_point_fallback_count"] = header[5] - header[6]
    cov["surface_coverage_fraction"] = header[6] / header[5]
    methods = {
        "connected_geodesic_compass_vertices": "compass_vertex_envelope_count",
        "connected_geodesic_topology_aware_exact_surface_points": "topology_aware_exact_surface_envelope_count",
    }
    cov[methods[e["surface_patch_method"]]] += 1
    from collections import Counter

    # Reconstruct from actual endpoint bindings. Older envelopes retained
    # historical rejection counts for two subsequently admitted attachments.
    reasons = Counter()
    for i, q in enumerate(m["endpoints"]):
        b = struct.unpack_from("<8I8f", t, 144 + 64 * i)
        if (
            b[0] != q["muscle_index"]
            or b[2] != q["route_node_index"]
            or b[3] != q["source_site_index"]
            or b[4] != q["body_index"]
        ):
            raise ValueError("Endpoint table disagrees with native owner")
        if (b[5] == 0) != (q["attachment_mode"] == "source_site_point"):
            raise ValueError("Native binding mode differs")
        if b[5] == 0:
            reasons[q["admission_reason"]] += 1
    if sum(reasons.values()) != cov["source_site_point_fallback_count"]:
        raise ValueError("Invalid NHTENDON coverage")
    m["admission"]["rejection_counts"] = dict(sorted(reasons.items()))
    packet = struct.unpack("<4I68f", t[-288:])
    nodes = np.array(packet[4:20]).reshape(4, 4)[:, :3]
    maps = np.array(packet[20:68]).reshape(4, 3, 4)[:, :, :3]
    # Exact induced 2-norms bound all possible terminal directions, including the
    # actual muscle force. The opposite tension acts at the unchanged via.
    skew = lambda u: np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    force_error = np.linalg.norm(maps.sum(0) - np.eye(3), 2)
    moment_error = np.linalg.norm(sum(skew(q - p) @ a for q, a in zip(nodes, maps)), 2)
    directed = (via - p) / np.linalg.norm(via - p)
    node_f = maps @ directed
    system_force = node_f.sum(0) - directed
    system_moment = np.cross(nodes - via, node_f).sum(0)
    if (
        force_error > 1e-5
        or moment_error > 1e-6
        or np.linalg.norm(system_moment) > 1e-6
    ):
        raise ValueError("Loss of internal wrench conservation")
    report = dict(
        parent_muscle=r["muscle"],
        parent_tendon=r["tendon"],
        source_bone=r["bone"],
        muscle=digest(binary),
        tendon=digest(target),
        source_launch=digest(case / "launch.json"),
        candidate=chosen,
        atlas=r["source_shape"],
        body=34,
        cortical_bank=10,
        source_site=r["source_site_id"],
        via_site=r["via_site_id"],
        program_endpoint=row["endpoint"],
        anatomical_role="scapular origin of right infraspinatus",
        original_cortex_distance_m=r["original_cortical_gap_m"],
        new_cortex_distance_m=proposal["cortical_gap_m"],
        displacement_from_original_m=float(np.linalg.norm(p - old)),
        unchanged_via_m=via.tolist(),
        actual_segment_length_change_m=float(delta),
        all_direction_force_operator_error=force_error,
        all_direction_moment_error_m=moment_error,
        same_body_force_error_per_N=np.linalg.norm(system_force),
        same_body_moment_error_m=np.linalg.norm(system_moment),
        old_envelopes=last,
        new_envelopes=header[6],
        facing_report=digest(fit / "facing.json"),
        footprint=e["surface_patch_method"],
    )
    m["source"]["scapular_infraspinatus_registration"] = report
    (out / tm.name).write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
    (out / "registration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        "relocation mm",
        report["displacement_from_original_m"] * 1000,
        "length change nm",
        delta * 1e9,
        "unit moment error",
        moment_error,
    )


if __name__ == "__main__":
    build(*(Path(p).resolve() for p in sys.argv[1:]))
