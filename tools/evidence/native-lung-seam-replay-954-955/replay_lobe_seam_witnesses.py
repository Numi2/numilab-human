#!/usr/bin/env python3
"""Exact selected lobe/pleura witness replay for one hash-pinned accepted capture.

This is an offline diagnostic. It does not change geometry or acceptance predicates.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import struct
import sys
from fractions import Fraction

import numpy as np

SCHEMA = "numi.human.selected-lobe-seam-witness-case.v1"
ALLOWED_LOBES = {305, 306, 307, 308, 309}
PLEURA = 310
PACK_LOBE_SEMANTIC = 51023
PACK_PLEURA_SEMANTIC = 51024


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pin_file(entry, name):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
        raise ValueError(f"input pin {name} must contain path and sha256")
    path = pathlib.Path(entry["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"pinned {name} is missing: {path}")
    actual = sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(f"{name} SHA-256 mismatch: expected {entry['sha256']}, got {actual}")
    return path, actual


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoothstep(t):
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def respiratory_smooth_field(p, cfg):
    rim = cfg["footprint_rim_ellipse_m"]
    rim_transition = cfg["footprint_rim_transition"]
    crural = cfg["footprint_crural_ellipses_m"]
    crural_transition = cfg["footprint_crural_transition"]

    def ellipse(e, transition, inside):
        x = (p[0] - e[0]) / e[2]
        z = (p[2] - e[1]) / e[3]
        radius = math.sqrt(x * x + z * z)
        t = min(max((radius - transition[0]) / transition[1], 0.0), 1.0)
        value = smoothstep(t)
        derivative = 6.0 * t * (1.0 - t) / transition[1] * (-1.0 if inside else 1.0)
        gx = derivative * x / (e[2] * radius) if radius else 0.0
        gz = derivative * z / (e[3] * radius) if radius else 0.0
        weight = 1.0 - value if inside else value
        return gx, gz, weight

    r = ellipse(rim, rim_transition, True)
    a = ellipse(crural[0], crural_transition, False)
    b = ellipse(crural[1], crural_transition, False)
    w = r[2] * a[2] * b[2]
    wx = r[0] * a[2] * b[2] + r[2] * a[0] * b[2] + r[2] * a[2] * b[0]
    wz = r[1] * a[2] * b[2] + r[2] * a[1] * b[2] + r[2] * a[2] * b[1]
    t = min(max((p[1] - cfg["basal_blend_start_m"]) / cfg["basal_blend_span_m"], 0.0), 1.0)
    g = 1.0 - smoothstep(t)
    dg = -6.0 * t * (1.0 - t) / cfg["basal_blend_span_m"]
    return g * wx, g * w * dg, g * wz, g * w


def respiratory_basis(p, cfg):
    h = float(cfg["conforming_grid_spacing_m"])
    u = [float(x) / h for x in p]
    low_index = [math.floor(x) for x in u]
    low = [x * h for x in low_index]
    fractions = [u[k] - low_index[k] for k in range(3)]
    order = sorted(range(3), key=lambda k: (-fractions[k], k))
    corner = list(low)
    previous = respiratory_smooth_field(corner, cfg)[3]
    out = [0.0, 0.0, 0.0, previous]
    for axis in order:
        corner[axis] += h
        value = respiratory_smooth_field(corner, cfg)[3]
        out[axis] = (value - previous) / h
        out[3] += out[axis] * (float(p[axis]) - low[axis])
        previous = value
    return out


def rotate_quaternion_xyzw(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + float(q[3]) * v)


def map_source_point64(point, cfg, axis, anchor, volume, swept_area, motion, body_pose):
    p = np.asarray(point, dtype=np.float64)
    weight = respiratory_basis(p, cfg)[3]
    # Preserve the established 952 evaluator's NumPy scalar promotion from
    # uploaded Float32 motion/area/volume fields; do not silently make this
    # replay a different all-binary64 respiratory parameter path.
    diaphragm_delta = motion[0] / swept_area
    volume_after = volume + motion[0]
    radial_scale = math.sqrt((volume_after + motion[1]) / volume_after)
    offset = p - anchor
    along = np.dot(offset, axis) * axis
    across = offset - along
    deformed = p - axis * (diaphragm_delta * weight) + (radial_scale - 1.0) * across
    return np.asarray(body_pose["position_m"], dtype=np.float64) + rotate_quaternion_xyzw(body_pose["quaternion_xyzw"], deformed)


def tri32(triangle, point_key):
    return tuple(point_key(tuple(float(x) for x in p)) for p in triangle)


def tri64(triangle):
    return tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in triangle)


def relation(a, b, predicate, lattice=False):
    points = list(dict.fromkeys(predicate(a, b)))
    span = 0.0
    if len(points) > 1:
        if lattice:
            coords = np.ldexp(np.asarray(points, dtype=np.float64), -149)
        else:
            coords = np.asarray([[float(x) for x in p] for p in points], dtype=np.float64)
        span = max(float(np.linalg.norm(coords[i] - coords[j]))
                   for i in range(len(coords)) for j in range(i + 1, len(coords)))
    count = len(points)
    classification = "disjoint" if count == 0 else ("single_point_contact" if count == 1 else "multi_point_intersection")
    return {"hit": bool(count), "points": count, "span_m_diagnostic": span, "classification": classification}


def point_triangle_distance(p, tri):
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return float(np.linalg.norm(p - (a + (d1 / (d1 - d3)) * ab)))
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return float(np.linalg.norm(p - (a + (d2 / (d2 - d6)) * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 >= d3 and d5 >= d6:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + t * (c - b))))
    denom = va + vb + vc
    if denom == 0.0:
        return min(float(np.linalg.norm(p - q)) for q in (a, b, c))
    v = vb / denom
    w = vc / denom
    return float(np.linalg.norm(p - (a + ab * v + ac * w)))


def segment_distance(p1, q1, p2, q2):
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    tiny = 1e-300
    if a <= tiny and e <= tiny:
        return float(np.linalg.norm(p1 - p2))
    if a <= tiny:
        s = 0.0
        t = min(1.0, max(0.0, f / e))
    else:
        c = float(np.dot(d1, r))
        if e <= tiny:
            t = 0.0
            s = min(1.0, max(0.0, -c / a))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > tiny * a * e else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def triangle_distance_diagnostic(a, b):
    distance = min(point_triangle_distance(p, b) for p in a)
    distance = min(distance, min(point_triangle_distance(p, a) for p in b))
    for i in range(3):
        for j in range(3):
            distance = min(distance, segment_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]))
    return distance


def upward_float(q):
    """Round a nonnegative exact Fraction upward to the next binary64 value."""
    if q < 0:
        raise ValueError("upward_float expects a nonnegative value")
    value = float(q)
    if not math.isfinite(value):
        raise ValueError("nonfinite envelope")
    if Fraction.from_float(value) < q:
        value = math.nextafter(value, math.inf)
    return value


def upward_l1(captured, replay):
    exact = sum(abs(Fraction.from_float(float(c)) - Fraction.from_float(float(r)))
                for c, r in zip(captured, replay))
    return upward_float(exact)


def upward_sum(a, b):
    return upward_float(Fraction.from_float(float(a)) + Fraction.from_float(float(b)))


def point_key_bits(point):
    return tuple(int(x) for x in np.asarray(point, dtype=np.float32).view(np.uint32))


def points_multiset_key(triangle):
    return sorted(point_key_bits(p) for p in triangle)


def pack_triangle(mapping, vertex_offset, faces, face_id):
    ids = faces[int(face_id)]
    coords = [struct.unpack_from("<3f", mapping, vertex_offset + int(vertex_id) * 80) for vertex_id in ids]
    return np.asarray(coords, dtype=np.float32), tuple(int(x) for x in ids)


def event_key(kind, owners, faces, pleura_faces=None):
    return (kind, tuple(int(x) for x in owners), tuple(int(x) for x in faces),
            None if pleura_faces is None else tuple(int(x) for x in pleura_faces))


def verify_case(case):
    if case.get("schema") != SCHEMA:
        raise ValueError("unsupported witness case schema")
    inputs = case.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("case inputs are missing")
    verified = {}
    for name in ("source_nha", "manifest", "anatomy_parameters", "native_pack",
                 "accepted_receipt", "full_scan_report", "classification_report",
                 "row310_face_lineage", "anatomy_parser", "exact_predicate", "pack_reader"):
        verified[name] = pin_file(inputs.get(name), name)

    source_path, source_sha = verified["source_nha"]
    manifest_path, _ = verified["manifest"]
    params_path, params_sha = verified["anatomy_parameters"]
    pack_path, pack_sha = verified["native_pack"]
    receipt_path, receipt_sha = verified["accepted_receipt"]
    scan_path, scan_sha = verified["full_scan_report"]
    class_path, class_sha = verified["classification_report"]
    lineage_path, lineage_sha = verified["row310_face_lineage"]
    parser_path, _ = verified["anatomy_parser"]
    predicate_path, _ = verified["exact_predicate"]
    reader_path, _ = verified["pack_reader"]

    scan = json.loads(scan_path.read_text())
    classification = json.loads(class_path.read_text())
    if scan.get("schema") != "numi.human.native-terminal-lung-full-row-exact-census.v1":
        raise ValueError("selection source is not the recognized full-row exact census")
    if scan.get("status") != "completed_exact_full_row_scan":
        raise ValueError("selection source full scan is not complete")
    coverage = scan.get("coverage", {})
    receipt = json.loads(receipt_path.read_text())
    step = int(case.get("accepted_step", -1))
    if step < 0 or int(receipt.get("accepted_step", -2)) != step:
        raise ValueError("case accepted step differs from the accepted receipt")
    if coverage.get("all_faces_considered") is not True:
        raise ValueError("full scan does not attest all faces considered")
    if coverage.get("source_nha_sha256") != source_sha:
        raise ValueError("full scan source NHA hash differs from case payload")
    if coverage.get("pack_sha256") != pack_sha or coverage.get("receipt_sha256") != receipt_sha:
        raise ValueError("full scan pack/receipt identity differs from case")
    if int(coverage.get("step", -1)) != step:
        raise ValueError("full scan accepted step differs from case")
    if scan.get("capture", {}).get("receipt_validation", {}).get("status") not in ("accepted", "passed", "valid"):
        # The accepted receipt validator below is authoritative if an older report uses another spelling.
        if receipt.get("surface_audit_endpoint") != "passed":
            raise ValueError("full scan/capture receipt does not attest an accepted geometry")
    if classification.get("schema") != "numi.human.true-terminal-lung-unallowed-event-source-review.v1":
        raise ValueError("classification source is not the recognized exact event report")
    if classification.get("status") != "complete_source_and_native_event_diagnosis":
        raise ValueError("classification source is incomplete")
    cls_inputs = classification.get("inputs", {})

    def classification_input_for(expected_path, expected_sha, label):
        matches = []
        for record in cls_inputs.values():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            try:
                record_path = pathlib.Path(record["path"]).resolve()
            except (OSError, RuntimeError):
                continue
            if record_path == expected_path and record.get("sha256") == expected_sha:
                matches.append(record)
        if len(matches) != 1:
            raise ValueError(f"classification must bind exactly one {label} by path and SHA-256; found {len(matches)}")
        return matches[0]

    classification_input_for(source_path, source_sha, "source NHA")
    classification_input_for(scan_path, scan_sha, "full scan")
    classification_input_for(lineage_path, lineage_sha, "row-310 lineage")
    param_receipt = receipt.get("skin_source_mapping", {}).get("anatomy_parameters", {})
    if pathlib.Path(param_receipt.get("path", "")).resolve() != params_path or param_receipt.get("sha256") != params_sha:
        raise ValueError("anatomy parameter blob is not the one bound by the accepted receipt")
    if receipt.get("pack_file_sha256") != pack_sha:
        raise ValueError("accepted receipt does not bind the supplied pack")
    if pathlib.Path(receipt.get("accepted_pack_path", "")).resolve() != pack_path:
        raise ValueError("accepted receipt pack path differs from supplied capture")
    if scan.get("source", {}).get("native_row_mapping") is None:
        raise ValueError("full scan lacks native row mapping and face-order evidence")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "numi.human.resting-anatomy-manifest.v1":
        raise ValueError("manifest has an unexpected schema")
    payload_record = manifest.get("payload", {})
    if payload_record.get("sha256") != source_sha:
        raise ValueError("manifest payload SHA differs from source NHA")
    respiratory_cfg = manifest.get("functional_bindings", {}).get("respiratory_geometry_binding")
    if not isinstance(respiratory_cfg, dict):
        raise ValueError("manifest lacks the respiratory geometry map")

    sys.path.insert(0, str(parser_path.parents[1]))
    from numilab_human.resting_anatomy_interface_patch import parse_payload
    parser_module_file = parser_path
    if pathlib.Path(parse_payload.__code__.co_filename).resolve() != parser_module_file.resolve():
        raise ValueError("loaded NHA parser is not the pinned parser")
    import importlib
    predicate_module = importlib.import_module("numilab_human.cardiac_cavity_intersections")
    if pathlib.Path(predicate_module.__file__).resolve() != predicate_path.resolve():
        raise ValueError("loaded exact predicate is not the pinned predicate module")
    reader = load_module(reader_path, "pinned_accepted_pack_reader")

    header, rows = parse_payload(source_path)
    if not isinstance(rows, dict):
        raise ValueError("NHA parser did not return a stable-id row map")
    owners_needed = set()
    witnesses = case.get("witnesses")
    if not isinstance(witnesses, list) or not witnesses:
        raise ValueError("case has no explicit witnesses")
    for witness in witnesses:
        owners = witness.get("owners")
        faces = witness.get("face_ids")
        if not isinstance(owners, list) or len(owners) != 2 or not isinstance(faces, list) or len(faces) != 2:
            raise ValueError("each witness needs two owners and two face ids")
        owners = tuple(int(x) for x in owners)
        if owners[0] == owners[1] or any(x not in ALLOWED_LOBES for x in owners):
            raise ValueError("witness owner scope is limited to distinct lobe rows 305-309")
        owners_needed.update(owners)
        if witness.get("kind") == "pleura_parent_pair":
            pf = witness.get("pleura_face_ids")
            if not isinstance(pf, list) or len(pf) != 2:
                raise ValueError("pleura-parent witness requires two explicit row-310 face ids")
            owners_needed.add(PLEURA)
        elif witness.get("kind") != "lobe_pair":
            raise ValueError("witness kind must be lobe_pair or pleura_parent_pair")

    for owner in owners_needed:
        if owner not in rows:
            raise ValueError(f"NHA lacks selected stable row {owner}")
        if int(rows[owner]["body_index"]) != 20:
            raise ValueError(f"selected row {owner} is not bound to body 20")
        native_meta = scan["source"]["native_row_mapping"].get(str(owner))
        expected_semantic = PACK_PLEURA_SEMANTIC if owner == PLEURA else PACK_LOBE_SEMANTIC
        if (not isinstance(native_meta, dict) or native_meta.get("source_face_order_identical") is not True
                or int(native_meta.get("face_count", -1)) != len(rows[owner]["faces"])
                or int(native_meta.get("link", -1)) != 20
                or int(native_meta.get("stable_id", -1)) != owner
                or int(native_meta.get("semantic", -1)) != expected_semantic):
            raise ValueError(f"full scan does not verify face order/identity for row {owner}")
        scan_counts = coverage.get("self_face_counts", {})
        if str(owner) in scan_counts and int(scan_counts[str(owner)]) != len(rows[owner]["faces"]):
            raise ValueError(f"full scan face count differs for row {owner}")

    lineage = np.load(lineage_path, allow_pickle=False)
    if lineage.ndim != 2 or lineage.shape[1] != 2 or lineage.shape[0] != len(rows[PLEURA]["faces"]):
        raise ValueError("row-310 lineage shape does not match the NHA face order")
    if not np.array_equal(lineage.astype(np.int64), lineage):
        raise ValueError("row-310 lineage contains nonintegral indices")

    classified_events = {}
    for event in classification.get("pleura_self", {}).get("events", []):
        key = event_key("pleura_parent_pair", event.get("parent_owner_pair", []),
                        event.get("parent_face_ids", []), event.get("pleura_face_ids", []))
        classified_events[key] = event
    for event in classification.get("cross", {}).get("events", []):
        key = event_key("lobe_pair", event.get("owners", []), event.get("face_ids", []))
        classified_events[key] = event

    unique = set()
    for witness in witnesses:
        kind = witness["kind"]
        owners = tuple(int(x) for x in witness["owners"])
        faces = tuple(int(x) for x in witness["face_ids"])
        pleura_faces = tuple(int(x) for x in witness["pleura_face_ids"]) if kind == "pleura_parent_pair" else None
        pair_key = "-".join(str(x) for x in sorted(owners))
        pair_counts = coverage.get("cross_face_counts", {}).get(pair_key)
        expected_pair_counts = [len(rows[x]["faces"]) for x in sorted(owners)]
        if pair_counts is None or [int(x) for x in pair_counts] != expected_pair_counts:
            raise ValueError(f"full scan does not attest selected cross-row pair and face counts {pair_key}")
        key = event_key(kind, owners, faces, pleura_faces)
        if key in unique:
            raise ValueError(f"duplicate explicit witness: {key}")
        unique.add(key)
        if key not in classified_events:
            raise ValueError(f"explicit witness is absent from the hash-pinned full-scan classification: {key}")
        for owner, face in zip(owners, faces):
            if face < 0 or face >= len(rows[owner]["faces"]):
                raise ValueError(f"selected face {face} is out of range for row {owner}")
        if pleura_faces is not None:
            for side, (owner, face, pleura_face) in enumerate(zip(owners, faces, pleura_faces)):
                if pleura_face < 0 or pleura_face >= len(rows[PLEURA]["faces"]):
                    raise ValueError("selected row-310 face is out of range")
                if tuple(int(x) for x in lineage[pleura_face]) != (owner, face):
                    raise ValueError(f"row-310 face {pleura_face} lineage does not equal selected parent {(owner, face)}")

    params = params_path.read_bytes()
    if len(params) != 1008:
        raise ValueError(f"unexpected native anatomy parameter size: {len(params)}")
    words = struct.unpack_from("<4I", params, 0)
    body_index = int(words[0])
    if body_index != 20:
        raise ValueError("native anatomy parameters select a different body index")
    anchor = np.asarray(struct.unpack_from("<3f", params, 16), dtype=np.float64)
    volume, = struct.unpack_from("<f", params, 28)
    axis = np.asarray(struct.unpack_from("<3f", params, 32), dtype=np.float64)
    swept_area, = struct.unpack_from("<f", params, 104)
    if not np.isfinite(anchor).all() or not np.isfinite(axis).all() or not math.isfinite(volume) or not math.isfinite(swept_area) or swept_area <= 0.0:
        raise ValueError("native anatomy parameters contain invalid respiratory values")
    body_poses = [x for x in receipt.get("accepted_registered_body_poses", []) if int(x.get("body_index", -1)) == body_index]
    if len(body_poses) != 1:
        raise ValueError("accepted receipt does not contain exactly one selected body pose")
    motion_record = receipt.get("accepted_respiratory_motion", {})
    motion = np.asarray([motion_record["diaphragm_swept_volume_m3"], motion_record["rib_swept_volume_m3"]], dtype=np.float32)

    mapping, stream, vertex_offset, surfaces = reader.read_pack(pack_path)
    try:
        reader.validate_accepted_receipt(pack_path, receipt_path, step, mapping, vertex_offset, surfaces)
        pack_keys = {(int(k[0]), int(k[1])) for k in surfaces}
        for owner in owners_needed:
            semantic = PACK_PLEURA_SEMANTIC if owner == PLEURA else PACK_LOBE_SEMANTIC
            key = (semantic, owner)
            if key not in pack_keys:
                raise ValueError(f"accepted pack lacks surface {key}")
            surface = surfaces[key]
            if int(surface.get("instance", -1)) != owner or int(surface.get("link", -1)) != body_index:
                raise ValueError(f"accepted pack surface {key} has wrong instance/link identity")
            if len(surface["faces"]) != len(rows[owner]["faces"]):
                raise ValueError(f"accepted pack face count/order extent differs for row {owner}")

        results = []
        for witness in witnesses:
            kind = witness["kind"]
            owners = tuple(int(x) for x in witness["owners"])
            face_ids = tuple(int(x) for x in witness["face_ids"])
            source_triangles = []
            ideal_triangles = []
            native_triangles = []
            native_vertex_ids = []
            face_rows = []
            for owner, face_id in zip(owners, face_ids):
                row = rows[owner]
                vertex_ids = np.asarray(row["faces"][face_id], dtype=np.int64)
                source_triangle = np.asarray(row["vertices6"][vertex_ids, :3], dtype=np.float32)
                ideal_triangle = np.asarray([map_source_point64(p, respiratory_cfg, axis, anchor, volume,
                                                                 swept_area, motion, body_poses[0])
                                             for p in source_triangle], dtype=np.float64)
                surface = surfaces[(PACK_LOBE_SEMANTIC, owner)]
                native_triangle, native_ids = pack_triangle(mapping, vertex_offset, surface["faces"], face_id)
                source_triangles.append(source_triangle)
                ideal_triangles.append(ideal_triangle)
                native_triangles.append(native_triangle)
                native_vertex_ids.append(native_ids)
                face_rows.append({
                    "owner": owner,
                    "face_id": face_id,
                    "nha_face_order_sha256": hashlib.sha256(np.asarray(row["faces"], dtype="<u4").tobytes()).hexdigest(),
                    "nha_vertex_ids_in_face_order": [int(x) for x in vertex_ids],
                    "native_global_vertex_ids_in_face_order": list(native_ids),
                    "source_body_index": int(row["body_index"]),
                })

            source_relation = relation(tri32(source_triangles[0], predicate_module.float32_point_lattice_key),
                                      tri32(source_triangles[1], predicate_module.float32_point_lattice_key),
                                      predicate_module.triangle_intersection_points, lattice=True)
            ideal_relation = relation(tri64(ideal_triangles[0]), tri64(ideal_triangles[1]),
                                      predicate_module.triangle_intersection_points)
            native_relation = relation(tri32(native_triangles[0], predicate_module.float32_point_lattice_key),
                                       tri32(native_triangles[1], predicate_module.float32_point_lattice_key),
                                       predicate_module.triangle_intersection_points, lattice=True)

            face_envelopes = []
            vertex_records = []
            for owner, captured, ideal in zip(owners, native_triangles, ideal_triangles):
                vertex_l1 = [upward_l1(captured[i], ideal[i]) for i in range(3)]
                vertex_euclidean = [float(np.linalg.norm(captured[i].astype(np.float64) - ideal[i])) for i in range(3)]
                face_envelopes.append(max(vertex_l1))
                for i in range(3):
                    vertex_records.append({
                        "owner": owner,
                        "face_corner": i,
                        "captured_world_f32_m": [float(x) for x in captured[i]],
                        "binary64_replay_world_m": [float(x) for x in ideal[i]],
                        "capture_to_replay_l1_upper_m": vertex_l1[i],
                        "capture_to_replay_euclidean_diagnostic_m": vertex_euclidean[i],
                    })
            pair_l1_upper = upward_sum(face_envelopes[0], face_envelopes[1])
            gap_diag = 0.0 if ideal_relation["hit"] else triangle_distance_diagnostic(ideal_triangles[0], ideal_triangles[1])
            parent_match = None
            if kind == "pleura_parent_pair":
                pfids = tuple(int(x) for x in witness["pleura_face_ids"])
                pleura_surface = surfaces[(PACK_PLEURA_SEMANTIC, PLEURA)]
                matches = []
                for owner, face_id, pleura_face_id, parent_tri in zip(owners, face_ids, pfids, native_triangles):
                    pleura_tri, _ = pack_triangle(mapping, vertex_offset, pleura_surface["faces"], pleura_face_id)
                    exact_match = points_multiset_key(pleura_tri) == points_multiset_key(parent_tri)
                    if not exact_match:
                        raise ValueError(f"row-310 native face {pleura_face_id} does not match lineage parent {(owner, face_id)}")
                    matches.append({"pleura_face_id": pleura_face_id, "owner": owner, "parent_face_id": face_id,
                                    "native_parent_coordinates_exact_match": True})
                parent_match = matches
            results.append({
                "kind": kind,
                "owners": list(owners),
                "face_ids": list(face_ids),
                "pleura_face_ids": list(witness["pleura_face_ids"]) if kind == "pleura_parent_pair" else None,
                "faces": face_rows,
                "source_relation": source_relation,
                "binary64_map_relation": ideal_relation,
                "captured_native_relation": native_relation,
                "ideal_pair_distance_diagnostic_m": gap_diag,
                "capture_to_replay_per_face_l1_upper_m": face_envelopes,
                "pairwise_capture_to_replay_l1_upper_m": pair_l1_upper,
                "ideal_gap_within_l1_envelope_diagnostic": gap_diag <= pair_l1_upper,
                "pleura_parent_coordinate_matches": parent_match,
                "vertices": vertex_records,
                "captured_exact_hit_is_not_waived": bool(native_relation["hit"]),
            })

        summary = {
            "witness_count": len(results),
            "source_relation_counts": {},
            "binary64_map_relation_counts": {},
            "captured_native_relation_counts": {},
            "captured_native_exact_hit_count": sum(bool(x["captured_native_relation"]["hit"]) for x in results),
            "captured_native_exact_hits_remain_unallowed": True,
            "all_binary64_disjoint_pair_gaps_within_measured_l1_envelope": all(
                x["ideal_gap_within_l1_envelope_diagnostic"] for x in results
            ),
            "max_pairwise_capture_to_replay_l1_upper_m": max(x["pairwise_capture_to_replay_l1_upper_m"] for x in results),
            "max_pairwise_capture_to_replay_euclidean_diagnostic_m": max(
                sum(max(float(v["capture_to_replay_euclidean_diagnostic_m"]) for v in x["vertices"] if int(v["owner"]) == owner)
                    for owner in x["owners"]) for x in results
            ),
        }
        for label in ("source_relation", "binary64_map_relation", "captured_native_relation"):
            counts = {}
            for event in results:
                relation_item = event[label]
                key = relation_item["classification"]
                counts[key] = counts.get(key, 0) + 1
            summary[label.replace("_relation", "_relation_counts")] = counts

        report = {
            "schema": "numi.human.selected-lobe-seam-capture-replay.v1",
            "status": "complete_selected_witness_replay",
            "scope": "Only the explicit full-scan witnesses in the case file were evaluated; no new pair scan was performed.",
            "case_sha256": None,
            "inputs": {name: {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
                       for name, (path, digest) in verified.items()},
            "runtime": {
                "accepted_step": step,
                "accepted_time_s": receipt.get("accepted_time_s"),
                "body_index": body_index,
                "body_pose": body_poses[0],
                "respiratory_motion_f32": motion.tolist(),
                "anatomy_parameter_header_words": list(words),
                "source_row_face_count": {str(owner): len(rows[owner]["faces"]) for owner in sorted(owners_needed)},
                "face_order_authority": "Hash-pinned NHA bytes plus full-scan native_row_mapping.source_face_order_identical and exact matching face counts; row-310 parent faces additionally checked by pinned per-face lineage and exact captured coordinate multisets.",
            },
            "method": {
                "binary64_map": "Selected source triangles are replayed through the respiratory geometry binding in the pinned NHA manifest using the accepted receipt body pose and respiratory motion plus the pinned native anatomy-parameter blob. Exact predicates classify the resulting binary64 coordinates; this is a replay, not an exact-real shader proof.",
                "l1_envelope": "For each captured/replay vertex, the L1 coordinate difference is summed as exact Fractions of the stored binary32/binary64 values and rounded upward to binary64. Convex barycentric interpolation cannot exceed the maximum vertex L1 displacement for a triangle; summing the two face bounds bounds corresponding-point separation. Euclidean norms are retained as diagnostics only.",
                "distance": "Disjoint binary64 triangle gaps are calculated by floating point closest-feature routines and are diagnostic; they are not directed-rounded exact distance proofs.",
                "limits": ["The explicit witness set is selected input, not a full-scene scan.", "The envelope covers stored capture coordinates versus this replay only; it is not a formal GPU arithmetic or continuous-time error bound.", "Exact native intersections remain unwaived and unallowed."],
            },
            "summary": summary,
            "witnesses": results,
        }
        return report
    finally:
        stream.close()
        mapping.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    case_path = args.case.resolve()
    out_path = args.out.resolve()
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite output: {out_path}")
    try:
        case = json.loads(case_path.read_text())
        report = verify_case(case)
        report["case_sha256"] = sha256(case_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps(report["summary"], sort_keys=True, indent=2))
        print("report_sha256", sha256(out_path))
    except Exception as exc:
        print(f"replay rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
