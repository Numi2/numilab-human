#!/usr/bin/env python3
"""One entrypoint for pinned native captures and existing regional NPZ candidates.

Exact geometry predicates, seam quotient, containment, units and typed identities
belong exclusively to numilab_human.resting_anatomy_verify. Candidate overlays are
explicitly counterfactual and cannot establish native or anatomical admission.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numilab_human import resting_anatomy_verify as verify


def pin(path, digest):
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source pin must be a lowercase SHA256")
    path = Path(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"source SHA256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual}



def reconstruction_references(path, digest, stable_id, table):
    """Check declared existing repair ancestry before preparing geometry.

    Raw compiler fixtures may omit the optional sidecar. When a sidecar is
    present, its payload and selected row's declared repair inputs must match.
    Results share the immutable source-table lifetime, just like source bytes.
    """
    cache = table.setdefault("_verified_reconstruction_references", {})
    if stable_id in cache:
        return cache[stable_id]
    manifest_path = Path(path).with_suffix(".manifest.json")
    if not manifest_path.exists():
        cache[stable_id] = None
        return None
    manifest_bytes = manifest_path.read_bytes()
    document = json.loads(manifest_bytes)
    if document["payload"]["sha256"] != digest:
        raise ValueError("source manifest payload SHA256 differs")
    rows = [row for row in document["source"]["surfaces"]
            if row["stable_id"] == stable_id]
    if len(rows) != 1:
        raise ValueError("source manifest requires one selected typed surface")
    repair = rows[0].get("reference_surface_reconstruction")
    references = {}
    if repair is not None:
        if repair["stable_id"] != stable_id:
            raise ValueError("source reconstruction stable identity differs")
        for name in ("candidate", "report"):
            references[name] = pin(repair[name], repair[name+"_sha256"])
    proof = {"manifest": pin(manifest_path, hashlib.sha256(manifest_bytes).hexdigest()),
             "declared_reconstruction_inputs": references,
             "source_ancestry_admission": False}
    cache[stable_id] = proof
    return proof


def identity(text):
    return verify.typed_key(tuple(int(x) for x in text.split(":")))


def source_field_preflight(arrays):
    """Check existing NPZ source lanes before any exact geometry predicate.

    These are preparation checks. Native owner emission and source ancestry
    still require their separately pinned compiler and capture proofs.
    """
    evidence = {}
    source_positions = arrays["source_positions"] if "source_positions" in arrays else None
    source_normals = arrays["source_normals"] if "source_normals" in arrays else None
    vertices6 = arrays["vertices6"] if "vertices6" in arrays else None
    source_faces = arrays["source_faces"] if "source_faces" in arrays else None
    faces = arrays["faces"] if "faces" in arrays else None
    if vertices6 is not None:
        if (vertices6.dtype != np.dtype("<f4") or vertices6.ndim != 2
                or vertices6.shape[1] != 6 or not np.isfinite(vertices6).all()):
            raise ValueError("source vertices6 must contain finite exact Float32 positions/normals")
        for label, declared, observed in (
                ("positions", source_positions, vertices6[:, :3]),
                ("normals", source_normals, vertices6[:, 3:])):
            if declared is not None:
                if declared.dtype != np.dtype("<f4"):
                    raise ValueError(f"duplicate source {label} must be exact Float32")
                if not np.array_equal(declared, observed):
                    raise ValueError(f"duplicate source {label} disagree with vertices6")
        source_positions, source_normals = vertices6[:, :3], vertices6[:, 3:]
    if source_faces is not None and faces is not None and not np.array_equal(source_faces, faces):
        raise ValueError("duplicate source face arrays disagree")
    source_faces = source_faces if source_faces is not None else faces
    if source_positions is not None or source_normals is not None:
        if (source_positions is None or source_normals is None or source_faces is None
                or source_positions.dtype != np.dtype("<f4")
                or source_normals.dtype != np.dtype("<f4")
                or source_positions.ndim != 2 or source_positions.shape[1] != 3
                or source_normals.shape != source_positions.shape
                or not np.isfinite(source_positions).all()
                or not np.isfinite(source_normals).all()):
            raise ValueError("source positions/normals require finite matching exact Float32 lanes and source faces")
        if (source_faces.ndim != 2 or source_faces.shape[1] != 3 or not len(source_faces)
                or source_faces.dtype.kind not in "iu" or source_faces.min() < 0
                or source_faces.max() >= len(source_positions)):
            raise ValueError("source face indices do not address declared source lanes")
        used = np.unique(source_faces)
        lengths = np.linalg.norm(source_normals[used].astype(np.float64), axis=1)
        if np.any(np.abs(lengths - 1.0) > 2e-3):
            raise ValueError("referenced source normals must be unit length")
        evidence.update(source_vertex_count=len(source_positions),
                        referenced_source_normal_count=len(used), source_normals_valid=True)
    if "vertex_records_raw" in arrays:
        raw = arrays["vertex_records_raw"]
        if (raw.dtype != np.dtype("uint8") or raw.ndim != 2 or raw.shape[1] != 56
                or vertices6 is None or len(raw) != len(vertices6)):
            raise ValueError("raw ABI5 vertex records require matching vertices6")
        emitted = np.ascontiguousarray(raw[:, :24]).view("<f4").reshape(-1, 6)
        if not np.array_equal(emitted, vertices6):
            raise ValueError("raw ABI5 vertex records disagree with source vertices6")
        weights = np.ascontiguousarray(raw[:, 40:56]).view("<f4").reshape(-1, 4)
        if (not np.isfinite(weights).all() or np.any(weights < 0)
                or np.any(np.abs(weights.sum(axis=1, dtype=np.float64) - 1) >= 1e-5)):
            raise ValueError("raw ABI5 vertex material weights must be finite, nonnegative and normalized")
        if "binding_records_raw" in arrays:
            indices = np.ascontiguousarray(raw[:, 24:40]).view("<u4").reshape(-1, 4)
            if np.any(indices[weights > 0] >= len(arrays["binding_records_raw"])):
                raise ValueError("raw ABI5 vertex material index exceeds source binding inventory")
        evidence["raw_vertex_records_match_source"] = True
    if "binding_records_raw" in arrays:
        raw = arrays["binding_records_raw"]
        if raw.dtype != np.dtype("uint8") or raw.ndim != 2 or raw.shape[1] != 36 or not len(raw):
            raise ValueError("raw ABI5 binding records have invalid byte layout")
        rest = np.ascontiguousarray(raw[:, 4:]).view("<f4").reshape(-1, 8)
        if (not np.isfinite(rest).all() or np.any(rest[:, 7] <= 0)
                or np.any(np.abs(np.sum(rest[:, 3:7].astype(np.float64) ** 2, axis=1) - 1) > 2e-5)):
            raise ValueError("raw ABI5 rigid rest transforms require finite positions, unit quaternions and positive scale")
        evidence["raw_binding_rest_transforms_valid"] = True
    if "capture_normals" not in arrays and source_normals is None:
        raise ValueError("candidate requires capture normals or validated source normals")
    return evidence


def source_owned_repacked_overlay(arrays, key, source, poses, cache):
    """Bind an explicitly repacked ABI5 row and replay its registered rest owner.

    This is offline counterfactual preparation. It does not establish native
    emission, source-face ancestry, accepted motion, or anatomical admission.
    """
    from numilab_human import passive_attachment_composition as pac

    if key[0] != 51005:
        raise ValueError("repacked source overlay requires a typed muscle identity")
    path, digest = source
    token = (str(Path(path).resolve()), digest)
    if token not in cache:
        cache[token] = (pin(path, digest), pac._read_nhtiss4(Path(path)))
    source_pin, table = cache[token]
    reconstruction = reconstruction_references(path, digest, key[1], table)
    rows = [r for r in table["records"] if int(r[6]) == key[1]]
    if len(rows) != 1:
        raise ValueError("repacked source has no unique typed muscle row")
    row = pac._nhtiss_row_arrays(table, rows[0])
    for name, field in (("source_positions", "positions"), ("source_normals", "normals"),
                        ("source_faces", "faces"), ("binding_indices", "binding_indices"),
                        ("weights", "weights")):
        if (name not in arrays or not np.array_equal(arrays[name], row[field])
                or (arrays[name].dtype.kind not in "iu" if field == "faces"
                    else arrays[name].dtype != row[field].dtype)):
            raise ValueError(f"repacked overlay differs from compiled source row: {name}")
    if (np.any(np.abs(row["weights"].sum(axis=1, dtype=float) - 1.) >= 1e-5)
            or np.any(row["binding_indices"][row["weights"] > 0] >= len(row["bindings"]))):
        raise ValueError("invalid compiled material weights or binding indices")
    if poses is None:
        raise ValueError("repacked source requires accepted rigid rest poses")
    checked = {}
    for core, pose in poses.items():
        pose = np.asarray(pose)
        if (type(core) is not int or core < 0 or pose.dtype != np.dtype("<f4")
                or pose.shape != (7,) or not np.isfinite(pose).all()
                or abs(float(np.linalg.norm(pose[3:].astype(float))) - 1.) > 2e-5):
            raise ValueError("invalid accepted rigid rest pose")
        checked[core] = pose
    for b in row["bindings"]:
        rest = np.asarray([*b["translation"], *b["quaternion"], b["scale"]], float)
        if (b["core"] not in checked or not np.isfinite(rest).all()
                or b["scale"] <= 0 or abs(float(np.linalg.norm(rest[3:7])) - 1.) > 2e-5):
            raise ValueError("invalid or missing compiled rigid rest transform")
    positions, faces, gids = (arrays[k] for k in (
        "capture_positions_m", "capture_faces", "capture_global_vertex_ids"))
    if (positions.shape != row["positions"].shape or not np.array_equal(faces, row["faces"])
            or gids.dtype.kind not in "iu"):
        raise ValueError("repacked overlay allocation or ordered source faces differ")
    _, binding = verify.retain_bound_source_faces(
        key, (positions, faces, None, gids), row["faces"], row["first_vertex"],
        np.arange(len(row["faces"]), dtype=np.int64))
    offset = binding["source_capture_vertex_namespace_offset"]
    expected = np.arange(row["vertex_count"], dtype=np.int64) + row["first_vertex"] + offset
    if not np.array_equal(gids, expected):
        raise ValueError("repacked overlay must preserve every ordered compiled source GID")
    emitted = pac._native_blended_rest_positions(row, row["positions"], checked)
    if emitted.dtype != positions.dtype or emitted.tobytes() != positions.tobytes():
        raise ValueError("repacked world coordinates differ from exact Float32 CPU rest-owner replay")
    return {"source": source_pin, "source_reconstruction_references": reconstruction,
            "ordered_compiled_source_binding": binding,
            "all_compiled_source_fields_exact": True, "cpu_rest_owner_emission_byte_exact": True,
            "coordinate_provenance": "counterfactual_owner_replay",
            "actual_native_emission_verified": False}


def apply_overlay(surfaces, key, path, digest, faces_key="capture_faces", *,
                  repacked_source=None, accepted_poses=None, source_cache=None):
    """Bind a fixed-allocation candidate or an explicitly source-owned repack."""
    key = verify.typed_key(key)
    source_pin = pin(path, digest)
    if key not in surfaces:
        raise ValueError(f"candidate typed identity absent from native capture: {key}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
        positions = arrays["capture_positions_m"].copy()
        faces = arrays[faces_key].copy()
        gids = arrays["capture_global_vertex_ids"].copy()
        normals = arrays["capture_normals"].copy() if "capture_normals" in arrays else None
        source_fields = source_field_preflight(arrays)
    if normals is not None and normals.dtype != np.dtype("<f4"):
        raise ValueError("capture normals must already be exact Float32")
    captured_gids_exact = np.array_equal(gids, surfaces[key][3]) and gids.dtype.kind in "iu"
    repacked_proof = None
    if repacked_source is not None:
        repacked_proof = source_owned_repacked_overlay(
            arrays, key, repacked_source, accepted_poses,
            source_cache if source_cache is not None else {})
    elif not captured_gids_exact:
        raise ValueError("candidate must preserve every ordered captured source GID")
    if positions.dtype != np.dtype("<f4"):
        raise ValueError("candidate world coordinates must already be exact Float32")
    prepared = verify.prepare_surface(key, positions, faces, normals=normals, source_vertex_ids=gids)
    surfaces[key] = prepared
    return {
        "typed_identity": list(key), "candidate": source_pin, "faces_key": faces_key,
        "all_ordered_capture_gids_exact": bool(captured_gids_exact),
        "repacked_source_owner": repacked_proof, "geometry_preparation": prepared.diagnostics(),
        "candidate_normals_present": normals is not None,
        "source_field_preflight": source_fields,
        "world_positions_are_counterfactual": True,
        "source_ancestry_and_rest_owner_validation": "separately pinned compiler evidence required",
    }


def rest_world_normals(world, faces):
    """Offline shading normals; unused slots have an inert unit placeholder."""
    tri = world[faces].astype(np.float64)
    crosses = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals = np.zeros_like(world, dtype=np.float64)
    for lane in range(3):
        np.add.at(normals, faces[:, lane], crosses)
    used = np.unique(faces)
    lengths = np.linalg.norm(normals[used], axis=1)
    if np.any(lengths == 0) or not np.isfinite(lengths).all():
        raise ValueError("rest-source has invalid referenced world normals")
    normals[used] /= lengths[:, None]
    unused = np.setdiff1d(np.arange(len(world)), used)
    normals[unused] = (1., 0., 0.)
    return normals.astype("<f4"), len(unused)


def apply_rest_source(surfaces, key, path, digest, accepted_poses, source_cache):
    """Prepare a pinned ABI5 row through its existing rigid rest owner.

    Offline geometry preparation only. This cannot reproduce parent-tissue
    deformation or establish native emission, source ancestry, or admission.
    """
    from numilab_human import passive_attachment_composition as pac

    key = verify.typed_key(key)
    if key[0] != 51005 or key not in surfaces:
        raise ValueError("rest-source requires a captured typed muscle identity")
    token = (str(Path(path).resolve()), digest)
    if token not in source_cache:
        source_cache[token] = (pin(path, digest), pac._read_nhtiss4(Path(path)))
    source_pin, table = source_cache[token]
    reconstruction_references(path, digest, key[1], table)
    records = [r for r in table["records"] if int(r[6]) == key[1]]
    if len(records) != 1 or int(records[0][7]) != 1:
        raise ValueError("rest-source requires one compiled muscle surface row")
    row = pac._nhtiss_row_arrays(table, records[0])
    if accepted_poses is None:
        raise ValueError("rest-source requires pinned accepted rigid poses")
    world = pac._native_blended_rest_positions(row, row["positions"], accepted_poses)
    faces = row["faces"].copy()
    normals, unused_normal_count = rest_world_normals(world, faces)
    arrays = dict(source_positions=row["positions"], source_normals=row["normals"],
                  source_faces=faces, binding_indices=row["binding_indices"],
                  weights=row["weights"], capture_positions_m=world,
                  capture_faces=faces, capture_normals=normals,
                  capture_global_vertex_ids=np.arange(row["vertex_count"], dtype="<u8")
                  + row["first_vertex"])
    fields = source_field_preflight(arrays)
    proof = source_owned_repacked_overlay(
        arrays, key, (path, digest), accepted_poses, source_cache)
    prepared = verify.prepare_surface(
        key, world, faces, normals=arrays["capture_normals"],
        source_vertex_ids=arrays["capture_global_vertex_ids"])
    surfaces[key] = prepared
    return {"typed_identity": list(key), "source": source_pin,
            "unreferenced_world_normal_placeholder_count": unused_normal_count,
            "unreferenced_world_normal_policy": "inert unit X; no emitted face references these slots",
            "repacked_source_owner": proof, "source_field_preflight": fields,
            "geometry_preparation": prepared.diagnostics(),
            "world_positions_are_counterfactual": True,
            "preparation": "compiled_rigid_rest_owner_replay_not_parent_tissue_deformation",
            "source_ancestry_and_rest_owner_validation":
                "separately pinned compiler evidence required"}


def shared_partition_declarations(raw):
    """Reject ambiguous pair declarations before loading capture or source bytes."""
    partitions = []
    seen = set()
    for first, second, source, digest in raw:
        key_a, key_b = identity(first), identity(second)
        if key_a == key_b:
            raise ValueError("shared partition requires two distinct typed identities")
        pair = tuple(sorted((key_a, key_b)))
        if pair in seen:
            raise ValueError("duplicate shared partition declaration")
        seen.add(pair)
        partitions.append((key_a, key_b, Path(source), digest))
    return partitions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--pack-sha256", required=True)
    p.add_argument("--capture-receipt", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--source-pin", action="append", default=[], help="PATH=SHA256; compiler/ancestry/owner evidence")
    p.add_argument("--overlay", action="append", default=[], help="SEM:ID=existing.npz=SHA256; counterfactual only")
    p.add_argument("--overlay-source", action="append", default=[],
                   help="SEM:ID=NHTISS=SHA256; explicit repack with byte-exact CPU rest-owner replay")
    p.add_argument("--rest-source", action="append", default=[],
                   help="SEM:ID=NHTISS=SHA256; offline rigid rest-owner replay, without NPZ; does not reproduce parent-tissue deformation")
    p.add_argument("--faces-key", action="append", default=[], help="SEM:ID=NPZ_KEY; default capture_faces")
    p.add_argument("--changed", action="append", required=True)
    p.add_argument("--target", action="append", help="SEM:ID; default all captured anatomy")
    p.add_argument("--required-closed", action="append", default=[])
    p.add_argument("--intended-enclosure", action="append", default=[], help="SEM:ID,SEM:ID; containment only")
    p.add_argument("--shared-partition", nargs=4, action="append", default=[],
                   metavar=("FIRST", "SECOND", "NHTISS", "SHA256"),
                   help="source-bound reciprocal passive muscle partition; strict exact certificate required")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("refusing to overwrite retained verification evidence")
    start = time.perf_counter()
    report = {"passed": False, "anatomical_admission": False, "native_admission": False,
              "counterfactual_candidates": bool(args.overlay or args.rest_source)}
    try:
        partitions = shared_partition_declarations(args.shared_partition)
        report["pack_pin"] = pin(args.pack, args.pack_sha256)
        report["receipt_pin"] = pin(args.capture_receipt, args.receipt_sha256)
        pins = []
        seen = set()
        for item in args.source_pin:
            path, digest = item.rsplit("=", 1)
            resolved = str(Path(path).resolve())
            if resolved in seen:
                raise ValueError("duplicate source pin")
            seen.add(resolved)
            pins.append(pin(path, digest))
        if (args.overlay or args.rest_source) and not pins:
            raise ValueError("counterfactual candidates require pinned compiler/ancestry evidence")
        report["source_pins"] = pins
        surfaces, report["capture_identity"] = verify.load_capture(args.pack, args.capture_receipt)
        faces_keys = {}
        for item in args.faces_key:
            text, name = item.split("=", 1)
            key = identity(text)
            if key in faces_keys:
                raise ValueError("duplicate candidate faces-key identity")
            faces_keys[key] = name
        source_overlays, source_cache = {}, {}
        accepted_poses = None
        for item in args.overlay_source:
            text, path, digest = item.split("=", 2)
            key = identity(text)
            if key in source_overlays:
                raise ValueError("duplicate repacked source overlay identity")
            source_overlays[key] = (Path(path), digest)
        if source_overlays or args.rest_source:
            raw = json.loads(args.capture_receipt.read_text())
            accepted_poses = {}
            for row in raw["accepted_body_poses"]:
                core = row["body_index"]
                if type(core) is not int or core in accepted_poses:
                    raise ValueError("duplicate or invalid accepted rigid rest identity")
                accepted_poses[core] = np.asarray(
                    [*row["position_m"], *row["quaternion_xyzw"]], dtype="<f4")
        overlay_keys, overlays = set(), []
        npz_overlay_keys = set()
        for item in args.overlay:
            text, path, digest = item.split("=", 2)
            key = identity(text)
            if key in overlay_keys:
                raise ValueError("duplicate candidate typed identity")
            overlay_keys.add(key)
            npz_overlay_keys.add(key)
            overlays.append(apply_overlay(
                surfaces, key, path, digest, faces_keys.get(key, "capture_faces"),
                repacked_source=source_overlays.get(key), accepted_poses=accepted_poses,
                source_cache=source_cache))
        for item in args.rest_source:
            text, path, digest = item.split("=", 2)
            key = identity(text)
            if key in overlay_keys:
                raise ValueError("duplicate candidate typed identity")
            overlay_keys.add(key)
            overlays.append(apply_rest_source(
                surfaces, key, Path(path), digest, accepted_poses, source_cache))
        if not set(source_overlays) <= npz_overlay_keys:
            raise ValueError("repacked source without matching overlay")
        if not set(faces_keys) <= npz_overlay_keys:
            raise ValueError("faces-key without matching overlay")
        report["overlays"] = overlays
        changed = [identity(x) for x in args.changed]
        targets = [identity(x) for x in args.target] if args.target else None
        closed = [identity(x) for x in args.required_closed]
        enclosures = [tuple(identity(x) for x in item.split(",")) for item in args.intended_enclosure]
        if any(len(pair) != 2 for pair in enclosures):
            raise ValueError("enclosure requires exactly two typed identities")
        for key_a, key_b, source, digest in partitions:
            pin(source, digest)
            if (key_a in overlay_keys) != (key_b in overlay_keys):
                raise ValueError("counterfactual shared partition requires both sides overlaid")
        if args.preflight_only:
            for key in changed:
                if key not in surfaces:
                    raise ValueError(f"changed identity absent: {key}")
            report.update(passed=True, status="PASS_PREPARATION_ONLY_NO_PAIR_PREDICATES")
        else:
            if partitions:
                from numilab_human.resting_shared_partition_verify import audit_with_shared_partitions
                result = audit_with_shared_partitions(
                    surfaces, changed, partitions=partitions, targets=targets,
                    required_closed=closed, intended_enclosures=enclosures,
                    coordinate_provenance=("counterfactual_owner_replay" if args.overlay or args.rest_source
                                           else "accepted_native_capture"))
            else:
                result = verify.audit_changed_surfaces(
                    surfaces, changed, targets=targets,
                    required_closed=closed, intended_enclosures=enclosures)
            report.update(result)
            report["status"] = "PASS_SELECTED_GEOMETRY_ONLY" if report["passed"] else "REJECTED_SELECTED_GEOMETRY"
    except (ValueError, TypeError, KeyError, OSError, verify.HumanImportError) as error:
        diagnostic = getattr(error, "diagnostic", None)
        report.update(passed=False, failed_stage=(diagnostic.get("stage", "exact_certificate")
                      if isinstance(diagnostic, dict) else "preparation"), error=str(error))
        if diagnostic is not None:
            report["exact_failure_witness"] = diagnostic
    report.update(elapsed_seconds=time.perf_counter()-start, caller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Selected geometry only. Source ancestry, actual rest-owner emission, motion, continuity and physiology require their existing separate proofs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: report.get(k) for k in ("passed", "status", "failed_stage", "elapsed_seconds", "error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
