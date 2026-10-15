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


def apply_overlay(surfaces, key, path, digest, faces_key="capture_faces"):
    """Bind a fixed-allocation candidate to all ordered captured source GIDs."""
    key = verify.typed_key(key)
    source_pin = pin(path, digest)
    if key not in surfaces:
        raise ValueError(f"candidate typed identity absent from native capture: {key}")
    with np.load(path, allow_pickle=False) as arrays:
        positions = arrays["capture_positions_m"].copy()
        faces = arrays[faces_key].copy()
        gids = arrays["capture_global_vertex_ids"].copy()
        normals = arrays["capture_normals"].copy() if "capture_normals" in arrays else None
        source_fields = source_field_preflight(arrays)
    if normals is not None and normals.dtype != np.dtype("<f4"):
        raise ValueError("capture normals must already be exact Float32")
    if not np.array_equal(gids, surfaces[key][3]) or gids.dtype.kind not in "iu":
        raise ValueError("candidate must preserve every ordered captured source GID")
    if positions.dtype != np.dtype("<f4"):
        raise ValueError("candidate world coordinates must already be exact Float32")
    prepared = verify.prepare_surface(key, positions, faces, normals=normals, source_vertex_ids=gids)
    surfaces[key] = prepared
    return {
        "typed_identity": list(key), "candidate": source_pin, "faces_key": faces_key,
        "all_ordered_capture_gids_exact": True, "geometry_preparation": prepared.diagnostics(),
        "candidate_normals_present": normals is not None,
        "source_field_preflight": source_fields,
        "world_positions_are_counterfactual": True,
        "source_ancestry_and_rest_owner_validation": "separately pinned compiler evidence required",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--pack-sha256", required=True)
    p.add_argument("--capture-receipt", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--source-pin", action="append", default=[], help="PATH=SHA256; compiler/ancestry/owner evidence")
    p.add_argument("--overlay", action="append", default=[], help="SEM:ID=existing.npz=SHA256; counterfactual only")
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
              "counterfactual_candidates": bool(args.overlay)}
    try:
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
        if args.overlay and not pins:
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
        overlay_keys, overlays = set(), []
        for item in args.overlay:
            text, path, digest = item.split("=", 2)
            key = identity(text)
            if key in overlay_keys:
                raise ValueError("duplicate candidate typed identity")
            overlay_keys.add(key)
            overlays.append(apply_overlay(surfaces, key, path, digest, faces_keys.get(key, "capture_faces")))
        if not set(faces_keys) <= overlay_keys:
            raise ValueError("faces-key without matching overlay")
        report["overlays"] = overlays
        changed = [identity(x) for x in args.changed]
        targets = [identity(x) for x in args.target] if args.target else None
        closed = [identity(x) for x in args.required_closed]
        enclosures = [tuple(identity(x) for x in item.split(",")) for item in args.intended_enclosure]
        if any(len(pair) != 2 for pair in enclosures):
            raise ValueError("enclosure requires exactly two typed identities")
        partitions = []
        for first, second, source, digest in args.shared_partition:
            key_a, key_b = identity(first), identity(second)
            pin(source, digest)
            if (key_a in overlay_keys) != (key_b in overlay_keys):
                raise ValueError("counterfactual shared partition requires both sides overlaid")
            partitions.append((key_a, key_b, Path(source), digest))
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
                    coordinate_provenance=("counterfactual_owner_replay" if args.overlay
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
