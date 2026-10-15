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
        if args.preflight_only:
            for key in changed:
                if key not in surfaces:
                    raise ValueError(f"changed identity absent: {key}")
            report.update(passed=True, status="PASS_PREPARATION_ONLY_NO_PAIR_PREDICATES")
        else:
            report.update(verify.audit_changed_surfaces(
                surfaces, changed, targets=targets, required_closed=closed, intended_enclosures=enclosures))
            report["status"] = "PASS_SELECTED_GEOMETRY_ONLY" if report["passed"] else "REJECTED_SELECTED_GEOMETRY"
    except (ValueError, TypeError, KeyError, OSError, verify.HumanImportError) as error:
        report.update(passed=False, failed_stage="preparation", error=str(error))
    report.update(elapsed_seconds=time.perf_counter()-start, caller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Selected geometry only. Source ancestry, actual rest-owner emission, motion, continuity and physiology require their existing separate proofs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: report.get(k) for k in ("passed", "status", "failed_stage", "elapsed_seconds", "error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
