#!/usr/bin/env python3
"""Bounded parity check: legacy vs prepared exact point location on one native capture."""
from __future__ import annotations
import csv, hashlib, json, sys, time, tracemalloc
from fractions import Fraction
from pathlib import Path

REPO = Path("/Users/n/numi-human-free-apex-two-family-1178")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import common_atlas_skin_clearance as clearance

RUN = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217-attempt2/native-run")
PACK = RUN / "accepted-geometry/step-0.mrvpack"
RECEIPT = PACK.with_suffix(".receipt.json")
INVOCATION = RUN / "invocation.json"
METADATA = RUN / "run-metadata.json"
INVENTORY = Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
TISS = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
OUT = Path("/Users/n/numi-human-retained-delivery-20261009/prepared-point-location-1228/real-mesh-comparison-001")
TARGET = (51005, 64)
SKIN = (51007, 1)
EXPECTED_PACK_SHA = "3f3db7c91b73bd737f8b317ffd6edf84004c5477c733b0ca1bdf89bce1f5980c"
EXPECTED_RECEIPT_SHA = "9bd94a969fdf73131ef9ed40486925bd35bea0c26161c0d6a22db1f50cca6d19"
EXPECTED_TISS_SHA = "1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48"
EXPECTED_PRED_SHA = "423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd"
EXPECTED_CLEARANCE_SHA = "c7c58cc902aaca199e86c32335936098c2ae4113155ec34e9d0f87aa337306be"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def deep_size(obj, seen=None):
    if seen is None:
        seen = set()
    marker = id(obj)
    if marker in seen:
        return 0
    seen.add(marker)
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list, set, frozenset)):
        total += sum(deep_size(v, seen) for v in obj)
    elif isinstance(obj, Fraction):
        total += deep_size(obj.numerator, seen) + deep_size(obj.denominator, seen)
    elif hasattr(obj, "__dict__"):
        total += deep_size(vars(obj), seen)
    return total


def main():
    script_path = Path(__file__).resolve()
    static = [PACK, RECEIPT, INVOCATION, METADATA, INVENTORY, TISS,
              Path(ci.__file__), Path(clearance.__file__), script_path]
    before = {str(p): sha(p) for p in static}
    require(before[str(PACK)] == EXPECTED_PACK_SHA, "accepted step-0 pack hash changed")
    require(before[str(RECEIPT)] == EXPECTED_RECEIPT_SHA, "accepted step-0 receipt hash changed")
    require(before[str(TISS)] == EXPECTED_TISS_SHA, "source NHTISS hash changed")
    require(before[str(Path(ci.__file__))] == EXPECTED_PRED_SHA, "predicate source hash changed")
    require(before[str(Path(clearance.__file__))] == EXPECTED_CLEARANCE_SHA, "common-owner source hash changed")
    receipt = json.loads(RECEIPT.read_text())
    meta = json.loads(METADATA.read_text())
    invocation = json.loads(INVOCATION.read_text())
    require(receipt.get("accepted_step") == 0 and receipt.get("accepted_pack_path") == str(PACK)
            and receipt.get("physical_endpoint") == "accepted"
            and receipt.get("surface_audit_endpoint") == "passed", "step-0 capture receipt is not accepted")
    require(meta.get("exit_code") == 0 and meta.get("argv") == invocation.get("argv")
            and meta.get("asset_sha256") == invocation.get("asset_sha256"),
            "native run metadata does not match invocation")

    owner = None
    with INVENTORY.open(newline="") as f:
        for row in csv.DictReader(f):
            if (int(row["second_semantic"]), int(row["second_stable_id"])) == TARGET:
                owner = row
                break
    require(owner is not None and "vastus lateralis" in owner["source_owner_or_label"],
            "target key is not the pinned left vastus lateralis")

    positions, surfaces, pack_counts = clearance._pack_surfaces(PACK, {TARGET})
    target_faces = surfaces[TARGET]["faces"]
    target_ids = np.unique(target_faces)
    target_local_faces = np.searchsorted(target_ids, target_faces)
    target = clearance._prepare_closed_clearance_target(
        positions[target_ids], target_local_faces, allow_nested_enclosure=True)
    clearance._require_prepared_closed_target(target)
    records = target["_interior_records"]
    require(bool(records), "prepared outer muscle component has no exact records")

    tracemalloc.start()
    current_before, _ = tracemalloc.get_traced_memory()
    start_prepare = time.perf_counter()
    prepared = ci.prepare_point_location(records)
    prepare_s = time.perf_counter() - start_prepare
    current_after, peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    require(prepared.records == records, "prepared records differ from caller's exact target records")
    require(prepared.record_identity_sha256, "prepared record identity is absent")

    points = []
    outer_vertices = sorted({tuple(p) for row in records for p in row[0]})
    boundary_indices = np.linspace(0, len(outer_vertices) - 1, min(12, len(outer_vertices)), dtype=int)
    for index in sorted(set(int(i) for i in boundary_indices)):
        points.append(("target_outer_boundary_vertex_%d" % index, outer_vertices[index]))

    skin_faces = surfaces[SKIN]["faces"]
    skin_ids = np.unique(skin_faces)
    skin_samples = np.linspace(0, len(skin_ids) - 1, min(24, len(skin_ids)), dtype=int)
    for ordinal, index in enumerate(sorted(set(int(i) for i in skin_samples))):
        point = clearance.float32_point_lattice_key(positions[int(skin_ids[index])])
        points.append(("captured_skin_vertex_%d" % ordinal, point))

    centroid = tuple(Fraction(sum(Fraction(p[k]) for p in outer_vertices), len(outer_vertices))
                     for k in range(3))
    inside_probe = None
    probe_attempts = 0
    candidates = [("outer-vertex-mean", centroid)]
    denom = 1 << 149
    for face_index in np.linspace(0, len(records) - 1, min(32, len(records)), dtype=int):
        triangle = records[int(face_index)][0]
        normal = ci._cross(ci._sub(triangle[1], triangle[0]), ci._sub(triangle[2], triangle[0]))
        norm = max(abs(v) for v in normal)
        center = tuple(sum(Fraction(v[k]) for v in triangle) / 3 for k in range(3))
        for epsilon_denominator in (100000, 10000):
            delta = Fraction(denom, epsilon_denominator)
            for sign in (-1, 1):
                candidate = tuple(center[k] + sign * Fraction(normal[k], norm) * delta for k in range(3))
                candidates.append(("face-offset-%d-%d-%d" % (face_index, epsilon_denominator, sign), candidate))
    for label, candidate in candidates:
        result = ci.point_location(candidate, records)
        probe_attempts += 1
        if result["location"] == "inside":
            inside_probe = (label, candidate, result)
            break
    require(inside_probe is not None, "could not find an exact inside probe in bounded search")
    points.append(("verified_inside_" + inside_probe[0], inside_probe[1]))
    mins = tuple(min(p[k] for p in outer_vertices) for k in range(3))
    maxs = tuple(max(p[k] for p in outer_vertices) for k in range(3))
    outside = tuple(maxs[k] + 2 * (maxs[k] - mins[k]) + 1 for k in range(3))
    points.append(("verified_outside_far", outside))

    expected = {}
    legacy_start = time.perf_counter()
    for label, point in points:
        expected[label] = ci.point_location(point, records)
    legacy_s = time.perf_counter() - legacy_start
    prepared_start = time.perf_counter()
    observed = {label: ci.point_location_prepared(point, prepared) for label, point in points}
    prepared_s = time.perf_counter() - prepared_start
    require(expected == observed, "prepared locator differs from full legacy result dictionaries")
    classifications = {}
    for result in observed.values():
        classifications[result["location"]] = classifications.get(result["location"], 0) + 1

    after = {str(p): sha(p) for p in static}
    require(after == before, "one or more pinned input sources changed during real-mesh comparison")
    report = {
        "schema": "numi.human.prepared-exact-point-location-real-mesh-comparison.v1",
        "status": "exact_full_result_parity",
        "native_capture": {"run_path": str(RUN), "accepted_step": 0,
                           "pack_path": str(PACK), "pack_sha256": before[str(PACK)],
                           "receipt_path": str(RECEIPT), "receipt_sha256": before[str(RECEIPT)],
                           "native_run_metadata_sha256": before[str(METADATA)],
                           "invocation_sha256": before[str(INVOCATION)]},
        "target": {"semantic": TARGET[0], "stable_id": TARGET[1],
                   "source_owner_or_label": owner["source_owner_or_label"],
                   "captured_face_count": int(len(target_faces)),
                   "captured_target_vertex_count": int(len(target_ids)),
                   "component_count": int(len(target["report"]["component_face_rows"])),
                   "closed_target": target["report"]["embedded_closed_target"],
                   "inside_semantics": target["report"]["inside_semantics"],
                   "outer_component_face_count": int(len(records)),
                   "prepared_record_identity_sha256": prepared.record_identity_sha256},
        "skin_queries": {"surface_key": list(SKIN), "sample_count": int(len(skin_samples)),
                         "captured_skin_vertex_count": int(len(skin_ids))},
        "query_count": len(points), "query_kinds": classifications,
        "exact_inside_probe_search_attempts": probe_attempts,
        "full_result_dictionaries_equal": True,
        "fallback_ray_cache_keys": sorted(prepared._fallback_ray_data),
        "timing_seconds": {"prepare": prepare_s, "legacy_all_queries": legacy_s,
                           "prepared_all_queries": prepared_s},
        "memory_bytes": {"tracemalloc_prepared_current_delta": int(current_after - current_before),
                         "tracemalloc_prepared_peak_delta": int(peak_after - current_before),
                         "prepared_object_deep_size_estimate": int(deep_size(prepared))},
        "pack_geometry_counts": pack_counts,
        "pinned_inputs_before_sha256": before,
        "pinned_inputs_after_sha256": after,
        "qualification": "one accepted native capture, one explicit closed muscle outer-envelope, bounded captured-skin/target probes; not a full skin clearance scan"
    }
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "report.json"
    require(not report_path.exists(), "report path already exists; refusing overwrite")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

