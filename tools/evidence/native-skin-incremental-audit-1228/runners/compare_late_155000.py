from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path("/Users/n/numi-human-free-apex-two-family-1178")
sys.path.insert(0, str(REPO / "src"))
from numilab_human.common_atlas_skin_clearance import (  # noqa: E402
    _exact_surface_records,
    _float32_xyz_sha256,
    _pack_surfaces,
    _target_intersection_audit,
    audit_incremental_skin_target_intersections,
)

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
RUN = ROOT / "native-baseline-310s-preparation/native-run"
PACK = RUN / "accepted-geometry/step-155000.mrvpack"
RECEIPT = RUN / "accepted-geometry/step-155000.receipt.json"
AUDIT = ROOT / "native-accepted-geometry-audit-001/late-cycle-001"
TARGETS = AUDIT / "step-155000.targets.jsonl"
WITNESSES = AUDIT / "step-155000.crossing-witnesses.jsonl"
RESULT = AUDIT / "step-155000.result.json"
NHA = ROOT / "package-preparation-002/composed-candidate/bodyparts3d-myosim-skinned-shell.nhskin"
OUT = ROOT / "incremental-audit-comparison-001"
REPORT = OUT / "comparison.json"

EXPECTED = {
    str(PACK): "965518a6cacf8e2ff7c48638c8eed8c8d989b8b32f68a1359451f7584a41e62a",
    str(RECEIPT): "c119e4c24f8e85f374d89ffc4d4422102e5b1588d94f22c07358dc5bea99f3cb",
    str(TARGETS): None,
    str(WITNESSES): None,
    str(RESULT): None,
    str(NHA): "ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7",
}
if REPORT.exists():
    raise SystemExit(f"refusing to overwrite {REPORT}")
OUT.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path_string, expected in EXPECTED.items():
    actual = sha(Path(path_string))
    if expected is not None and actual != expected:
        raise SystemExit(f"input hash mismatch: {path_string}: {actual}")
    EXPECTED[path_string] = actual

target_rows = [json.loads(line) for line in TARGETS.read_text().splitlines() if line.strip()]
if len(target_rows) != 859:
    raise SystemExit(f"expected 859 target rows, got {len(target_rows)}")
target_keys = {tuple(map(int, row["surface"])) for row in target_rows}
if len(target_keys) != 859 or any(not row.get("pair_coverage_complete") for row in target_rows):
    raise SystemExit("target inventory has duplicate keys or incomplete pair coverage")
if any(row.get("degenerate_face_rows") for row in target_rows):
    raise SystemExit("retained target inventory has degenerate target faces")

pack_positions, surfaces, pack_meta = _pack_surfaces(PACK, target_keys)
skin_faces = surfaces[(51007, 1)]["faces"]
target_faces = {key: surfaces[key]["faces"] for key in target_keys}
result_identity = json.loads(RESULT.read_text())
if (result_identity.get("skin_sha256") != EXPECTED[str(NHA)]
        or result_identity.get("surface_target_count") != 859):
    raise SystemExit("accepted-pose geometry result does not bind the pinned skin and target inventory")
if set(target_faces) != target_keys:
    raise SystemExit("packed target surface coverage differs from 859-target inventory")

summary = {f"{int(row['surface'][0])}:{int(row['surface'][1])}": row for row in target_rows}
baseline_pairs: dict[str, set[tuple[int, int]]] = {key: set() for key in summary}
witness_count = 0
for line in WITNESSES.read_text().splitlines():
    if not line:
        continue
    row = json.loads(line)
    if row.get("role") != "surface_crossing" or row.get("predicate") != "exact_float32_lattice_triangle_intersection":
        raise SystemExit("unexpected retained witness type")
    key = f"{int(row['target_surface'][0])}:{int(row['target_surface'][1])}"
    pair = (int(row["skin_source_face_row"]), int(row["target_surface_face_row"]))
    if key not in baseline_pairs or pair in baseline_pairs[key]:
        raise SystemExit("retained witness has missing target or duplicate face pair")
    baseline_pairs[key].add(pair)
    witness_count += 1

baseline_audits = {}
for key, row in summary.items():
    pairs = sorted(baseline_pairs[key])
    if len(pairs) != int(row["intersecting_triangle_pairs"]):
        raise SystemExit(f"target count does not match retained exact witnesses for {key}")
    baseline_audits[key] = {
        "triangle_pairs": [[int(a), int(b)] for a, b in pairs],
        "count": len(pairs),
        "aabb_candidate_pairs": int(row["aabb_candidate_pairs"]),
        "degenerate_face_rows": row["degenerate_face_rows"],
    }
baseline_total = sum(len(pairs) for pairs in baseline_pairs.values())
if baseline_total != witness_count or baseline_total != sum(int(row["intersecting_triangle_pairs"]) for row in target_rows):
    raise SystemExit("retained witness stream and full 859-target summaries disagree")
if baseline_total != int(result_identity["all_skin_crossing_pair_count"]):
    raise SystemExit("retained witness rows disagree with accepted-pose aggregate pair count")

# Synthetic audit fixture: translate the three vertices of one retained crossing
# face by 20 mm along the selected target-face normal. This is not a proposal.
first = next(json.loads(line) for line in WITNESSES.read_text().splitlines() if line)
skin_face_row = int(first["skin_source_face_row"])
target_key = tuple(map(int, first["target_surface"]))
target_face_row = int(first["target_surface_face_row"])
skin_vertex_ids = np.asarray(skin_faces[skin_face_row], dtype=np.int64)
target_triangle_ids = np.asarray(target_faces[target_key][target_face_row], dtype=np.int64)
if len(set(map(int, skin_vertex_ids))) != 3:
    raise SystemExit("selected source face has repeated vertex IDs")
all_target_vertex_ids = np.unique(np.concatenate([rows.reshape(-1) for rows in target_faces.values()]))
if np.intersect1d(skin_vertex_ids, all_target_vertex_ids).size:
    raise SystemExit("selected skin/target face shares packed vertex IDs; fixture cannot hold targets fixed")

baseline_world = np.ascontiguousarray(pack_positions, dtype="<f4")
candidate_world = baseline_world.copy()
target_triangle = baseline_world[target_triangle_ids].astype(np.float64)
normal = np.cross(target_triangle[1] - target_triangle[0], target_triangle[2] - target_triangle[0])
normal_length = float(np.linalg.norm(normal))
if not np.isfinite(normal_length) or normal_length == 0.0:
    raise SystemExit("selected target triangle has invalid normal")
normal /= normal_length
candidate_world[skin_vertex_ids] = (
    candidate_world[skin_vertex_ids].astype(np.float64) + 0.02 * normal
).astype("<f4")
base_hash = _float32_xyz_sha256(baseline_world)

started = time.perf_counter()
incremental = audit_incremental_skin_target_intersections(
    baseline_world_positions=baseline_world,
    candidate_world_positions=candidate_world,
    baseline_faces=skin_faces,
    candidate_faces=skin_faces.copy(),
    baseline_target_audits=baseline_audits,
    target_faces_by_key=target_faces,
    target_positions=baseline_world,
    expected_baseline_world_f32_sha256=base_hash,
)
incremental_elapsed = time.perf_counter() - started

started = time.perf_counter()
full = _target_intersection_audit(
    _exact_surface_records(candidate_world, skin_faces),
    target_faces,
    baseline_world,
)
full_elapsed = time.perf_counter() - started

added_total = removed_total = unchanged_total = 0
per_target = {}
for key in sorted(summary):
    incremental_pairs = {tuple(pair) for pair in incremental["target_audits"][key]["triangle_pairs"]}
    full_pairs = {tuple(pair) for pair in full[key]["triangle_pairs"]}
    if incremental_pairs != full_pairs or incremental["target_audits"][key]["count"] != full[key]["count"]:
        raise SystemExit(f"incremental/full exact pair mismatch for target {key}")
    old = baseline_pairs[key]
    added, removed, unchanged = incremental_pairs - old, old - incremental_pairs, incremental_pairs & old
    reported = incremental["pair_changes_by_target"][key]
    if (set(map(tuple, reported["added"])) != added
            or set(map(tuple, reported["removed"])) != removed
            or set(map(tuple, reported["unchanged"])) != unchanged):
        raise SystemExit(f"reported pair delta mismatch for target {key}")
    added_total += len(added)
    removed_total += len(removed)
    unchanged_total += len(unchanged)
    per_target[key] = {
        "baseline_pairs": len(old),
        "candidate_pairs": len(full_pairs),
        "added": len(added),
        "removed": len(removed),
        "unchanged": len(unchanged),
        "changed_face_aabb_candidates": int(full[key]["aabb_candidate_pairs"]),
    }

if set(incremental["target_audits"]) != set(summary) or len(per_target) != 859:
    raise SystemExit("candidate audit did not cover every target")

module_path = REPO / "src/numilab_human/common_atlas_skin_clearance.py"
predicate_path = REPO / "src/numilab_human/cardiac_cavity_intersections.py"
report = {
    "schema": "numi.human.skin-incremental-target-audit-actual-pose-comparison.v1",
    "status": "pass_exact_pair_equivalence",
    "qualification": "offline one-pose algorithm equivalence fixture only; not an anatomical candidate or native qualification",
    "inputs": {
        str(path): {"sha256": digest, "bytes": Path(path).stat().st_size}
        for path, digest in EXPECTED.items()
    },
    "owners": {
        "incremental_helper": str(module_path),
        "incremental_helper_sha256": sha(module_path),
        "exact_predicate_owner": str(predicate_path),
        "exact_predicate_owner_sha256": sha(predicate_path),
        "comparison_script": str(Path(__file__)),
        "comparison_script_sha256": sha(Path(__file__)),
        "algorithm": "_target_intersection_audit and cardiac_cavity_intersections exact Float32-lattice triangle predicate",
    },
    "baseline": {
        "accepted_step": 155000,
        "source_skin_face_count": int(len(skin_faces)),
        "source_skin_payload_sha256": EXPECTED[str(NHA)],
        "target_count": len(target_keys),
        "exact_pair_count": baseline_total,
        "world_f32_sha256": base_hash,
        "target_pair_sets_reconstructed_from": str(AUDIT / "step-155000.crossing-witnesses.jsonl"),
        "target_counts_reconstructed_from": str(TARGETS),
        "aggregate_result_identity": str(RESULT),
    },
    "fixture": {
        "kind": "synthetic sparse world-coordinate perturbation; target geometry held fixed",
        "selected_baseline_skin_face_row": skin_face_row,
        "selected_target": [int(target_key[0]), int(target_key[1])],
        "selected_target_face_row": target_face_row,
        "moved_skin_pack_vertex_ids": [int(value) for value in skin_vertex_ids],
        "direction": "positive unit normal of the selected target triangle",
        "translation_m": 0.02,
        "changed_skin_face_rows": incremental["changed_skin_face_rows"],
        "candidate_world_f32_sha256": incremental["candidate_world_f32_sha256"],
    },
    "comparison": {
        "all_859_target_pair_sets_equal": True,
        "target_rows_compared": 859,
        "baseline_pair_count": baseline_total,
        "candidate_pair_count": sum(int(row["count"]) for row in full.values()),
        "added_pair_count": added_total,
        "removed_pair_count": removed_total,
        "unchanged_pair_count": unchanged_total,
        "incremental_fresh_changed_face_exact_pair_count": incremental["fresh_changed_face_exact_pair_count"],
        "incremental_reused_baseline_exact_pair_count": incremental["reused_baseline_exact_pair_count"],
        "incremental_fresh_changed_face_aabb_candidate_pairs": incremental["fresh_changed_face_aabb_candidate_pairs"],
        "full_scan_aabb_candidate_pairs": sum(int(row["aabb_candidate_pairs"]) for row in full.values()),
        "incremental_wall_seconds": incremental_elapsed,
        "full_reference_wall_seconds": full_elapsed,
        "per_target": per_target,
    },
    "limits": [
        "The baseline pair sets are reconstructed from the retained exact crossing witness stream and checked against all 859 target summary counts; the incremental and full candidate audits both use the pinned owner predicate.",
        "This is a single retained accepted pose and a synthetic three-vertex displacement. It tests face-row remapping and exact pair-set equivalence only.",
        "No fitter, source geometry, native runtime, GPU, or simulation input was changed.",
    ],
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "report": str(REPORT),
    "baseline_pairs": baseline_total,
    "candidate_pairs": report["comparison"]["candidate_pair_count"],
    "added": added_total,
    "removed": removed_total,
    "unchanged": unchanged_total,
    "changed_face_rows": incremental["changed_skin_face_rows"],
    "target_rows": len(per_target),
    "incremental_wall_s": incremental_elapsed,
    "full_wall_s": full_elapsed,
    "report_sha256": sha(REPORT),
}, indent=2))
