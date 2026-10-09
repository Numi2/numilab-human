from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path("/Users/n/numi-human-free-apex-two-family-1178")
sys.path.insert(0, str(REPO / "src"))
from numilab_human.common_atlas_skin_clearance import (
    _baseline_target_pair_table_sha256,
    _face_index_sha256,
    _float32_xyz_sha256,
    _pack_surfaces,
    _target_geometry_f32_sha256,
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
FULL_REFERENCE = ROOT / "incremental-audit-comparison-001/comparison.json"
OUT = ROOT / "incremental-audit-comparison-002/replay-output"
REPORT = OUT / "pinned-baseline-replay.json"
PAIR_TABLE_SHA256 = "4d43f9c63aa7e95951d27d4ca0765b3e3a0d00de7f690175d47d585e99b1cdf4"
EXPECTED_INPUTS = {
    PACK: "965518a6cacf8e2ff7c48638c8eed8c8d989b8b32f68a1359451f7584a41e62a",
    RECEIPT: "c119e4c24f8e85f374d89ffc4d4422102e5b1588d94f22c07358dc5bea99f3cb",
    TARGETS: "c061bd7bf55654bdaef0fb7a3987806b3f7b2c7f8a45f2d84fda052507d40f2c",
    WITNESSES: "08f6f42bdaec8c317dd1df1b042b6fbd4b80da10a3536957c344118ddd62ff5b",
    RESULT: "090977ebadd91362703350be99b7da7823e24e5cc090a8c524bac56e486de67f",
    NHA: "ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7",
    FULL_REFERENCE: "2f485fd0f041b28a57d53e986f4a82bfcd96c2f152e078af715f20908fe5409b",
}
EXPECTED_WORLD_SHA256 = "9f6c3a7f1a2fd5dcfc192557444203cf25043eac1cc74ab3b181da431bcb126e"
EXPECTED_FACE_SHA256 = "4284336940104f2101a9adb57bc419b7fdfd7e782be19276c462e59b210664f8"
EXPECTED_TARGET_GEOMETRY_SHA256 = "941a14530ea8c425f9ef420c669ece15ab66e8653cabb36e7d830fa5dbdbcff5"
EXPECTED_CANDIDATE_WORLD_SHA256 = "946629e4e87c37d069a9ba04ed7f186151babf50e39bd6ed7da402e11917f0ff"
EXPECTED_CHANGED_ROWS = [20625, 20626, 20641, 20652, 20653, 21950, 21951, 21952, 21959, 21960, 21965, 23004, 23005, 23034]

if REPORT.exists() or OUT.exists():
    raise SystemExit(f"refusing to overwrite {OUT}")
OUT.mkdir(parents=True)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

for path, expected in EXPECTED_INPUTS.items():
    actual = sha(path)
    if actual != expected:
        raise SystemExit(f"pinned input hash mismatch: {path}: {actual}")

rows = [json.loads(line) for line in TARGETS.read_text().splitlines() if line.strip()]
if len(rows) != 859:
    raise SystemExit(f"expected 859 target rows, got {len(rows)}")
keys = {tuple(map(int, row["surface"])) for row in rows}
if len(keys) != 859 or any(not row.get("pair_coverage_complete") for row in rows):
    raise SystemExit("target inventory is incomplete or has duplicate keys")
if any(row.get("degenerate_face_rows") for row in rows):
    raise SystemExit("target inventory includes degenerate faces")
summary = {f"{int(row['surface'][0])}:{int(row['surface'][1])}": row for row in rows}

positions, surfaces, pack_meta = _pack_surfaces(PACK, keys)
skin_faces = surfaces[(51007, 1)]["faces"]
target_faces = {key: surfaces[key]["faces"] for key in keys}
result_identity = json.loads(RESULT.read_text())
if (result_identity.get("skin_sha256") != EXPECTED_INPUTS[NHA]
        or result_identity.get("surface_target_count") != 859
        or result_identity.get("pair_coverage_complete") is not True):
    raise SystemExit("retained full-scan identity does not match pinned skin/target coverage")
baseline_world = np.ascontiguousarray(positions, dtype="<f4")
world_sha = _float32_xyz_sha256(baseline_world)
face_sha = _face_index_sha256(skin_faces)
target_geometry_sha = _target_geometry_f32_sha256(baseline_world, target_faces)
if world_sha != EXPECTED_WORLD_SHA256 or face_sha != EXPECTED_FACE_SHA256 or target_geometry_sha != EXPECTED_TARGET_GEOMETRY_SHA256:
    raise SystemExit("packed baseline world/topology/target geometry differs from independent pins")

baseline_pairs = {key: set() for key in summary}
witness_count = 0
for line in WITNESSES.read_text().splitlines():
    if not line:
        continue
    item = json.loads(line)
    if (item.get("role") != "surface_crossing"
            or item.get("predicate") != "exact_float32_lattice_triangle_intersection"):
        raise SystemExit("unexpected baseline witness kind")
    key = f"{int(item['target_surface'][0])}:{int(item['target_surface'][1])}"
    pair = (int(item["skin_source_face_row"]), int(item["target_surface_face_row"]))
    if key not in baseline_pairs or pair in baseline_pairs[key]:
        raise SystemExit("baseline witness table has a missing target or duplicate face pair")
    baseline_pairs[key].add(pair)

baseline_audits = {}
for key, row in summary.items():
    pairs = sorted(baseline_pairs[key])
    if len(pairs) != int(row["intersecting_triangle_pairs"]):
        raise SystemExit(f"exact baseline witnesses disagree with target scan count for {key}")
    baseline_audits[key] = {
        "triangle_pairs": [[int(a), int(b)] for a, b in pairs],
        "count": len(pairs),
        "aabb_candidate_pairs": int(row["aabb_candidate_pairs"]),
        "degenerate_face_rows": row["degenerate_face_rows"],
    }
    witness_count += len(pairs)
if witness_count != int(result_identity["all_skin_crossing_pair_count"]):
    raise SystemExit("exact baseline witness total differs from full scan aggregate")
pair_sha = _baseline_target_pair_table_sha256(baseline_audits)
if pair_sha != PAIR_TABLE_SHA256:
    raise SystemExit(f"independent exact baseline pair-table pin mismatch: {pair_sha}")

full_reference = json.loads(FULL_REFERENCE.read_text())
if full_reference.get("status") != "pass_exact_pair_equivalence":
    raise SystemExit("pinned full-reference comparison is not a passing exact result")
previous_per_target = full_reference["comparison"]["per_target"]
if set(previous_per_target) != set(summary):
    raise SystemExit("full-reference output does not cover the current 859 targets")

first = next(json.loads(line) for line in WITNESSES.read_text().splitlines() if line)
skin_face_row = int(first["skin_source_face_row"])
target_key = tuple(map(int, first["target_surface"]))
target_face_row = int(first["target_surface_face_row"])
skin_vertex_ids = np.asarray(skin_faces[skin_face_row], dtype=np.int64)
target_vertex_ids = np.asarray(target_faces[target_key][target_face_row], dtype=np.int64)
if len(set(map(int, skin_vertex_ids))) != 3:
    raise SystemExit("selected source face has repeated vertex IDs")
all_target_vertex_ids = np.unique(np.concatenate([face.reshape(-1) for face in target_faces.values()]))
if np.intersect1d(skin_vertex_ids, all_target_vertex_ids).size:
    raise SystemExit("fixture skin/target face shares vertex IDs")
candidate = baseline_world.copy()
target_triangle = baseline_world[target_vertex_ids].astype(np.float64)
normal = np.cross(target_triangle[1] - target_triangle[0], target_triangle[2] - target_triangle[0])
length = float(np.linalg.norm(normal))
if not np.isfinite(length) or length == 0.0:
    raise SystemExit("selected target triangle has invalid normal")
normal /= length
candidate[skin_vertex_ids] = (candidate[skin_vertex_ids].astype(np.float64) + 0.02 * normal).astype("<f4")
if _float32_xyz_sha256(candidate) != EXPECTED_CANDIDATE_WORLD_SHA256:
    raise SystemExit("synthetic sparse fixture differs from pinned full-reference input")

started = time.perf_counter()
incremental = audit_incremental_skin_target_intersections(
    baseline_world_positions=baseline_world,
    candidate_world_positions=candidate,
    baseline_faces=skin_faces,
    candidate_faces=skin_faces.copy(),
    baseline_target_audits=baseline_audits,
    target_faces_by_key=target_faces,
    target_positions=baseline_world,
    expected_baseline_world_f32_sha256=EXPECTED_WORLD_SHA256,
    expected_face_index_sha256=EXPECTED_FACE_SHA256,
    expected_target_geometry_f32_sha256=EXPECTED_TARGET_GEOMETRY_SHA256,
    expected_baseline_target_audits_sha256=PAIR_TABLE_SHA256,
)
elapsed = time.perf_counter() - started
changed_rows = incremental["changed_skin_face_rows"]
if changed_rows != EXPECTED_CHANGED_ROWS:
    raise SystemExit(f"changed face coverage differs from pinned full comparison: {changed_rows}")

per_target = {}
for key in sorted(summary):
    current_count = int(incremental["target_audits"][key]["count"])
    expected_count = int(previous_per_target[key]["candidate_pairs"])
    if current_count != expected_count:
        raise SystemExit(f"incremental result count differs from prior full reference for {key}")
    per_target[key] = {
        "full_reference_candidate_pair_count": expected_count,
        "pinned_incremental_candidate_pair_count": current_count,
        "count_matches": True,
        "fresh_changed_face_aabb_candidates": int(incremental["target_audits"][key]["fresh_changed_face_aabb_candidate_pairs"]),
        "reused_baseline_exact_pairs": int(incremental["target_audits"][key]["reused_baseline_exact_pair_count"]),
    }
added = sum(len(row["added"]) for row in incremental["pair_changes_by_target"].values())
removed = sum(len(row["removed"]) for row in incremental["pair_changes_by_target"].values())
unchanged = sum(len(row["unchanged"]) for row in incremental["pair_changes_by_target"].values())
if (added, removed, unchanged) != (0, 26, 827):
    raise SystemExit(f"pair-change totals differ from the prior full comparison: {(added, removed, unchanged)}")
if len(per_target) != 859:
    raise SystemExit("pinned target comparison did not cover all target surfaces")

module_path = REPO / "src/numilab_human/common_atlas_skin_clearance.py"
predicate_path = REPO / "src/numilab_human/cardiac_cavity_intersections.py"
report = {
    "schema": "numi.human.skin-incremental-target-audit-pinned-baseline-replay.v1",
    "status": "pass_baseline_identity_and_prior_full_scan_count_equivalence",
    "qualification": "offline one-pose exact-audit replay; prior full scan compared pair identities; this final-helper replay enforces all four baseline identity pins and rechecks all 859 per-target counts; not a new full candidate scan or anatomical/native qualification",
    "inputs": {
        str(path): {"sha256": digest, "bytes": path.stat().st_size}
        for path, digest in EXPECTED_INPUTS.items()
    },
    "independent_baseline_pins": {
        "baseline_world_f32_sha256": EXPECTED_WORLD_SHA256,
        "baseline_skin_face_index_sha256": EXPECTED_FACE_SHA256,
        "target_geometry_f32_sha256": EXPECTED_TARGET_GEOMETRY_SHA256,
        "baseline_exact_pair_table_sha256": PAIR_TABLE_SHA256,
        "pair_table_derived_from_sha256_pinned_original_witnesses": True,
        "target_summary_all_859_counts_match_witnesses": True,
    },
    "owners": {
        "incremental_helper": str(module_path),
        "incremental_helper_sha256": sha(module_path),
        "exact_predicate_owner": str(predicate_path),
        "exact_predicate_owner_sha256": sha(predicate_path),
        "runner": str(Path(__file__)),
        "runner_sha256": sha(Path(__file__)),
        "prior_full_reference_report": str(FULL_REFERENCE),
        "prior_full_reference_report_sha256": EXPECTED_INPUTS[FULL_REFERENCE],
    },
    "fixture": {
        "kind": "synthetic sparse world-coordinate perturbation; target geometry held fixed",
        "selected_baseline_skin_face_row": skin_face_row,
        "selected_target": [int(target_key[0]), int(target_key[1])],
        "selected_target_face_row": target_face_row,
        "moved_skin_pack_vertex_ids": [int(value) for value in skin_vertex_ids],
        "translation_m": 0.02,
        "changed_skin_face_rows": changed_rows,
        "candidate_world_f32_sha256": incremental["candidate_world_f32_sha256"],
    },
    "comparison": {
        "all_859_target_counts_match_prior_full_exact_scan": True,
        "target_count": len(per_target),
        "baseline_exact_pair_count": witness_count,
        "candidate_exact_pair_count": sum(int(row["pinned_incremental_candidate_pair_count"]) for row in per_target.values()),
        "added_pair_count": added,
        "removed_pair_count": removed,
        "unchanged_pair_count": unchanged,
        "fresh_changed_face_exact_pair_count": incremental["fresh_changed_face_exact_pair_count"],
        "reused_baseline_exact_pair_count": incremental["reused_baseline_exact_pair_count"],
        "fresh_changed_face_aabb_candidate_pairs": incremental["fresh_changed_face_aabb_candidate_pairs"],
        "incremental_wall_seconds": elapsed,
        "prior_full_reference_wall_seconds": full_reference["comparison"]["full_reference_wall_seconds"],
        "per_target": per_target,
    },
    "limits": [
        "The original full exact candidate scan and pair-identity comparison are retained in the pinned prior report; its per-target pair identities were not serialized, so this replay checks per-target counts rather than independently comparing candidate pair identities again.",
        "The exact baseline pair table is rebuilt from SHA-pinned retained crossing witnesses and checked against all 859 SHA-pinned target summary counts before the helper receives its independently frozen digest.",
        "This is one retained pose and a synthetic three-vertex translation. It changes no fitter, source geometry, native runtime, GPU, or simulation input.",
    ],
}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"report": str(REPORT), "baseline_pairs": witness_count, "candidate_pairs": report["comparison"]["candidate_exact_pair_count"], "changed_face_rows": changed_rows, "targets": len(per_target), "incremental_wall_seconds": elapsed, "report_sha256": sha(REPORT)}, indent=2))
