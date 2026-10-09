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
    _exact_surface_records,
    _face_index_sha256,
    _float32_xyz_sha256,
    _pack_surfaces,
    _target_geometry_f32_sha256,
    _target_intersection_audit,
    audit_incremental_skin_target_intersections,
)

BUNDLE = REPO / "tools/evidence/native-skin-incremental-audit-1228"
ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
RUN = ROOT / "native-baseline-310s-preparation/native-run"
PACK = RUN / "accepted-geometry/step-155000.mrvpack"
RECEIPT = RUN / "accepted-geometry/step-155000.receipt.json"
AUDIT = ROOT / "native-accepted-geometry-audit-001/late-cycle-001"
TARGETS = AUDIT / "step-155000.targets.jsonl"
WITNESSES = AUDIT / "step-155000.crossing-witnesses.jsonl"
RESULT = AUDIT / "step-155000.result.json"
SKIN = ROOT / "package-preparation-002/composed-candidate/bodyparts3d-myosim-skinned-shell.nhskin"
PREVIOUS_FULL = BUNDLE / "reports/comparison-001.json"
OUT = BUNDLE / "reports/comparison-003-final-c7-full-vs-incremental.json"

EXPECTED = {
    str(PACK): "965518a6cacf8e2ff7c48638c8eed8c8d989b8b32f68a1359451f7584a41e62a",
    str(RECEIPT): "c119e4c24f8e85f374d89ffc4d4422102e5b1588d94f22c07358dc5bea99f3cb",
    str(TARGETS): "c061bd7bf55654bdaef0fb7a3987806b3f7b2c7f8a45f2d84fda052507d40f2c",
    str(WITNESSES): "08f6f42bdaec8c317dd1df1b042b6fbd4b80da10a3536957c344118ddd62ff5b",
    str(RESULT): "090977ebadd91362703350be99b7da7823e24e5cc090a8c524bac56e486de67f",
    str(SKIN): "ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7",
    str(PREVIOUS_FULL): "2f485fd0f041b28a57d53e986f4a82bfcd96c2f152e078af715f20908fe5409b",
}
EXPECTED_BASELINE_WORLD = "9f6c3a7f1a2fd5dcfc192557444203cf25043eac1cc74ab3b181da431bcb126e"
EXPECTED_SKIN_FACES = "4284336940104f2101a9adb57bc419b7fdfd7e782be19276c462e59b210664f8"
EXPECTED_TARGET_GEOMETRY = "941a14530ea8c425f9ef420c669ece15ab66e8653cabb36e7d830fa5dbdbcff5"
EXPECTED_BASELINE_PAIR_TABLE = "4d43f9c63aa7e95951d27d4ca0765b3e3a0d00de7f690175d47d585e99b1cdf4"
EXPECTED_CANDIDATE_WORLD = "946629e4e87c37d069a9ba04ed7f186151babf50e39bd6ed7da402e11917f0ff"
EXPECTED_CHANGED_ROWS = [20625, 20626, 20641, 20652, 20653, 21950, 21951, 21952, 21959, 21960, 21965, 23004, 23005, 23034]
EXPECTED_HELPER_SHA = "c7c58cc902aaca199e86c32335936098c2ae4113155ec34e9d0f87aa337306be"
EXPECTED_PREDICATE_SHA = "934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4"

if OUT.exists():
    raise SystemExit(f"refusing to overwrite {OUT}")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pair_digest(pairs: set[tuple[int, int]]) -> str:
    encoded = json.dumps([list(pair) for pair in sorted(pairs)], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

helper_path = REPO / "src/numilab_human/common_atlas_skin_clearance.py"
predicate_path = REPO / "src/numilab_human/cardiac_cavity_intersections.py"
if sha(helper_path) != EXPECTED_HELPER_SHA or sha(predicate_path) != EXPECTED_PREDICATE_SHA:
    raise SystemExit("final source or exact-predicate owner hash changed before execution")
input_hashes_before = {path: sha(Path(path)) for path in EXPECTED}
for path, expected in EXPECTED.items():
    if input_hashes_before[path] != expected:
        raise SystemExit(f"input pin mismatch: {path}")

target_rows = [json.loads(line) for line in TARGETS.read_text().splitlines() if line.strip()]
if len(target_rows) != 859:
    raise SystemExit(f"expected 859 target rows, got {len(target_rows)}")
target_keys = {tuple(map(int, row["surface"])) for row in target_rows}
if len(target_keys) != 859 or any(not row.get("pair_coverage_complete") for row in target_rows):
    raise SystemExit("target inventory is incomplete or has duplicate keys")
if any(row.get("degenerate_face_rows") for row in target_rows):
    raise SystemExit("target inventory includes degenerate faces")
summary = {f"{int(row['surface'][0])}:{int(row['surface'][1])}": row for row in target_rows}

positions, surfaces, _pack_meta = _pack_surfaces(PACK, target_keys)
skin_faces = surfaces[(51007, 1)]["faces"]
target_faces = {key: surfaces[key]["faces"] for key in target_keys}
result_identity = json.loads(RESULT.read_text())
if (result_identity.get("skin_sha256") != EXPECTED[str(SKIN)]
        or result_identity.get("surface_target_count") != 859
        or result_identity.get("pair_coverage_complete") is not True):
    raise SystemExit("retained source scan identity does not bind the pinned skin and target coverage")
baseline_world = np.ascontiguousarray(positions, dtype="<f4")
world_sha = _float32_xyz_sha256(baseline_world)
face_sha = _face_index_sha256(skin_faces)
target_geometry_sha = _target_geometry_f32_sha256(baseline_world, target_faces)
if (world_sha, face_sha, target_geometry_sha) != (EXPECTED_BASELINE_WORLD, EXPECTED_SKIN_FACES, EXPECTED_TARGET_GEOMETRY):
    raise SystemExit("packed world/topology/target geometry does not match the independent identity pins")

baseline_pairs: dict[str, set[tuple[int, int]]] = {key: set() for key in summary}
for line in WITNESSES.read_text().splitlines():
    if not line:
        continue
    row = json.loads(line)
    if (row.get("role") != "surface_crossing"
            or row.get("predicate") != "exact_float32_lattice_triangle_intersection"):
        raise SystemExit("unexpected source witness type")
    key = f"{int(row['target_surface'][0])}:{int(row['target_surface'][1])}"
    pair = (int(row["skin_source_face_row"]), int(row["target_surface_face_row"]))
    if key not in baseline_pairs or pair in baseline_pairs[key]:
        raise SystemExit("source witness table contains missing target or duplicate pair")
    baseline_pairs[key].add(pair)

baseline_audits = {}
for key, row in summary.items():
    pairs = sorted(baseline_pairs[key])
    if len(pairs) != int(row["intersecting_triangle_pairs"]):
        raise SystemExit(f"original exact witnesses disagree with target-summary count for {key}")
    baseline_audits[key] = {
        "triangle_pairs": [[int(a), int(b)] for a, b in pairs],
        "count": len(pairs),
        "aabb_candidate_pairs": int(row["aabb_candidate_pairs"]),
        "degenerate_face_rows": row["degenerate_face_rows"],
    }
baseline_total = sum(len(pairs) for pairs in baseline_pairs.values())
if baseline_total != int(result_identity["all_skin_crossing_pair_count"]):
    raise SystemExit("baseline exact pair total disagrees with original scan aggregate")
baseline_pair_sha = _baseline_target_pair_table_sha256(baseline_audits)
if baseline_pair_sha != EXPECTED_BASELINE_PAIR_TABLE:
    raise SystemExit(f"independently retained baseline pair-table pin mismatch: {baseline_pair_sha}")

first = next(json.loads(line) for line in WITNESSES.read_text().splitlines() if line)
skin_face_row = int(first["skin_source_face_row"])
target_key = tuple(map(int, first["target_surface"]))
target_face_row = int(first["target_surface_face_row"])
skin_vertex_ids = np.asarray(skin_faces[skin_face_row], dtype=np.int64)
target_vertex_ids = np.asarray(target_faces[target_key][target_face_row], dtype=np.int64)
if len(set(map(int, skin_vertex_ids))) != 3:
    raise SystemExit("selected source face has repeated vertex IDs")
all_target_vertex_ids = np.unique(np.concatenate([rows.reshape(-1) for rows in target_faces.values()]))
if np.intersect1d(skin_vertex_ids, all_target_vertex_ids).size:
    raise SystemExit("synthetic fixture skin and target share packed vertex IDs")
candidate_world = baseline_world.copy()
triangle = baseline_world[target_vertex_ids].astype(np.float64)
normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
norm = float(np.linalg.norm(normal))
if not np.isfinite(norm) or norm == 0.0:
    raise SystemExit("selected target triangle normal is invalid")
normal /= norm
candidate_world[skin_vertex_ids] = (candidate_world[skin_vertex_ids].astype(np.float64) + 0.02 * normal).astype("<f4")
if _float32_xyz_sha256(candidate_world) != EXPECTED_CANDIDATE_WORLD:
    raise SystemExit("synthetic audit fixture differs from prior full-scan candidate")
started = time.perf_counter()
incremental = audit_incremental_skin_target_intersections(
    baseline_world_positions=baseline_world,
    candidate_world_positions=candidate_world,
    baseline_faces=skin_faces,
    candidate_faces=skin_faces.copy(),
    baseline_target_audits=baseline_audits,
    target_faces_by_key=target_faces,
    target_positions=baseline_world,
    expected_baseline_world_f32_sha256=EXPECTED_BASELINE_WORLD,
    expected_face_index_sha256=EXPECTED_SKIN_FACES,
    expected_target_geometry_f32_sha256=EXPECTED_TARGET_GEOMETRY,
    expected_baseline_target_audits_sha256=EXPECTED_BASELINE_PAIR_TABLE,
)
incremental_seconds = time.perf_counter() - started
started = time.perf_counter()
full = _target_intersection_audit(
    _exact_surface_records(candidate_world, skin_faces),
    target_faces,
    baseline_world,
)
full_seconds = time.perf_counter() - started

if incremental["changed_skin_face_rows"] != EXPECTED_CHANGED_ROWS:
    raise SystemExit(f"changed face row set differs from pinned fixture: {incremental['changed_skin_face_rows']}")
full_pairs_by_target = {}
incremental_pairs_by_target = {}
baseline_table_by_target = {}
target_records = {}
added_total = removed_total = unchanged_total = 0
for key in sorted(summary):
    full_pairs = {tuple(map(int, pair)) for pair in full[key]["triangle_pairs"]}
    inc_pairs = {tuple(map(int, pair)) for pair in incremental["target_audits"][key]["triangle_pairs"]}
    base_pairs = baseline_pairs[key]
    if full_pairs != inc_pairs:
        raise SystemExit(f"full and incremental exact pair identities differ for {key}")
    changes = incremental["pair_changes_by_target"][key]
    if (set(map(tuple, changes["added"])) != full_pairs - base_pairs
            or set(map(tuple, changes["removed"])) != base_pairs - full_pairs
            or set(map(tuple, changes["unchanged"])) != full_pairs & base_pairs):
        raise SystemExit(f"incremental pair deltas differ from exact set arithmetic for {key}")
    full_pairs_by_target[key] = [[a, b] for a, b in sorted(full_pairs)]
    incremental_pairs_by_target[key] = [[a, b] for a, b in sorted(inc_pairs)]
    baseline_table_by_target[key] = [[a, b] for a, b in sorted(base_pairs)]
    full_pair_sha = pair_digest(full_pairs)
    inc_pair_sha = pair_digest(inc_pairs)
    target_records[key] = {
        "baseline_count": len(base_pairs),
        "baseline_pair_sha256": pair_digest(base_pairs),
        "full_candidate_count": len(full_pairs),
        "full_candidate_pair_sha256": full_pair_sha,
        "full_aabb_candidate_pairs": int(full[key]["aabb_candidate_pairs"]),
        "incremental_candidate_count": len(inc_pairs),
        "incremental_candidate_pair_sha256": inc_pair_sha,
        "incremental_fresh_changed_face_aabb_candidates": int(incremental["target_audits"][key]["fresh_changed_face_aabb_candidate_pairs"]),
        "incremental_aabb_work_scope": incremental["target_audits"][key]["aabb_work_scope"],
        "exact_pair_identities_equal": full_pair_sha == inc_pair_sha,
    }
    added_total += len(changes["added"])
    removed_total += len(changes["removed"])
    unchanged_total += len(changes["unchanged"])

if len(target_records) != 859 or (added_total, removed_total, unchanged_total) != (0, 26, 827):
    raise SystemExit("complete exact target comparison counts mismatch")
full_aabb_total = sum(int(row["aabb_candidate_pairs"]) for row in full.values())
incremental_aabb_total = int(incremental["fresh_changed_face_aabb_candidate_pairs"])
if (full_aabb_total, incremental_aabb_total) != (154938, 79):
    raise SystemExit(f"full/incremental AABB totals differ from retained fixture: {(full_aabb_total, incremental_aabb_total)}")

input_hashes_after = {path: sha(Path(path)) for path in EXPECTED}
if input_hashes_after != input_hashes_before:
    raise SystemExit("an external input changed during this comparison")
if sha(helper_path) != EXPECTED_HELPER_SHA or sha(predicate_path) != EXPECTED_PREDICATE_SHA:
    raise SystemExit("source or predicate owner changed during comparison")

full_pair_table = {key: {"triangle_pairs": full_pairs_by_target[key], "count": len(full_pairs_by_target[key])} for key in full_pairs_by_target}
incremental_pair_table = {key: {"triangle_pairs": incremental_pairs_by_target[key], "count": len(incremental_pairs_by_target[key])} for key in incremental_pairs_by_target}
full_pair_table_sha = _baseline_target_pair_table_sha256(full_pair_table)
incremental_pair_table_sha = _baseline_target_pair_table_sha256(incremental_pair_table)
if full_pair_table_sha != incremental_pair_table_sha:
    raise SystemExit("full and incremental canonical pair-table digests differ")

report = {
    "schema": "numi.human.skin-incremental-target-audit-final-source-full-comparison.v1",
    "status": "pass_exact_pair_identity_equivalence",
    "qualification": "offline one-pose synthetic algorithm equivalence fixture only; not an anatomical candidate, native qualification, or simulation performance result",
    "execution": {
        "command": "PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 runners/compare_final_c7_full_vs_incremental.py",
        "incremental_wall_seconds": incremental_seconds,
        "full_reference_wall_seconds": full_seconds,
        "host_exclusivity": "not established",
    },
    "owners": {
        "incremental_helper_path": str(helper_path),
        "incremental_helper_sha256": sha(helper_path),
        "exact_predicate_owner_path": str(predicate_path),
        "exact_predicate_owner_sha256": sha(predicate_path),
        "runner_path": str(Path(__file__)),
        "runner_sha256": sha(Path(__file__)),
    },
    "external_input_pins_sha256_before_and_after": input_hashes_before,
    "identity_pins": {
        "baseline_world_f32_sha256": world_sha,
        "skin_face_index_sha256": face_sha,
        "target_geometry_f32_sha256": target_geometry_sha,
        "baseline_exact_pair_table_sha256": baseline_pair_sha,
        "full_candidate_exact_pair_table_sha256": full_pair_table_sha,
        "incremental_candidate_exact_pair_table_sha256": incremental_pair_table_sha,
        "full_and_incremental_pair_tables_identical": True,
    },
    "fixture": {
        "kind": "synthetic sparse world-coordinate perturbation; target geometry held fixed",
        "accepted_step": 155000,
        "selected_baseline_skin_face_row": skin_face_row,
        "selected_target": [int(target_key[0]), int(target_key[1])],
        "selected_target_face_row": target_face_row,
        "moved_skin_pack_vertex_ids": [int(v) for v in skin_vertex_ids],
        "translation_m": 0.02,
        "candidate_world_f32_sha256": _float32_xyz_sha256(candidate_world),
        "changed_skin_face_rows": incremental["changed_skin_face_rows"],
    },
    "comparison": {
        "target_surfaces_compared": len(target_records),
        "all_859_full_vs_incremental_exact_pair_identities_equal": True,
        "baseline_exact_pair_count": baseline_total,
        "full_candidate_exact_pair_count": sum(int(row["count"]) for row in full.values()),
        "incremental_candidate_exact_pair_count": sum(int(row["count"]) for row in incremental["target_audits"].values()),
        "added_pair_count": added_total,
        "removed_pair_count": removed_total,
        "unchanged_pair_count": unchanged_total,
        "changed_skin_face_count": len(incremental["changed_skin_face_rows"]),
        "full_scan_aabb_candidate_pairs": full_aabb_total,
        "incremental_fresh_changed_face_aabb_candidates": incremental_aabb_total,
        "per_target": target_records,
        "baseline_exact_pair_table": baseline_table_by_target,
        "full_candidate_exact_pair_table": full_pairs_by_target,
        "incremental_candidate_exact_pair_table": incremental_pairs_by_target,
    },
    "comparison_001_correction": {
        "original_report_sha256": EXPECTED[str(PREVIOUS_FULL)],
        "full_scan_top_level_aabb_total": 154938,
        "incremental_top_level_aabb_total": 79,
        "original_per_target_changed_face_aabb_candidates_field_is_mislabeled": True,
        "correct_per_target_source": "this report per_target.incremental_fresh_changed_face_aabb_candidates; only these changed-face AABB queries were fresh",
    },
    "limits": [
        "This repeats the full exact candidate scan on one retained accepted pose to compare the final helper source directly; it does not establish anatomical validity or native-simulation qualification.",
        "The coordinate change is a synthetic 20 mm translation of three skin vertices along a retained target-face normal and is solely an audit fixture.",
        "No native runtime, GPU, source geometry, target geometry, fitter, or simulation input was changed.",
    ],
}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT),"full_pairs":report["comparison"]["full_candidate_exact_pair_count"],"incremental_pairs":report["comparison"]["incremental_candidate_exact_pair_count"],"targets":len(target_records),"full_aabb":full_aabb_total,"incremental_aabb":incremental_aabb_total,"incremental_wall_s":incremental_seconds,"full_wall_s":full_seconds,"report_sha256":sha(OUT)},indent=2))
