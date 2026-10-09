#!/usr/bin/env python3
"""Collect eight completed 1218 geometry audits without rerunning predicates."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-accepted-geometry-audit-001")
ROOT = BASE.parent
RUN = ROOT / "native-baseline-310s-preparation/native-run"
OUT_DEFAULT = BASE / "eight-pose-aggregate-001"
SCHEDULE = (0, 9983, 19999, 47519, 152191, 153183, 154143, 155000)
F32_DT = 0.0020000000949949026
TARGET_COUNT = 859
MUSCLE_FACES = {(51005, 63): 6688, (51005, 64): 6690}
SKIN_SHA = "ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7"
NHA_SHA = "1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
TISS_SHA = "1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48"
ANATOMY_RECEIPT_SHA = "ebfb61b926f33dd6497176734324e7208e1eeab2f2b518e6d4f884e0a8da99ec"
SCENE_SHA = "6da4ff54bc75029f8d107a1cc9d8a1a2eef233eb8303c5c2943e9c7867a5009d"
GROUPS = {
    BASE / "early-0-20s-001": (0, 9983),
    BASE / "early-40s-002": (19999,),
    BASE / "early-95s-001": (47519,),
    BASE / "late-cycle-001": (152191, 153183, 154143, 155000),
}
SUPPLEMENT = BASE / "source-seam-supplement-001"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def regular(path):
    raw = Path(path).expanduser()
    need(raw.exists() and not raw.is_symlink(), "missing or symlinked evidence: " + str(raw))
    path = raw.resolve()
    need(path.is_file(), "not a regular file: " + str(path))
    return path


def readj(path):
    return json.loads(regular(path).read_text())


def track_file(path, current, expected=None, label="artifact"):
    path = regular(path)
    actual = sha(path)
    need(expected is None or actual == expected, label + " hash mismatch: " + str(path))
    prior = current.get(str(path))
    need(prior is None or prior == actual, "conflicting current hash for " + str(path))
    current[str(path)] = actual
    return path, actual


def merge_current_pins(before, after, label, current):
    need(isinstance(before, dict) and isinstance(after, dict) and bool(before),
         label + " lacks complete before/after input hashes")
    need(before == after, label + " reports changed parent inputs")
    for raw, expected in before.items():
        track_file(raw, current, expected, label + " parent input")
    return len(before)


def arg_value(argv, key):
    try:
        return argv[argv.index(key) + 1]
    except (ValueError, IndexError):
        raise ValueError("native invocation is missing " + key)


def verify_closed_native(current):
    invocation_path = regular(RUN / "invocation.json")
    metadata_path = regular(RUN / "run-metadata.json")
    invocation, metadata = readj(invocation_path), readj(metadata_path)
    argv, assets = invocation.get("argv"), invocation.get("asset_sha256")
    need(isinstance(argv, list) and isinstance(assets, dict), "native invocation lacks argv/assets")
    need(metadata.get("exit_code") == 0 and metadata.get("argv") == argv
         and metadata.get("asset_sha256") == assets,
         "native metadata does not seal this successful invocation")
    need(metadata.get("loaded_metal_runtime", {}).get("verified") is True,
         "loaded native runtime is not verified")
    changed = metadata.get("source_files_changed_during_run")
    need(changed == [] or changed is False, "native source-change state is not clean")
    need(arg_value(argv, "--muscle-step-seconds") == "0.002"
         and arg_value(argv, "--muscle-step-count") == "155000",
         "native run is not the pinned 310 s / 2 ms trial")
    expected_assets = {
        "--skin-payload": (ROOT / "package-preparation-002/composed-candidate/bodyparts3d-myosim-skinned-shell.nhskin", SKIN_SHA),
        "--soft-tissue-payload": (Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"), TISS_SHA),
        "--torso-anatomy-payload": (Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"), NHA_SHA),
        "--resting-anatomy-receipt": (ROOT / "package-preparation-002/composed-candidate/composed-anatomy/resting-anatomy-receipt.json", ANATOMY_RECEIPT_SHA),
    }
    for option, (expected_path, digest) in expected_assets.items():
        actual_path = Path(arg_value(argv, option)).resolve()
        need(actual_path == expected_path.resolve() and assets.get(str(actual_path)) == digest,
             "native invocation asset binding mismatch for " + option)
    scene_path = ROOT / "package-preparation-002/composed-candidate/resting-supine-scene.manifest.json"
    need(assets.get(str(scene_path)) == SCENE_SHA, "native scene manifest pin mismatch")
    for raw, expected in assets.items():
        path = regular(raw)
        need(sha(path) == expected, "current native asset hash mismatch: " + raw)
        current[str(path)] = expected
    for path in (invocation_path, metadata_path):
        current[str(path)] = sha(path)
    terminal_pack = regular(RUN / "accepted-geometry/step-155000.mrvpack")
    terminal_receipt = regular(RUN / "accepted-geometry/step-155000.receipt.json")
    terminal = readj(terminal_receipt)
    need(int(terminal.get("accepted_step", -1)) == 155000
         and math.isclose(float(terminal.get("accepted_time_s", math.nan)),
                          155000 * F32_DT, rel_tol=0, abs_tol=1e-10),
         "successful native run lacks exact accepted terminal capture")
    current[str(terminal_pack)] = sha(terminal_pack)
    current[str(terminal_receipt)] = sha(terminal_receipt)
    return {
        "path": str(RUN), "invocation_path": str(invocation_path),
        "invocation_sha256": sha(invocation_path), "metadata_path": str(metadata_path),
        "metadata_sha256": sha(metadata_path), "exit_code": 0, "loaded_runtime_verified": True,
        "terminal_step": 155000, "terminal_time_s": terminal["accepted_time_s"],
        "terminal_pack_sha256": sha(terminal_pack), "terminal_receipt_sha256": sha(terminal_receipt),
    }


def checked_jsonl(directory, step, suffix, digest, current):
    path = directory / ("step-%d.%s.jsonl" % (step, suffix))
    return track_file(path, current, digest, "evidence artifact")[0]


def collect_skin_group(directory, steps, current):
    summary_path, declaration_path = regular(directory / "summary.json"), regular(directory / "declaration.json")
    summary, declaration = readj(summary_path), readj(declaration_path)
    track_file(summary_path, current)
    track_file(declaration_path, current)
    need(summary.get("schema") == "numi.human.native-accepted-geometry-audit-1218.summary.v1",
         "unexpected skin audit summary schema: " + str(summary_path))
    need(summary.get("native_run") == str(RUN) and declaration.get("native_run") == str(RUN),
         "skin audit used another native run")
    need(summary.get("capture_steps_requested") == list(steps) and summary.get("input_hashes_unchanged") is True,
         "skin audit has wrong steps or changed parent inputs")
    pin_count = merge_current_pins(summary.get("input_hashes_before"),
                                   summary.get("input_hashes_after"), str(summary_path), current)
    raw_poses = summary.get("pose_results", [])
    need(isinstance(raw_poses, list) and len(raw_poses) == len(steps),
         "skin summary has missing or duplicate pose rows")
    poses = {int(row["accepted_step"]): row for row in raw_poses}
    need(len(poses) == len(steps) and set(poses) == set(steps),
         "skin summary pose rows are incomplete or duplicated")
    preflight = declaration.get("capture_file_hashes_preflight", {})
    output = {}
    for step in steps:
        result_path = regular(directory / ("step-%d.result.json" % step))
        result = readj(result_path)
        track_file(result_path, current)
        need(result.get("accepted_step") == step and result.get("status") == "complete_pair_coverage"
             and result.get("pair_coverage_complete") is True
             and result.get("surface_target_count") == TARGET_COUNT
             and result.get("ocular_target_count") == 17
             and result.get("skin_sha256") == SKIN_SHA and result.get("NHA_sha256") == NHA_SHA
             and result.get("contact_exemptions") == [],
             "skin result is not a complete exact 859-target scan at step %d" % step)
        pose = poses[step]
        for key in ("accepted_step", "all_skin_crossing_pair_count", "skin_self_crossing_pair_count",
                    "pair_coverage_complete", "surface_target_count", "invalid_target_triangle_count"):
            need(pose.get(key) == result.get(key), "summary/result mismatch at step %d: %s" % (step, key))
        evidence = {
            "crossing_witnesses": ("crossing-witnesses", result["crossing_witnesses_sha256"]),
            "self_witnesses": ("self-witnesses", result["self_witnesses_sha256"]),
            "invalid_triangles": ("invalid-triangles", result["invalid_triangles_sha256"]),
            "targets": ("targets", result["targets_sha256"]),
        }
        evidence_paths = {label: checked_jsonl(directory, step, suffix, digest, current)
                          for label, (suffix, digest) in evidence.items()}
        pack = regular(RUN / "accepted-geometry" / ("step-%d.mrvpack" % step))
        receipt = regular(RUN / "accepted-geometry" / ("step-%d.receipt.json" % step))
        pack_hash, receipt_hash = sha(pack), sha(receipt)
        declared = preflight.get(str(step), {})
        need(declared.get("pack_sha256") == pack_hash and declared.get("receipt_sha256") == receipt_hash,
             "audit preflight pack/receipt differs from retained run at step %d" % step)
        receipt_doc = readj(receipt)
        expected_time = step * F32_DT
        need(int(receipt_doc.get("accepted_step", -1)) == step
             and math.isclose(float(receipt_doc.get("accepted_time_s", math.nan)), expected_time,
                              rel_tol=0, abs_tol=1e-10),
             "capture receipt step/time mismatch at step %d" % step)
        accepted = result.get("accepted_receipt", {})
        need(accepted.get("accepted_step") == step and accepted.get("sha256") == receipt_hash
             and Path(accepted.get("path", "")).resolve() == receipt.resolve()
             and accepted.get("accepted_time_s") == receipt_doc.get("accepted_time_s")
             and pose.get("accepted_receipt") == accepted,
             "skin result receipt binding mismatch at step %d" % step)
        current[str(pack)] = pack_hash
        current[str(receipt)] = receipt_hash
        output[step] = {
            "audit_directory": str(directory), "summary_path": str(summary_path),
            "summary_sha256": sha(summary_path), "declaration_path": str(declaration_path),
            "declaration_sha256": sha(declaration_path), "parent_input_count": pin_count,
            "audit_started_while_run_open": declaration.get("closed_successful_native_run_at_start") is False,
            "native_log_changed_during_scan": summary.get("native_log_changed_during_scan") is True,
            "result_path": str(result_path), "result_sha256": sha(result_path),
            "pack_path": str(pack), "pack_sha256": pack_hash,
            "receipt_path": str(receipt), "receipt_sha256": receipt_hash,
            "target_count": TARGET_COUNT, "pair_coverage_complete": True,
            "skin_crossing_pair_count": int(result["all_skin_crossing_pair_count"]),
            "skin_self_pair_count": int(result["skin_self_crossing_pair_count"]),
            "invalid_surface_count": int(result["invalid_target_surface_count"]),
            "invalid_triangle_count": int(result["invalid_target_triangle_count"]),
            "degenerate_face_count": len(result.get("skin_degenerate_face_rows", [])),
            "witnesses": {k: {"path": str(v), "sha256": evidence[k][1]} for k, v in evidence_paths.items()},
        }
    return output


def check_muscle(path, step, pack_sha, receipt_sha, current):
    path, report_hash = track_file(path, current, label="muscle report")
    report = readj(path)
    need(report.get("accepted_step") == step and report.get("capture_sha256") == pack_sha
         and report.get("receipt_sha256") == receipt_sha,
         "muscle report capture/receipt mismatch at step %d" % step)
    rows = report.get("surfaces", [])
    by_surface = {tuple(row.get("surface", [])): row for row in rows}
    need(len(rows) == 2 and len(by_surface) == 2 and set(by_surface) == set(MUSCLE_FACES),
         "muscle report does not structurally cover both stable rows")
    per_row_self_ok = []
    per_row_outer_ok = []
    for key, count in MUSCLE_FACES.items():
        row = by_surface[key]
        source_seam = row.get("source_seam_identity_proof", {})
        need(row.get("face_count") == count and row.get("source_face_count") == count
             and row.get("face_count_in_prepared_target") == count
             and row.get("source_face_order_exact") is True
             and source_seam.get("all_identifications_match_source_position_and_route") is True,
             "muscle face/lineage/source-seam structure failed at step %d row %s" % (step, key))
        self_count = int(row.get("self_intersection_count", -1))
        quotient_count = int(row.get("quotient_self_intersection_count", -2))
        inside_count = int(row.get("inside_skin_vertex_count", -1))
        degenerate = row.get("degenerate_face_rows")
        need(self_count >= 0 and quotient_count >= 0 and inside_count >= 0
             and isinstance(degenerate, list),
             "muscle geometry result lacks numeric gate counts at step %d row %s" % (step, key))
        need((row.get("embedded_closed_target") is True) == (quotient_count == 0
             and row.get("all_components_closed_oriented_unused_free") is True),
             "muscle embedded-target flag disagrees with self/topology counts at step %d" % step)
        outer_expected = bool(row.get("embedded_closed_target") is True
             and row.get("all_components_closed_oriented_unused_free") is True and inside_count == 0)
        need(row.get("outer_envelope_clear") is outer_expected,
             "muscle outer-envelope flag disagrees with inside/topology findings at step %d" % step)
        per_row_self_ok.append(bool(self_count == 0 and quotient_count == 0 and not degenerate
             and row.get("all_components_closed_oriented_unused_free") is True))
        per_row_outer_ok.append(outer_expected)
    expected_self = bool(all(per_row_self_ok))
    expected_outer = bool(all(per_row_outer_ok))
    cross_count = int(report.get("stable_63_64_cross_pair_count", -1))
    need(cross_count >= 0 and report.get("all_rows_match_source_face_topology") is True
         and (report.get("all_muscle_self_zero") is True) == expected_self
         and (report.get("all_outer_envelopes_clear") is True) == expected_outer,
         "muscle summary gate booleans disagree with measured per-row findings at step %d" % step)
    witness, witness_hash = track_file(path.parent / ("step-%d.muscle-crossing-witnesses.jsonl" % step),
                                       current, report.get("crossing_witnesses_sha256"), "muscle witness")
    return report, witness


def collect(out):
    out = Path(out).expanduser().resolve()
    need(out.parent == BASE.resolve() and not out.exists(), "output must be a fresh direct child of audit root")
    current = {}
    script_path = regular(__file__)
    script_sha = sha(script_path)
    current[str(script_path)] = script_sha
    native = verify_closed_native(current)
    skin = {}
    for directory, steps in GROUPS.items():
        part = collect_skin_group(directory, steps, current)
        need(not (set(skin) & set(part)), "duplicate skin-step source")
        skin.update(part)
    need(set(skin) == set(SCHEDULE), "skin evidence does not cover exactly the eight declared steps")

    supplement_summary_path = regular(SUPPLEMENT / "summary.json")
    supplement_decl_path = regular(SUPPLEMENT / "declaration.json")
    supplement, supplement_decl = readj(supplement_summary_path), readj(supplement_decl_path)
    track_file(supplement_summary_path, current)
    track_file(supplement_decl_path, current)
    need(supplement.get("schema") == "numi.human.native-muscle-raw-self-seam-diagnosis.v1"
         and supplement.get("status") == "complete_source_seam_contract_supplement"
         and supplement.get("steps") == [19999, 47519]
         and supplement.get("inputs_unchanged") is True
         and supplement_decl.get("steps") == [19999, 47519],
         "source-seam supplement is missing or incomplete")
    supplement_pin_count = merge_current_pins(supplement.get("inputs_before"),
        supplement.get("inputs_after"), str(supplement_summary_path), current)
    raw_supplement_poses = supplement.get("pose_results", [])
    need(isinstance(raw_supplement_poses, list) and len(raw_supplement_poses) == 2,
         "source-seam supplement has missing or duplicate poses")
    supplement_poses = {int(row["accepted_step"]): row for row in raw_supplement_poses}
    need(len(supplement_poses) == 2 and set(supplement_poses) == {19999, 47519},
         "source-seam supplement pose list mismatch")

    steps = []
    for step in SCHEDULE:
        row = skin[step]
        if step in (19999, 47519):
            muscle_path = SUPPLEMENT / ("step-%d.muscle-geometry.json" % step)
            muscle_obj = supplement_poses[step].get("muscle_geometry")
            muscle_from = "source-seam-supplement-001"
            muscle_file = readj(muscle_path)
            track_file(muscle_path, current)
            need(muscle_file == muscle_obj, "supplement report differs from summary at step %d" % step)
        else:
            muscle_path = Path(row["audit_directory"]) / ("step-%d.muscle-geometry.json" % step)
            muscle_from = Path(row["audit_directory"]).name
        muscle, witness = check_muscle(muscle_path, step, row["pack_sha256"], row["receipt_sha256"], current)
        original_muscle_ref = None
        if step in (19999, 47519):
            original_muscle_path = Path(row["audit_directory"]) / ("step-%d.muscle-geometry.json" % step)
            original = readj(original_muscle_path)
            track_file(original_muscle_path, current)
            need(original.get("accepted_step") == step
                 and original.get("capture_sha256") == row["pack_sha256"]
                 and original.get("receipt_sha256") == row["receipt_sha256"],
                 "pre-supplement muscle report is not bound to the same capture")
            old_rows = {tuple(x.get("surface", [])): x for x in original.get("surfaces", [])}
            new_rows = {tuple(x.get("surface", [])): x for x in muscle.get("surfaces", [])}
            need(set(old_rows) == set(new_rows) == set(MUSCLE_FACES),
                 "pre-supplement and source-seam reports differ in stable row coverage")
            for key in MUSCLE_FACES:
                need(old_rows[key].get("self_intersection_count") == new_rows[key].get("raw_indexed_contact_pair_count")
                     and old_rows[key].get("quotient_self_intersection_count") == new_rows[key].get("quotient_self_intersection_count")
                     and old_rows[key].get("face_count") == new_rows[key].get("face_count"),
                     "source-seam supplement did not preserve raw/quotient findings at step %d" % step)
            original_muscle_ref = {"path": str(regular(original_muscle_path)), "sha256": sha(original_muscle_path)}
        steps.append({
            "accepted_step": step, "accepted_time_s": step * F32_DT, "skin": row,
            "muscle_report_source": muscle_from, "muscle_report_path": str(regular(muscle_path)),
            "muscle_report_sha256": sha(muscle_path), "replaced_raw_muscle_report": original_muscle_ref,
            "muscle_witness_path": str(witness),
            "muscle_witness_sha256": sha(witness), "all_muscle_self_zero": muscle["all_muscle_self_zero"],
            "all_outer_envelopes_clear": muscle["all_outer_envelopes_clear"],
            "stable_63_64_cross_pair_count": muscle["stable_63_64_cross_pair_count"],
            "muscle_rows": [{"surface": r["surface"], "face_count": r["face_count"],
                "inside_skin_vertex_count": r["inside_skin_vertex_count"],
                "raw_indexed_contact_pair_count_retained": r["raw_indexed_contact_pair_count"],
                "source_seam_identifications": r["source_seam_identity_proof"]["identified_duplicate_coordinate_records"]}
                for r in muscle["surfaces"]],
        })

    need(steps[3]["skin"]["skin_crossing_pair_count"] > 0,
         "known 95 s skin-intersection failure was not preserved")
    skin_clear = all(s["skin"]["skin_crossing_pair_count"] == 0
        and s["skin"]["skin_self_pair_count"] == 0
        and s["skin"]["invalid_surface_count"] == 0
        and s["skin"]["invalid_triangle_count"] == 0
        and s["skin"]["degenerate_face_count"] == 0 for s in steps)
    muscle_clear = all(s["all_muscle_self_zero"] and s["all_outer_envelopes_clear"]
                       and s["stable_63_64_cross_pair_count"] == 0 for s in steps)
    before = dict(current)
    after = {raw: sha(Path(raw)) for raw in before}
    need(before == after, "one or more inputs changed during aggregation")
    all_geometry_gates_clear = skin_clear and muscle_clear
    status = ("complete_exact_geometry_clear" if all_geometry_gates_clear
              else "complete_with_geometry_failures")
    summary = {
        "schema": "numi.human.native-accepted-geometry-audit-1218.aggregate.v1",
        "status": status, "native_run": native,
        "required_steps": list(SCHEDULE), "completed_steps": [s["accepted_step"] for s in steps],
        "capture_count": len(steps), "target_surface_count": TARGET_COUNT,
        "all_eight_steps_have_complete_859_target_coverage": True,
        "all_parent_inputs_current_and_unchanged": True,
        "source_seam_supplement": {"summary_path": str(supplement_summary_path),
            "summary_sha256": sha(supplement_summary_path), "declaration_path": str(supplement_decl_path),
            "declaration_sha256": sha(supplement_decl_path), "steps": [19999, 47519],
            "parent_input_count": supplement_pin_count},
        "audit_timing_limitation": "The current native run is closed successfully. Parent partial-audit records retain their original open-at-start/native.log-overlap flags; immutable accepted pack/receipt hashes and all tracked parent inputs are independently current. These flags are preserved rather than rewritten as closed-at-scan.",
        "preserved_95_second_failure": {"accepted_step": 47519,
            "expected_original_skin_crossing_pair_count": 18,
            "skin_crossing_pair_count": skin[47519]["skin_crossing_pair_count"],
            "skin_self_pair_count": skin[47519]["skin_self_pair_count"],
            "status": "skin_target_intersection_gate_failed"},
        "all_skin_target_intersection_gates_clear": skin_clear,
        "all_muscle_self_enclosure_cross_gates_clear": muscle_clear,
        "all_geometry_gates_clear": all_geometry_gates_clear,
        "steps": steps, "input_hashes_before_aggregation": before,
        "input_hashes_after_aggregation": after, "inputs_unchanged_during_aggregation": True,
        "qualification": "Eight exact accepted-pose geometry audits only. The known 95 s 18-pair skin intersection failure is preserved. No continuous-time or full-horizon anatomical clearance, physiology, or clinical qualification.",
    }
    out.mkdir()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "README.md").write_text(
        "# 1218 eight-pose geometry audit collector\n\n"
        "References already completed exact scans and witnesses; it does not rerun predicates or duplicate raw evidence. "
        "Requires all eight accepted steps, complete 859-target coverage, current unchanged parent inputs, matching capture hashes, and a closed successful native run. "
        "The 95 s skin-intersection failure is retained; audit overlap/open-at-start flags remain as recorded.\n\n"
        "See summary.json for step-level paths and hashes.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    result = collect(args.out)
    print(json.dumps({"status": result["status"], "steps": result["completed_steps"],
        "parent_inputs_current": result["all_parent_inputs_current_and_unchanged"],
        "95s_failure_pairs": result["preserved_95_second_failure"]["skin_crossing_pair_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()

