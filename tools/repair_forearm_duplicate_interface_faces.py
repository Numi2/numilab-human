#!/usr/bin/env python3
"""Pinned local repair for exact duplicate row132/134 interface births.

The script changes face indexing only. Source positions, canonical GIDs, source
slots, all 21 material coefficients and the full source payload remain byte
identical. It stops after topology prechecks; the shared strict verifier still
owns self, mutual, target and anatomical admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/n/numi-human-forearm-parent-bridge-20261017")
D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
H = Path("/Users/n/numi-human-aortic-port-proof-consumer-20261016")
SOURCE_REPORT = D / "forearm-current972-full21-source-emission-1228/report.json"
WORLD = D / "forearm-current972-full21-source-emission-1228/forearm-current972-world-rows.npz"
ROW132_INPUT = D / "forearm-current972-shared-collector-inputs-1233/row132-full21-world-construction-overlay.npz"
ROW134_INPUT = D / "forearm-current972-shared-collector-inputs-1237/row134-full21-world-construction-overlay.npz"
ROW134_INPUT_REPORT = D / "forearm-current972-shared-collector-inputs-1237/report.json"
ROW134_INPUT_ANCESTRY = D / "forearm-current972-shared-collector-inputs-1237/face-ancestry.npz"
ANCESTRY132 = D / "forearm-current972-full21-source-emission-1228/patches/row132-source-ancestry.json"
ANCESTRY134 = D / "forearm-current972-full21-source-emission-1228/patches/row134-source-ancestry.json"

PINS = {
    SOURCE_REPORT: "6fbe0ed784950fd83f7b57e1c77fe63ca3a5f20651e71c2124d62d4008044008",
    WORLD: "753adf4e07f94da5e5a1ae43f2b91c26e94b15a9f70a9f4fddf66a4d53c71ecb",
    ROW132_INPUT: "432debef88da94fc74254edd1d757c55b6665502e169657ca2242d7eed7a9acd",
    ROW134_INPUT: "ed27b03f184083e511102d0028a8b49153b745ed6310ee024f02d4e746bf0472",
    ROW134_INPUT_REPORT: "3550902c2a073523efa3c0530703fec91035f13fb750243b0a3a67851e53bb34",
    ROW134_INPUT_ANCESTRY: "148cd8747777f9f414f28691cc7a48459da01cea80962bb50efb1e58df921cdf",
    ANCESTRY132: "460887139a31d2c2b7c1ae8dd2c8ea0e2be68002b9ce259f625ac546c39564c0",
    ANCESTRY134: "070e9bdd40c26d107bfa1c838d26715fd85ab54325bf4f01d7f774732a915796",
}
PACK_SHA = "80fadf29f92ffff1ec7a3e22c70f645b003738720844689668a7081ed0232e7f"
RECEIPT_SHA = "6b24f77959226729d54fddc8365c4d9e89fd932988b6279479df619f611192cd"
BASE_SOURCE_SHA = "ff8db8396bf49b307ca50ad85f22f19c1167a0f43a625c6439a05b2548bd9126"
ANCESTRAL_CANDIDATE_SHA = "51fd5d77aaf1e1ac839cb3e9d83e947c6386df63d2cf9533db9f92da1dbe25c7"
CHILD_SHA = "44d5dc1c3256962cb85818059ffe1f890cebf864c1f4dbc16ba9b74da4b14b93"
PARENT_EXTENSION_SHA = "aee19bc8e771d6c5ff4d511f6bd75306879705297a73e657336fe34120267331"
H_HEAD = "d5e04d34c2e97b462630a4ac63970cabfb82f8da"
VERIFY_SHA = "d5f12d19b91de330127c883bf0f59fefaf7ed55b4171896bcdda704058956a78"
GEOMETRY_SHA = "f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72"
SOURCE_IDS = {
    132: [
        {"keep_gid": 7962488312534552538, "retire_gid": 8208173142688564573,
         "face_indices": [22263, 22266],
         "representative_reason": "source-backed exact-FDS arrangement ancestry supersedes joint-construction duplicate"},
    ],
    134: [
        {"keep_gid": 7968017219540297305, "retire_gid": 8208901463230337842,
         "face_indices": [2, 5],
         "representative_reason": "both exact-FDS source-backed births; stable lower canonical GID selected deterministically"},
        {"keep_gid": 7962488312534552538, "retire_gid": 8208173142688564573,
         "face_indices": [49, 50],
         "representative_reason": "source-backed exact-FDS arrangement ancestry supersedes joint-construction duplicate"},
    ],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def birth_map(path: Path, expected_payload_sha: str) -> dict[int, dict]:
    data = json.loads(path.read_text())
    if data.get("source_payload_sha256") != expected_payload_sha:
        raise ValueError(f"{path.name}: source payload pin mismatch")
    if data.get("candidate_sha256") != ANCESTRAL_CANDIDATE_SHA:
        raise ValueError(f"{path.name}: ancestral construction candidate pin mismatch")
    if data.get("accepted_capture_step_receipt_sha256") != RECEIPT_SHA:
        raise ValueError(f"{path.name}: accepted current972 receipt pin mismatch")
    if data.get("typed_identity") != [51005, 132] and data.get("typed_identity") != [51005, 134]:
        raise ValueError(f"{path.name}: wrong typed surface lineage")
    result = {int(row["canonical_material_gid"]): row for row in data["births"]}
    if len(result) != len(data["births"]):
        raise ValueError(f"{path.name}: duplicate canonical material GID in births")
    return result


def compact_birth(record: dict, local: int, positions: np.ndarray, weights: np.ndarray,
                  source_slots: np.ndarray, row: int) -> dict:
    if record.get("lineage_source_point_and_full21_field_used") is not True:
        raise ValueError("selected birth lacks source-point/full21 lineage")
    point = np.asarray(record["full21_source_point_f32"], dtype="<f4")
    actual = np.asarray(record["actual_current972_full21_world_f32"], dtype="<f4")
    terms = np.asarray(record["full21_source_weights_float64"], dtype="<f8")
    if point.shape != (3,) or actual.shape != (3,) or terms.shape != (21,):
        raise ValueError("birth source/full21 record has the wrong shape")
    if actual.tobytes() != positions[local].tobytes():
        raise ValueError(f"row{row} birth GID {record['canonical_material_gid']} world target differs from current972")
    if terms.tobytes() != weights[local].tobytes():
        raise ValueError(f"row{row} birth GID {record['canonical_material_gid']} full21 terms differ from emitted rows")
    if not np.all(terms > 0) or abs(float(terms.sum()) - 1.0) > 1e-12:
        raise ValueError("selected birth lacks positive normalized full21 support")
    return {
        "canonical_material_gid": int(record["canonical_material_gid"]),
        "local_vertex": int(local),
        "source_slot": int(source_slots[local]),
        "source_face_support_provenance": record.get("source_face_support_provenance"),
        "source_parent_gids": list(record.get("source_parent_gids", [])),
        "source_point_f32": point.tolist(),
        "actual_current972_full21_world_f32": actual.tolist(),
        "full21_source_weights_float64": terms.tolist(),
        "legacy_route": record.get("legacy_route"),
        "lineage_source_point_and_full21_field_used": True,
    }


def compose_face_ancestry(row: int, original_faces: np.ndarray, original_gids: np.ndarray,
                          current_faces: np.ndarray, final_faces: np.ndarray,
                          current_to_final: np.ndarray, output: Path) -> dict:
    if row == 132:
        if len(original_faces) != len(current_faces):
            raise ValueError("row132 input face count differs from original full21 source row")
        original_to_current = np.arange(len(original_faces), dtype="<i8")
    else:
        with np.load(ROW134_INPUT_ANCESTRY, allow_pickle=False) as z:
            old = z["old_face_indices"].astype("<i8")
            current = z["new_face_indices"].astype("<i8")
        if len(old) != len(current) or len(current) != len(current_faces):
            raise ValueError("row134 retained face ancestry dimensions differ")
        original_to_current = np.full(len(original_faces), -1, dtype="<i8")
        original_to_current[old] = current
    original_to_final = np.full(len(original_faces), -1, dtype="<i8")
    valid = original_to_current >= 0
    original_to_final[valid] = current_to_final[original_to_current[valid]]
    kept_old = np.flatnonzero(original_to_final >= 0).astype("<u4")
    new_ordinals = original_to_final[kept_old].astype("<u4")
    if not np.array_equal(new_ordinals, np.arange(len(final_faces), dtype="<u4")):
        raise ValueError("composed original-to-final face ordinal map is incomplete")
    old_gids = original_gids[original_faces[kept_old]].astype("<u8", copy=True)
    new_gids = original_gids[final_faces].astype("<u8", copy=True)
    changed = np.flatnonzero(np.any(old_gids != new_gids, axis=1)).astype("<u4")
    removed_old = np.flatnonzero(original_to_final < 0).astype("<u4")
    ancestry_path = output / f"row{row}-complete-face-ancestry.npz"
    np.savez_compressed(
        ancestry_path,
        old_to_new_face_ordinal=original_to_final,
        retained_old_face_indices=kept_old,
        retained_new_face_indices=new_ordinals,
        retained_old_face_canonical_gid_corners=old_gids,
        retained_new_face_canonical_gid_corners=new_gids,
        changed_old_face_indices=kept_old[changed],
        changed_old_face_canonical_gid_corners=old_gids[changed],
        changed_new_face_indices=new_ordinals[changed],
        changed_new_face_canonical_gid_corners=new_gids[changed],
        removed_old_face_indices=removed_old,
        removed_old_face_canonical_gid_corners=original_gids[original_faces[removed_old]].astype("<u8"),
    )
    return {
        "path": str(ancestry_path), "sha256": sha(ancestry_path),
        "original_face_count": int(len(original_faces)),
        "final_face_count": int(len(final_faces)),
        "removed_original_face_indices": removed_old.tolist(),
        "rewritten_retained_original_face_indices": kept_old[changed].tolist(),
        "complete_old_to_new_ordinal_map": True,
        "all_retained_faces_have_ordered_old_and_new_canonical_gid_corners": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    output = args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite retained output: {output}")
    for path, expected in PINS.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"input SHA mismatch for {path}: {actual}")
    source_report = json.loads(SOURCE_REPORT.read_text())
    input_hashes = source_report.get("input_sha256", {})
    pack_path = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002/accepted-geometry/step-0.mrvpack"
    receipt_path = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002/accepted-geometry/step-0.receipt.json"
    if (source_report.get("composed_child_sha256") != CHILD_SHA
            or source_report.get("parent_bindings_extension_sha256") != PARENT_EXTENSION_SHA
            or source_report.get("source_candidate_sha256_expected") != "51fd5d77aaf1e1ac839cb3e9d83e947c6386df63d2cf9533db9f92da1dbe25c7"
            or source_report.get("accepted_capture_path") != str(pack_path)
            or source_report.get("accepted_capture_step_receipt_path") != str(receipt_path)
            or input_hashes.get(str(pack_path)) != PACK_SHA
            or input_hashes.get(str(receipt_path)) != RECEIPT_SHA):
        raise ValueError("source emission report differs from accepted full21/current972 pins")

    with np.load(WORLD, allow_pickle=False) as z:
        base = {name: z[name].copy() for name in z.files}
    with np.load(ROW132_INPUT, allow_pickle=False) as z:
        input132 = {name: z[name].copy() for name in z.files}
    with np.load(ROW134_INPUT, allow_pickle=False) as z:
        input134 = {name: z[name].copy() for name in z.files}
    maps = {
        132: {"positions": base["row132_positions_world_f32"],
              "gids": base["row132_canonical_material_gids_u64"],
              "source_slots": base["row132_source_slots_i64"],
              "weights": base["row132_core_weights_float64"],
              "original_faces": base["row132_faces_local_u32"],
              "faces": input132["capture_faces"]},
        134: {"positions": base["row134_positions_world_f32"],
              "gids": base["row134_canonical_material_gids_u64"],
              "source_slots": base["row134_source_slots_i64"],
              "weights": base["row134_core_weights_float64"],
              "original_faces": base["row134_faces_local_u32"],
              "faces": input134["capture_faces"]},
    }
    for row, overlay in ((132, input132), (134, input134)):
        if (not np.array_equal(overlay["capture_positions_m"], maps[row]["positions"])
                or not np.array_equal(overlay["capture_global_vertex_ids"], maps[row]["gids"])):
            raise ValueError(f"row{row} overlay points/GIDs differ from pinned full21 emission")
    if not np.array_equal(maps[132]["faces"], maps[132]["original_faces"]):
        raise ValueError("row132 source face base differs from its current overlay")
    # The earlier row134 repair has a pinned complete source-face map; verify its
    # ordered canonical GID corners before composing a second local patch.
    with np.load(ROW134_INPUT_ANCESTRY, allow_pickle=False) as z:
        old = z["old_face_indices"].astype(np.int64)
        new = z["new_face_indices"].astype(np.int64)
        old_corners = z["old_face_canonical_gids"]
        new_corners = z["new_face_canonical_gids"]
    current134_gids = maps[134]["gids"][maps[134]["faces"]]
    if (not np.array_equal(maps[134]["gids"][maps[134]["original_faces"][old]], old_corners)
            or not np.array_equal(current134_gids[new], new_corners)):
        raise ValueError("row134 previous face-ancestry GID corners fail exact source crosscheck")

    ancestry_maps = {
        132: birth_map(ANCESTRY132, BASE_SOURCE_SHA),
        134: birth_map(ANCESTRY134, BASE_SOURCE_SHA),
    }
    birth_proofs = {}
    for row, specs in SOURCE_IDS.items():
        data = maps[row]
        gid_to_local = {int(gid): i for i, gid in enumerate(data["gids"])}
        records = ancestry_maps[row]
        pair_reports = []
        normalized_specs = []
        for spec in specs:
            keep_gid, retire_gid = spec["keep_gid"], spec["retire_gid"]
            if keep_gid not in records or retire_gid not in records:
                raise ValueError(f"row{row} selected GIDs lack source birth records")
            keep = gid_to_local.get(keep_gid)
            retire = gid_to_local.get(retire_gid)
            if keep is None or retire is None:
                raise ValueError(f"row{row} selected GIDs are absent from emitted row")
            keep_record = compact_birth(records[keep_gid], keep, data["positions"],
                                        data["weights"], data["source_slots"], row)
            retire_record = compact_birth(records[retire_gid], retire, data["positions"],
                                          data["weights"], data["source_slots"], row)
            if data["source_slots"][keep] != -1 or data["source_slots"][retire] != -1:
                raise ValueError("duplicate interface endpoint is no longer an inferred source birth")
            keep_src = np.asarray(keep_record["source_point_f32"], dtype="<f4")
            retire_src = np.asarray(retire_record["source_point_f32"], dtype="<f4")
            if keep_src.tobytes() != retire_src.tobytes():
                raise ValueError("duplicate interface births have different authenticated source points")
            delta = data["weights"][keep] - data["weights"][retire]
            pair_reports.append({
                "keep": keep_record, "retire": retire_record,
                "full21_weight_difference_keep_minus_retire": delta.tolist(),
                "full21_coefficient_max_abs_difference": float(np.max(np.abs(delta))),
                "full21_coefficient_l1_difference": float(np.sum(np.abs(delta))),
                "representative_reason": spec["representative_reason"],
                "same_source_point_bytes": True,
                "same_actual_current972_world_bytes": True,
            })
            normalized_specs.append({k: spec[k] for k in ("keep_gid", "retire_gid", "face_indices")})
        birth_proofs[str(row)] = pair_reports
        output = args.output

    sys.path.insert(0, str(REPO / "src"))
    from numilab_human.forearm_topology_patch import (
            repair_duplicate_interface_birth_faces, referenced_surface_topology_gate)
    output.mkdir(parents=True)
    topology_rows = {}
    for row in (132, 134):
        data = maps[row]
        repaired, patch = repair_duplicate_interface_birth_faces(
            data["faces"], data["positions"], data["gids"],
            [{k: spec[k] for k in ("keep_gid", "retire_gid", "face_indices")}
             for spec in SOURCE_IDS[row]],
            require_closed_oriented=False,
        )
        # Independently certify the helper's output local-vertex/GID contract.
        if len(data["positions"]) != len(data["gids"]) or not np.array_equal(
                data["gids"][repaired], data["gids"][repaired].copy()):
            raise ValueError("internal candidate index/GID crosscheck failed")
        out_overlay = output / f"row{row}-full21-world-construction-overlay.npz"
        np.savez(out_overlay, capture_positions_m=data["positions"],
                 capture_faces=repaired, capture_global_vertex_ids=data["gids"])
        ancestry = compose_face_ancestry(
            row, data["original_faces"], data["gids"], data["faces"], repaired,
            patch["old_to_new_face_ordinal_map"], output)
        topology_rows[str(row)] = {
            "typed_identity": [51005, row],
            "vertices_allocated": int(len(data["positions"])),
            "faces_before_current_patch": int(len(data["faces"])),
            "faces_after_current_patch": int(len(repaired)),
            "overlay_path": str(out_overlay), "overlay_sha256": sha(out_overlay),
            "face_ancestry": ancestry,
            "face_patch": {
                **{k: v for k, v in patch.items()
                   if k not in ("old_to_new_face_ordinal_map", "rewritten_faces_before_removal")},
                "indexed_topology": patch["indexed_topology"],
            },
            "positions_byte_exact_to_current972_source_rows": True,
            "canonical_gids_byte_exact_to_current972_source_rows": True,
            "source_slots_and_all_21_material_terms_byte_exact": True,
            "all_lineage_records_retained_in_source_payload": True,
        }

    output_report = {
        "schema": "numi.human.forearm-exact-duplicate-interface-topology-repair.v1",
        "status": "TOPOLOGY_CANDIDATE_WRITTEN_PENDING_EXACT_QUOTIENT_AND_STRICT_GEOMETRY",
        "typed_identities": [[51005, 132], [51005, 134]],
        "anatomical_admission": False,
        "native_admission": False,
        "accepted_capture": {"pack_sha256": PACK_SHA, "receipt_sha256": RECEIPT_SHA},
        "source_emission": {
            "report_path": str(SOURCE_REPORT), "report_sha256": PINS[SOURCE_REPORT],
            "world_rows_path": str(WORLD), "world_rows_sha256": PINS[WORLD],
            "child_payload_sha256": CHILD_SHA,
            "parent_binding_extension_sha256": PARENT_EXTENSION_SHA,
            "row132_source_ancestry_sha256": PINS[ANCESTRY132],
            "row134_source_ancestry_sha256": PINS[ANCESTRY134],
        },
        "input_overlays": {
            "row132": {"path": str(ROW132_INPUT), "sha256": PINS[ROW132_INPUT]},
            "row134": {"path": str(ROW134_INPUT), "sha256": PINS[ROW134_INPUT]},
            "row134_face_ancestry_sha256": PINS[ROW134_INPUT_ANCESTRY],
        },
        "birth_lineage_and_full21": birth_proofs,
        "rows": topology_rows,
        "strict_topology_precheck": {"status": "NOT_RUN"},
        "strict_geometry_gates": {"status": "NOT_RUN"},
        "note": ("Topology-only face construction. Every source point/GID/source-slot/full21 record remains untouched; "
                 "both lineage records in each duplicate pair remain in the child and parent extension. "
                 "The shared strict verifier must still validate exact-coordinate quotient, closed oriented "
                 "topology, self intersection, mutual surface clearance, and all nearby targets."),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(output_report, sort_keys=True, indent=2) + "\n")

    topology_owner = {
        "git_head_expected": H_HEAD,
        "resting_anatomy_verify_sha256": VERIFY_SHA,
        "cardiac_cavity_geometry_sha256": GEOMETRY_SHA,
    }
    try:
        if (sha(H / "src/numilab_human/resting_anatomy_verify.py") != VERIFY_SHA
                or sha(H / "src/numilab_human/cardiac_cavity_geometry.py") != GEOMETRY_SHA):
            raise ValueError("shared topology owner file SHA differs from pinned review")
        import subprocess
        head = subprocess.check_output(["git", "-C", str(H), "rev-parse", "HEAD"], text=True).strip()
        if head != H_HEAD:
            raise ValueError("shared verifier checkout HEAD differs from pinned review")
        sys.path.insert(0, str(H / "src"))
        import numilab_human
        numilab_human.__path__.append(str(REPO / "src/numilab_human"))
        from numilab_human.resting_anatomy_verify import prepare_surface
        results = {}
        for row in (132, 134):
            data = maps[row]
            with np.load(output / f"row{row}-full21-world-construction-overlay.npz",
                         allow_pickle=False) as z:
                pts, faces, gids = (z["capture_positions_m"], z["capture_faces"],
                                    z["capture_global_vertex_ids"])
            prepared = prepare_surface((51005, row), pts, faces, source_vertex_ids=gids)
            diagnostics = prepared.diagnostics()
            quotient = prepared.quotient_topology
            gate = referenced_surface_topology_gate(prepared)
            ok = gate["passed_referenced_surface_topology"]
            results[str(row)] = {
                **gate,
                "diagnostics": diagnostics,
                "quotient_topology": {
                    key: quotient.get(key) for key in (
                        "vertex_count", "face_count", "edge_count", "face_component_count",
                        "boundary_edge_count", "boundary_edges", "nonmanifold_edges",
                        "orientation_defect_edges", "vertex_manifold_defect_ids",
                        "degenerate_face_ids", "repeated_vertex_face_ids",
                        "duplicate_face_ids", "unused_vertex_ids",
                        "closed_oriented_manifold_candidate",
                    )
                },
            }
        output_report["topology_owner"] = topology_owner
        output_report["strict_topology_precheck"] = {
            "status": "PASS" if all(v["passed_referenced_surface_topology"]
                                    for v in results.values()) else "FAIL",
            "owner": topology_owner,
            "rows": results,
        }
        output_report["status"] = (
            "TOPOLOGY_PASS_PENDING_SHARED_STRICT_GEOMETRY"
            if output_report["strict_topology_precheck"]["status"] == "PASS"
            else "REJECTED_TOPOLOGY_PRECHECK"
        )
    except Exception as exc:
        output_report["topology_owner"] = topology_owner
        output_report["strict_topology_precheck"] = {
            "status": "FAIL", "owner": topology_owner,
            "error_type": type(exc).__name__, "error": str(exc),
        }
        output_report["status"] = "REJECTED_TOPOLOGY_PRECHECK"
    report_path.write_text(json.dumps(output_report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": output_report["status"],
        "output": str(output),
        "report": str(report_path),
        "row_faces": {k: [v["faces_before_current_patch"], v["faces_after_current_patch"]]
                      for k, v in topology_rows.items()},
        "precheck": output_report["strict_topology_precheck"]["status"],
    }, sort_keys=True))
    return 0 if output_report["strict_topology_precheck"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
