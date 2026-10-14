#!/usr/bin/env python3
"""Append the current vessel owner carrier to the exact cap006 ABI5 receipt."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/n/numi-human-current-capture-network-20261014")
sys.path.insert(0, str(REPO / "src"))
from numilab_human.resting_vessel_owner_blend import (
    _sha,
    compose_owner_field_parent_bindings,
)

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
D = A / "full-anatomy-delivery-audit-1378"
CAP = D / "axillary-477-cap-reinstatement-successor-006"
PREP = D / "brachiocephalic-exact-coordinate-owner-preparation-002"
OUT = D / "brachiocephalic-seam-owner-composition-001"
EXPECTED_CAP_RECEIPT = "6333843870187ad4dfc3d68e4847a3e2330b56d9d6913e34ae5f67f87ff80f08"
EXPECTED_CAP_NHA = "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
EXPECTED_PREP_REPORT = "c37ee901ebd617943a8e30885710053d6026348559b4c1c565ab9f45ed8e9aa3"
EXPECTED_PREP_CARRIER = "f4f68f0439d78284d6db578119198ae89250facd88b005d467cda6df5352a687"
EXPECTED_PREP_UPDATES = "6b45944e692f7a8b6faae116d3fa5fe9fe2db240fcc37f4440038a04106a8b47"
EXPECTED_NORMAL_ADDENDUM = "a170bca38831867f6ba9e5940791156ae74848d7fbb808b3513163225b103468"
PREP_SOURCE_GIT_COMMIT = "945d76b8ddaa55bba30dcb402f52c7c1c5bc456e"
PREP_SOURCE_MODULE_SHA = "6fc151f1746c8f3848fdb9fda9bbc9c35f716bbce92af7fecc23cb00d88ec907"
COMPOSITION_MODULE_SHA = "336f1199d6ea44b1c3ddf2b3559b0b08213a169802aa97b34fbaa8b2aa2248ce"

def canonical_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def link_exact(source: Path, target: Path, expected_sha: str) -> None:
    if _sha(source) != expected_sha:
        raise ValueError(f"input asset hash changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"refusing to overwrite output asset: {target}")
    try:
        os.link(source, target)
    except OSError:
        import shutil
        shutil.copy2(source, target)
    if _sha(target) != expected_sha:
        raise ValueError(f"output asset hash differs: {target}")

if OUT.exists():
    raise SystemExit("composition output exists; refusing overwrite")
receipt_path = CAP / "resting-anatomy-receipt.json"
cap_nha = Path(json.loads(receipt_path.read_text())["payload"]["path"])
if _sha(receipt_path) != EXPECTED_CAP_RECEIPT or _sha(cap_nha) != EXPECTED_CAP_NHA:
    raise SystemExit("cap006 receipt or NHA hash changed")
prep_report_path = PREP / "owner-preparation-report.json"
prep_report = json.loads(prep_report_path.read_text())
updates_path = PREP / "parent-surface-updates.json"
carrier_path = PREP / "assets/route110-dual-vessel-reference-carriers.nhtissue"
normal_path = PREP / "normal-hash-addendum.json"
if _sha(prep_report_path) != EXPECTED_PREP_REPORT:
    raise SystemExit("current source-preparation report hash changed")
if _sha(carrier_path) != EXPECTED_PREP_CARRIER:
    raise SystemExit("source-preparation carrier hash changed")
if _sha(updates_path) != EXPECTED_PREP_UPDATES:
    raise SystemExit("source-preparation owner-update hash changed")
if _sha(normal_path) != EXPECTED_NORMAL_ADDENDUM:
    raise SystemExit("normal-hash addendum changed")
module_path = REPO / "src/numilab_human/resting_vessel_owner_blend.py"
if _sha(module_path) != COMPOSITION_MODULE_SHA:
    raise SystemExit("current owner-composition module differs from the tested revision")
prep_module_blob = subprocess.check_output(
    [
        "git", "-C", str(REPO), "show",
        f"{PREP_SOURCE_GIT_COMMIT}:src/numilab_human/resting_vessel_owner_blend.py",
    ]
)
if hashlib.sha256(prep_module_blob).hexdigest() != PREP_SOURCE_MODULE_SHA:
    raise SystemExit("immutable source-preparation Git blob differs from its recorded hash")
if prep_report["repository"]["git_head"] != PREP_SOURCE_GIT_COMMIT:
    raise SystemExit("source-preparation report is not bound to its producing Git revision")

base = json.loads(receipt_path.read_text())
def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result

owner_updates = json.loads(updates_path.read_text(), object_pairs_hook=unique_json_object)
if set(owner_updates) != {"475", "477"}:
    raise SystemExit("prepared owner update keys are not canonical 475/477 strings")
old_bindings = base["passive_tissue_parent_bindings"]
if base["payload"]["sha256"] != EXPECTED_CAP_NHA:
    raise SystemExit("cap006 receipt does not bind its pinned NHA")
if prep_report["owner_field"]["cap006_local_geometry_exact_to_current_source_owner_preparation"] is not True:
    raise SystemExit("source-preparation report lacks exact cap006 row binding")
if prep_report["outputs"]["carrier_payload_sha256"] != EXPECTED_PREP_CARRIER:
    raise SystemExit("source-preparation report carrier binding changed")
if prep_report["outputs"]["parent_surface_updates_sha256"] != EXPECTED_PREP_UPDATES:
    raise SystemExit("source-preparation report map binding changed")
if prep_report["outputs"]["normal_hash_addendum_sha256"] != EXPECTED_NORMAL_ADDENDUM:
    raise SystemExit("source-preparation report normal binding changed")

old_sources = copy.deepcopy(old_bindings["source_payloads"])
base_dir = CAP
source_inventory_hashes = {}
for index, entry in enumerate(old_sources):
    source = Path(entry["path"])
    if not source.is_absolute():
        source = base_dir / source
    got = _sha(source)
    if got != entry["sha256"]:
        raise SystemExit(f"existing parent carrier {index} hash mismatch")
    source_inventory_hashes[str(index)] = {
        "path": str(source),
        "sha256": got,
    }
if len(old_sources) != 7:
    raise SystemExit("cap006 passive source inventory is not the exact seven-entry prefix")
if base["passive_tissue_parent_bindings"]["anatomy_payload_sha256"] != EXPECTED_CAP_NHA:
    raise SystemExit("cap006 passive owner bindings do not pin the exact NHA")

common_files = {
    "common-cardiac-map-f32.bin": "0d88db155f7dfe9a93030badf6e570f49050164a7d0aa0ae011065aeada7d3da",
    "common-cardiac-domains-f32.bin": "7652785c9ea164b16ee96f64a3dcf940398af36cfd535a5f9b7a7d6bd2c7a611",
    "common-cardiac-volumes-f32.bin": "7be4f62996c996c6cfdd49480d27766c8253387bb24aeb504fef73cd553ae699",
}
common_input_hashes = {name: _sha(base_dir / name) for name in common_files}
if common_input_hashes != common_files:
    raise SystemExit("cap006 common-field sidecar hashes changed")
common_field = base["provenance"]["cardiac_geometry_binding"]["common_field"]
if common_field["map"]["record_count"] != 621369 or common_field["map"]["record_stride_bytes"] != 112:
    raise SystemExit("cap006 common map dimensions changed")
if common_field["map"]["sha256"] != common_files["common-cardiac-map-f32.bin"]:
    raise SystemExit("cap006 common map descriptor differs")
range_rows = common_field["passive_attachment_ranges"]
if any(type(row.get("stable_id")) is not int for row in range_rows):
    raise SystemExit("common-field stable IDs must be JSON integers")
range_map = {row["stable_id"]: row for row in range_rows}
if len(range_map) != len(range_rows):
    raise SystemExit("common-field stable IDs are duplicated")
if (range_map[525]["first_vertex"], range_map[525]["vertex_count"]) != (620718, 376):
    raise SystemExit("left bridge common range changed")
if (range_map[526]["first_vertex"], range_map[526]["vertex_count"]) != (621094, 275):
    raise SystemExit("right bridge common range changed")

new_asset_rel = Path("assets/brachiocephalic-seam-owner-carriers.nhtissue")
new_asset_abs = OUT / new_asset_rel
new_bindings, binding_proof = compose_owner_field_parent_bindings(
    old_bindings,
    owner_updates,
    source_payload_path=str(new_asset_abs),
    source_payload_sha256=EXPECTED_PREP_CARRIER,
    prepared_source_index=int(prep_report["owner_field"]["carrier_source_index_standalone"]),
    anatomy_payload_sha256=EXPECTED_CAP_NHA,
    expected_source_count=7,
)
if binding_proof["source_index_appended"] != 7 or binding_proof["source_payload_count_after"] != 8:
    raise SystemExit("prepared owner field did not append as source index 7")
if new_bindings["source_payloads"][:7] != old_sources:
    raise SystemExit("an existing parent source entry changed")
if new_bindings["surfaces"] != old_bindings["surfaces"]:
    raise SystemExit("non-anatomy parent surface mappings changed")

if any(type(row.get("stable_id")) is not int for row in old_bindings["anatomy_surfaces"]):
    raise SystemExit("cap006 anatomy stable IDs must be JSON integers")
if any(type(row.get("stable_id")) is not int for row in new_bindings["anatomy_surfaces"]):
    raise SystemExit("composed anatomy stable IDs must be JSON integers")
old_map_by_id = {row["stable_id"]: row for row in old_bindings["anatomy_surfaces"]}
new_map_by_id = {row["stable_id"]: row for row in new_bindings["anatomy_surfaces"]}
if set(old_map_by_id) != set(new_map_by_id):
    raise SystemExit("anatomy owner stable-ID inventory changed")
for sid, old_row in old_map_by_id.items():
    if sid not in (475, 477) and old_row != new_map_by_id[sid]:
        raise SystemExit(f"non-target owner map {sid} changed")
    if sid in (475, 477):
        old_meta = {k: v for k, v in old_row.items() if k != "vertices"}
        new_meta = {k: v for k, v in new_map_by_id[sid].items() if k != "vertices"}
        if old_meta != new_meta:
            raise SystemExit(f"row {sid} owner metadata changed")

# The inherited extension path is historical and relative. Carry its old payload
# under that name; the new source index receives a distinct filename.
OUT.mkdir()
assets_dir = OUT / "assets"
assets_dir.mkdir()
link_exact(cap_nha, OUT / "resting-thorax.nhanatomy", EXPECTED_CAP_NHA)
for name, digest in common_files.items():
    link_exact(base_dir / name, OUT / name, digest)
old_ext_carrier = Path(old_sources[6]["path"])
if not old_ext_carrier.is_absolute():
    old_ext_carrier = base_dir / old_ext_carrier
link_exact(old_ext_carrier, assets_dir / "route110-dual-vessel-reference-carriers.nhtissue", old_sources[6]["sha256"])
link_exact(carrier_path, new_asset_abs, EXPECTED_PREP_CARRIER)

new_receipt = copy.deepcopy(base)
new_receipt["payload"]["path"] = str(OUT / "resting-thorax.nhanatomy")
new_receipt["passive_tissue_parent_bindings"] = new_bindings
receipt_prov = new_receipt["provenance"]
owner_provenance = {
    "schema": "numi.human.brachiocephalic-exact-coordinate-owner-composition.v1",
    "status": "SOURCE_ONLY_OWNER_FIELD_COMPOSITION_NOT_ANATOMICAL_OR_NATIVE_ADMISSION",
    "input_cap006_receipt_path": str(receipt_path),
    "input_cap006_receipt_sha256": EXPECTED_CAP_RECEIPT,
    "input_payload_sha256": EXPECTED_CAP_NHA,
    "source_preparation_report_path": str(prep_report_path),
    "source_preparation_report_sha256": EXPECTED_PREP_REPORT,
    "source_preparation_normal_hash_addendum_path": str(normal_path),
    "source_preparation_normal_hash_addendum_sha256": EXPECTED_NORMAL_ADDENDUM,
    "source_owner_module_path_at_execution": str(module_path),
    "source_owner_module_sha256": PREP_SOURCE_MODULE_SHA,
    "source_owner_module_git_blob": f"{PREP_SOURCE_GIT_COMMIT}:src/numilab_human/resting_vessel_owner_blend.py",
    "source_owner_module_git_commit": PREP_SOURCE_GIT_COMMIT,
    "composition_owner_module_path": str(module_path),
    "composition_owner_module_sha256": COMPOSITION_MODULE_SHA,
    "source_owner_git_commit": prep_report["repository"]["git_head"],
    "prepared_source_index_standalone": int(prep_report["owner_field"]["carrier_source_index_standalone"]),
    "final_source_index_appended": binding_proof["source_index_appended"],
    "new_carrier_payload_path": str(new_asset_abs),
    "new_carrier_payload_sha256": EXPECTED_PREP_CARRIER,
    "existing_source_payload_prefix_count": 7,
    "existing_source_payload_prefix_semantics_preserved": True,
    "final_source_payload_count": 8,
    "replaced_stable_ids": [475, 477],
    "parent_binding_proof": binding_proof,
    "route110_reference_field_history_preserved": True,
    "left_and_right_bridge_nha_rows_and_common_field_ranges_preserved": True,
    "common_field_sidecars_byte_exact": True,
    "historical_cap_retirement_and_reinstatement_preserved": True,
    "per_slot_normal_bytes_unchanged": True,
    "per_slot_normal_hash_addendum_sha256": EXPECTED_NORMAL_ADDENDUM,
    "additional_force_or_state": False,
    "additional_physical_mass_kg": 0,
    "anatomical_admission": False,
    "native_confirmation_required": True,
    "scope": "Rebind only passive owner rows 475 and 477 to the source-prepared Body20/Body91 carriers; preserve route history, appended bridge rows525/526, all original field bytes, cap history, physical state, and every unrelated owner term.",
}
if "passive_brachiocephalic_exact_coordinate_owner_composition" in receipt_prov:
    raise SystemExit("new provenance key already exists")
receipt_prov["passive_brachiocephalic_exact_coordinate_owner_composition"] = owner_provenance

# Exact support/weight changes relative to cap006, excluding source index.
delta = {}
for sid in (475, 477):
    old_terms = old_map_by_id[sid]["vertices"]
    new_terms = new_map_by_id[sid]["vertices"]
    changed_vertices = 0
    changed_coefficients = []
    support_changes = 0
    minimum_positive = float("inf")
    subnormal_count = 0
    maximum_sum_error = 0.0
    for old_vertex, new_vertex in zip(old_terms, new_terms):
        if any(type(term[1]) is not int or type(term[2]) is not int for term in old_vertex + new_vertex):
            raise SystemExit(f"row {sid} owner surface/vertex IDs must be JSON integers")
        old_weights = {(t[1], t[2]): float(t[3]) for t in old_vertex}
        new_weights = {(t[1], t[2]): float(t[3]) for t in new_vertex}
        if old_weights != new_weights:
            changed_vertices += 1
        for key in set(old_weights) | set(new_weights):
            left = old_weights.get(key, 0.0)
            right = new_weights.get(key, 0.0)
            if left != right:
                changed_coefficients.append(abs(left - right))
            if (left == 0.0) != (right == 0.0):
                support_changes += 1
        total = sum(new_weights.values())
        maximum_sum_error = max(maximum_sum_error, abs(total - 1.0))
        for coefficient in new_weights.values():
            minimum_positive = min(minimum_positive, coefficient)
            if coefficient < 2.0**-126:
                subnormal_count += 1
    delta[str(sid)] = {
        "vertex_count": len(old_terms),
        "changed_vertex_owner_maps": changed_vertices,
        "coefficient_differences": len(changed_coefficients),
        "support_changes": support_changes,
        "maximum_absolute_coefficient_delta": max(changed_coefficients, default=0.0),
        "minimum_positive_coefficient": minimum_positive,
        "subnormal_positive_coefficients": subnormal_count,
        "maximum_positive_sum_error": maximum_sum_error,
    }
owner_provenance["incoming_source_weight_delta"] = delta
owner_provenance["incoming_weights_deliberately_differ_from_cap006"] = True

# Recheck invariants against the composed document before writing it.
if new_receipt["passive_tissue_parent_bindings"]["source_payloads"][:7] != old_sources:
    raise SystemExit("source-prefix preservation failed")
if new_receipt["provenance"]["passive_brachiocephalic_vein_extension"] != base["provenance"]["passive_brachiocephalic_vein_extension"]:
    raise SystemExit("historical brachiocephalic extension changed")
if new_receipt["provenance"]["cardiac_geometry_binding"]["common_field"] != common_field:
    raise SystemExit("common-field descriptor changed")
if new_receipt["functional_bindings"] != base["functional_bindings"]:
    raise SystemExit("functional/circulation binding changed")
if new_receipt["mass_geometry_accounting"] != base["mass_geometry_accounting"]:
    raise SystemExit("mass accounting changed")
if new_receipt["passive_tissue_parent_bindings"]["anatomy_payload_sha256"] != EXPECTED_CAP_NHA:
    raise SystemExit("parent bindings no longer bind exact anatomy payload")

receipt_out = OUT / "resting-anatomy-receipt.json"
with receipt_out.open("w") as stream:
    json.dump(new_receipt, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
if _sha(OUT / "resting-thorax.nhanatomy") != EXPECTED_CAP_NHA:
    raise SystemExit("output NHA changed")
output_common_hashes = {name: _sha(OUT / name) for name in common_files}
if output_common_hashes != common_files:
    raise SystemExit("output common-field sidecar differs")
if _sha(assets_dir / "route110-dual-vessel-reference-carriers.nhtissue") != old_sources[6]["sha256"]:
    raise SystemExit("historical relative extension carrier path changed")
if _sha(new_asset_abs) != EXPECTED_PREP_CARRIER:
    raise SystemExit("new owner carrier changed")

# Reload output and validate unchanged old fields semantically, allowing only its
# payload path, passive parent maps, and the scoped new provenance key to differ.
written = json.loads(receipt_out.read_text())
normalized_bindings = copy.deepcopy(written["passive_tissue_parent_bindings"])
normalized_bindings["source_payloads"] = copy.deepcopy(old_bindings["source_payloads"])
normalized_bindings["anatomy_surfaces"] = copy.deepcopy(old_bindings["anatomy_surfaces"])
normalized_bindings["anatomy_payload_sha256"] = old_bindings["anatomy_payload_sha256"]
if canonical_sha(normalized_bindings) != canonical_sha(old_bindings):
    raise SystemExit("passive parent binding changed outside source append and target owner maps")

def remove_scoped(value):
    result = copy.deepcopy(value)
    result["payload"]["path"] = base["payload"]["path"]
    result["passive_tissue_parent_bindings"] = copy.deepcopy(old_bindings)
    result["provenance"].pop("passive_brachiocephalic_exact_coordinate_owner_composition", None)
    return result
if canonical_sha(remove_scoped(written)) != canonical_sha(base):
    raise SystemExit("receipt changed outside payload relocation, target parent maps, and new scoped provenance")

repo_head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
script_path = Path(__file__).resolve()
source_payload_inventory = {
    "existing_source_payloads": source_inventory_hashes,
    "new_source_payload": {"path": str(new_asset_abs), "sha256": EXPECTED_PREP_CARRIER},
}
report = {
    "schema": "numi.human.brachiocephalic-exact-coordinate-owner-composition-report.v1",
    "status": "SOURCE_ONLY_NOT_LAUNCH_RECEIPT_NOT_ANATOMICAL_ADMISSION",
    "repository": {"path": str(REPO), "git_head": repo_head},
    "runner": {"path": str(script_path), "sha256": _sha(script_path)},
    "code_pins": {
        f"git:{PREP_SOURCE_GIT_COMMIT}:src/numilab_human/resting_vessel_owner_blend.py": PREP_SOURCE_MODULE_SHA,
        str(module_path): COMPOSITION_MODULE_SHA,
        str(script_path): _sha(script_path),
    },
    "inputs": {
        str(receipt_path): EXPECTED_CAP_RECEIPT,
        str(cap_nha): EXPECTED_CAP_NHA,
        str(prep_report_path): EXPECTED_PREP_REPORT,
        str(carrier_path): EXPECTED_PREP_CARRIER,
        str(updates_path): EXPECTED_PREP_UPDATES,
        str(normal_path): EXPECTED_NORMAL_ADDENDUM,
        **{item["path"]: item["sha256"] for item in source_inventory_hashes.values()},
    },
    "source_payload_inventory": source_payload_inventory,
    "outputs": {
        str(receipt_out): _sha(receipt_out),
        str(OUT / "resting-thorax.nhanatomy"): _sha(OUT / "resting-thorax.nhanatomy"),
        str(new_asset_abs): _sha(new_asset_abs),
    },
    "common_field": {
        "map_sha256": output_common_hashes["common-cardiac-map-f32.bin"],
        "domain_boxes_sha256": output_common_hashes["common-cardiac-domains-f32.bin"],
        "polynomials_sha256": output_common_hashes["common-cardiac-volumes-f32.bin"],
        "record_count": common_field["map"]["record_count"],
        "record_stride_bytes": common_field["map"]["record_stride_bytes"],
        "left_bridge_range_525": range_map[525],
        "right_bridge_range_526": range_map[526],
        "descriptors_unchanged": True,
    },
    "composition": {
        "source_index_appended": binding_proof["source_index_appended"],
        "source_payload_count_before": 7,
        "source_payload_count_after": 8,
        "existing_source_prefix_preserved": True,
        "anatomy_surface_ids_preserved": sorted(new_map_by_id),
        "only_anatomy_owner_maps_changed": [475, 477],
        "incoming_source_weight_delta": delta,
        "row475_477_body20_body91_source_owner_proof": prep_report["emitted_owner_map_proof"],
        "all_non_target_parent_map_rows_unchanged": True,
        "non_anatomy_parent_surfaces_unchanged": True,
        "common_map_and_seven_volume_ranges_unchanged": True,
        "historical_route_and_cap_provenance_unchanged": True,
        "functional_bindings_unchanged": True,
        "mass_state_force_fields_unchanged": True,
        "nha_rows_525_526_and_common_attachment_ranges_unchanged": True,
        "per_slot_normal_bytes_unchanged": True,
        "normal_hash_addendum_sha256": EXPECTED_NORMAL_ADDENDUM,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "geometry_predicates_run": False,
        "native_or_gpu_run": False,
        "anatomical_admission": False,
    },
    "limitations": [
        "This is an owner-field source composition on the exact cap006 receipt, not a launch receipt.",
        "No collision, self-intersection, accepted-pose, connected-lumen, caliber, or native gate was run.",
        "The Body20/Body91 source field is inferred and requires the existing all-pose/native gates.",
    ],
}
report_path = OUT / "source-only-composition-report.json"
report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
report_path_sha = _sha(report_path)
print(json.dumps({
    "output_directory": str(OUT),
    "receipt_path": str(receipt_out),
    "receipt_sha256": _sha(receipt_out),
    "payload_sha256": _sha(OUT / "resting-thorax.nhanatomy"),
    "carrier_sha256": _sha(new_asset_abs),
    "common_sidecars": output_common_hashes,
    "source_only_report_path": str(report_path),
    "source_only_report_sha256": report_path_sha,
    "source_index": binding_proof["source_index_appended"],
    "source_payload_count": len(new_bindings["source_payloads"]),
    "owner_deltas": delta,
    "git_head": repo_head,
    "runner_sha256": _sha(script_path),
}, sort_keys=True, indent=2))
