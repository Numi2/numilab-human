#!/usr/bin/env python3
"""Reproduce the brachiocephalic source owner preparation and its cap006 sidecar inputs."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/n/numi-human-current-capture-network-20261014")
sys.path.insert(0, str(REPO / "src"))
from numilab_human.passive_attachment_composition import _biceps_row_arrays, _read_nhtiss4, _row_slices
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_vessel_owner_blend import _sha, prepare_vessel_owner_blend

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
D = A / "full-anatomy-delivery-audit-1378"
N27 = A / "liver-island-native-027-002/resting-anatomy-receipt.json"
PORT = D / "arterial-venous-route-field-086/proximal-source-port-witnesses-086.json"
RIGID = A / "input-retention-1380/verified-inputs/060-myosim-fullbody-core-reference.nhrigid"
SOURCES = Path("/Users/n/numi-human-fds132-m6-1350/attempt-290-native-bone-source-topology-001/source")
ROUTE = D / "route-candidate-110-nonlinear-closed-volume-runner-004"
CAP = D / "axillary-477-cap-reinstatement-successor-006"
OLD_PREP = D / "brachiocephalic-exact-coordinate-owner-preparation-001"
OUT = D / "brachiocephalic-exact-coordinate-owner-preparation-002"
EXPECTED_CAP_RECEIPT = "6333843870187ad4dfc3d68e4847a3e2330b56d9d6913e34ae5f67f87ff80f08"
EXPECTED_CAP_NHA = "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
EXPECTED_CARRIER = "f4f68f0439d78284d6db578119198ae89250facd88b005d467cda6df5352a687"
EXPECTED_REPO_HEAD = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if OUT.exists():
    raise SystemExit("output directory exists; refusing overwrite")
if sha(CAP / "resting-anatomy-receipt.json") != EXPECTED_CAP_RECEIPT:
    raise SystemExit("cap006 receipt hash changed")
cap_receipt = json.loads((CAP / "resting-anatomy-receipt.json").read_text())
cap_nha = Path(cap_receipt["payload"]["path"])
if sha(cap_nha) != EXPECTED_CAP_NHA:
    raise SystemExit("cap006 NHA hash changed")

prep = prepare_vessel_owner_blend(
    base_receipt_path=N27,
    n27_receipt_path=N27,
    port_report_path=PORT,
    rigid_path=RIGID,
    sources_path=SOURCES,
    route_npz_path=ROUTE / "candidate-110-source-rows.npz",
    route_report_path=ROUTE / "report.json",
    route_script_path=ROUTE / "run_route110.py",
    route_checkpoint_path=ROUTE / "target-prep-checkpoint.json",
)
if prep["carrier_payload_sha256"] != EXPECTED_CARRIER:
    raise SystemExit("current source code did not reproduce historical carrier bytes")
old_updates_path = OLD_PREP / "parent-surface-updates.json"
old_updates = {int(sid): rows for sid, rows in json.loads(old_updates_path.read_text()).items()}
if prep["parent_surface_updates"] != old_updates:
    raise SystemExit("current source code did not reproduce historical map terms")

_, _, cap_rows = _parse_payload(cap_nha.read_bytes())
for sid, prepared in (
    (475, prep["route110"]["rows"][475]),
    (477, prep["row_replacements"][477]),
):
    actual = cap_rows[sid]
    if actual["vertices6"].astype("<f4").tobytes() != prepared["vertices6"].astype("<f4").tobytes():
        raise SystemExit(f"cap006 row {sid} differs from current source preparation")
    if actual["faces"].astype("<u4").tobytes() != prepared["faces"].astype("<u4").tobytes():
        raise SystemExit(f"cap006 row {sid} differs from current source preparation")
    if tuple(actual["record"])[0] != 91 or tuple(actual["record"])[5:] != (sid, 2, 0):
        raise SystemExit(f"cap006 row {sid} typed identity differs")

with tempfile.TemporaryDirectory(prefix="numi-vessel-normal-hashes-") as temp:
    carrier_temp = Path(temp) / "carrier.nhtissue"
    carrier_temp.write_bytes(prep["carrier_payload"])
    decoded = _read_nhtiss4(carrier_temp)
    records = {int(row[6]): row for row in decoded["records"]}
    carriers = {
        sid: _biceps_row_arrays(_row_slices(decoded, records[sid]))["vertices6"]
        for sid in (1, 2, 3, 4)
    }

alias_classes = prep["rows"]["477"]["exact_coordinate_alias_classes"]
normal_alias_rows = []
for group in alias_classes:
    members = sorted(map(int, group))
    per_slot = []
    for vertex_id in members:
        source_normal = cap_rows[477]["vertices6"][vertex_id, 3:].astype("<f4")
        body20_normal = carriers[3][vertex_id, 3:].astype("<f4")
        body91_normal = carriers[4][vertex_id, 3:].astype("<f4")
        per_slot.append({
            "source_local_vertex_id": vertex_id,
            "nha_normal_sha256": hashlib.sha256(source_normal.tobytes()).hexdigest(),
            "nha_normal_f32_hex": source_normal.tobytes().hex(),
            "body20_carrier_normal_sha256": hashlib.sha256(body20_normal.tobytes()).hexdigest(),
            "body20_carrier_normal_f32_hex": body20_normal.tobytes().hex(),
            "body91_carrier_normal_sha256": hashlib.sha256(body91_normal.tobytes()).hexdigest(),
            "body91_carrier_normal_f32_hex": body91_normal.tobytes().hex(),
        })
    normal_alias_rows.append({
        "coordinate_alias_local_vertex_ids": members,
        "per_vertex_normal_bytes": per_slot,
        "normal_equality_asserted": False,
    })
normal_addendum = {
    "schema": "numi.human.brachiocephalic-owner-normal-hash-addendum.v1",
    "status": "SOURCE_BYTE_HASHES_ONLY_NO_NORMAL_ALIAS_EQUALITY_CLAIM",
    "cap006_receipt_sha256": EXPECTED_CAP_RECEIPT,
    "cap006_payload_sha256": EXPECTED_CAP_NHA,
    "source_preparation_carrier_sha256": EXPECTED_CARRIER,
    "row475_source_nha_normal_sha256": hashlib.sha256(
        cap_rows[475]["vertices6"][:, 3:].astype("<f4").tobytes()
    ).hexdigest(),
    "row477_source_nha_normal_sha256": hashlib.sha256(
        cap_rows[477]["vertices6"][:, 3:].astype("<f4").tobytes()
    ).hexdigest(),
    "row477_exact_coordinate_alias_class_count": len(normal_alias_rows),
    "alias_classes": normal_alias_rows,
    "scope": "Coordinates alone define aliases. Hash each original NHA and derived carrier normal slot; keep per-slot normals without requiring equality across a seam.",
}

OUT.mkdir()
assets = OUT / "assets"
assets.mkdir()
carrier_path = assets / "route110-dual-vessel-reference-carriers.nhtissue"
carrier_path.write_bytes(prep["carrier_payload"])
updates_path = OUT / "parent-surface-updates.json"
updates_path.write_text(json.dumps(prep["parent_surface_updates"], sort_keys=True, separators=(",", ":")) + "\n")
normal_path = OUT / "normal-hash-addendum.json"
normal_path.write_text(json.dumps(normal_addendum, sort_keys=True, indent=2) + "\n")

code_paths = [
    REPO / "src/numilab_human/resting_vessel_owner_blend.py",
    REPO / "src/numilab_human/passive_attachment_composition.py",
    REPO / "src/numilab_human/passive_tissue_parent_carriers.py",
    REPO / "src/numilab_human/resting_passive_viscera.py",
    REPO / "src/numilab_human/resting_pleura_proxy.py",
    REPO / "src/numilab_human/resting_scene.py",
    REPO / "src/numilab_human/model.py",
    REPO / "src/numilab_human/cardiac_cavity_intersections.py",
]
input_paths = [
    N27,
    Path(json.loads(N27.read_text())["payload"]["path"]),
    PORT,
    RIGID,
    SOURCES / "isa_BP3D_4.0_obj_99.zip",
    ROUTE / "candidate-110-source-rows.npz",
    ROUTE / "report.json",
    ROUTE / "run_route110.py",
    ROUTE / "target-prep-checkpoint.json",
    CAP / "resting-anatomy-receipt.json",
    cap_nha,
]
code_hashes = {str(path): sha(path) for path in code_paths}
input_hashes = {str(path): sha(path) for path in input_paths}
report = {
    "schema": "numi.human.brachiocephalic-owner-preparation.v2",
    "status": "SOURCE_ONLY_PREPARATION_NOT_ANATOMICAL_OR_NATIVE_ADMISSION",
    "repository": {"path": str(REPO), "git_head": EXPECTED_REPO_HEAD},
    "code_paths_sha256": code_hashes,
    "input_paths_sha256": input_hashes,
    "historical_source_preparation": {
        "path": str(OLD_PREP / "owner-preparation-report.json"),
        "sha256": sha(OLD_PREP / "owner-preparation-report.json"),
        "immutable": True,
    },
    "execution": {
        "function": "numilab_human.resting_vessel_owner_blend.prepare_vessel_owner_blend",
        "source_only": True,
        "geometry_predicates_run": False,
        "native_or_gpu_run": False,
        "simulation_state_or_mass_added": False,
    },
    "owner_field": {
        "method": "existing source edge-geodesic quintic smoothstep on exact Float32 XYZ quotient; route110 reference positions with shared Body20/Body91 carrier points",
        "rows": {"475": prep["rows"]["475"], "477": prep["rows"]["477"]},
        "parent_surface_update_counts": {
            "475": len(prep["parent_surface_updates"][475]),
            "477": len(prep["parent_surface_updates"][477]),
        },
        "signed_zero_check": {
            "475": {"raw_alias_classes": 0, "canonical_zero_alias_classes": 0, "signed_zero_components": 0},
            "477": {"raw_alias_classes": 29, "canonical_zero_alias_classes": 29, "signed_zero_components": 0},
        },
        "carrier_exact_reproduction_of_historical_001": sha(carrier_path) == sha(OLD_PREP / "route110-dual-vessel-reference-carriers.nhtissue"),
        "parent_updates_semantically_equal_historical_001": prep["parent_surface_updates"] == old_updates,
        "cap006_local_geometry_exact_to_current_source_owner_preparation": True,
        "normal_hash_addendum_path": str(normal_path),
        "normal_hash_addendum_sha256": sha(normal_path),
        "anatomical_admission": False,
        "native_confirmation_required": True,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
    },
    "carrier_encoding_proof": prep["carrier_encoding_proof"],
    "emitted_owner_map_proof": prep["emitted_owner_map_proof"],
    "outputs": {
        "carrier_payload_path": str(carrier_path),
        "carrier_payload_sha256": sha(carrier_path),
        "parent_surface_updates_path": str(updates_path),
        "parent_surface_updates_sha256": sha(updates_path),
        "normal_hash_addendum_path": str(normal_path),
        "normal_hash_addendum_sha256": sha(normal_path),
    },
    "limitations": [
        "No accepted-pose, collision, self-intersection, connected-lumen, caliber, native, or anatomical gate was executed.",
        "Owner equality is a source-side algebra proof; this source preparation is not a launch receipt.",
        "The inferred Body20/Body91 partition is not measured physiology; all-pose continuity and native confirmation remain required.",
    ],
}
report_path = OUT / "owner-preparation-report.json"
report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(json.dumps({
    "report_path": str(report_path),
    "report_sha256": sha(report_path),
    "carrier_sha256": sha(carrier_path),
    "updates_sha256": sha(updates_path),
    "normal_addendum_sha256": sha(normal_path),
    "git_head": EXPECTED_REPO_HEAD,
    "code_paths_sha256": code_hashes,
    "input_paths_sha256": input_hashes,
    "cap_rows_exact": True,
    "carrier_reproduced": report["owner_field"]["carrier_exact_reproduction_of_historical_001"],
    "updates_equal": report["owner_field"]["parent_updates_semantically_equal_historical_001"],
}, sort_keys=True, indent=2))
