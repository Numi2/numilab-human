"""Source-only interpolation of the pinned common cardiac map onto brachiocephalic veins."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .myosim_bone_proximity import _closest_point_on_triangle
from .resting_anatomy_interface_patch import HEADER as NHANAT_HEADER
from .resting_anatomy_interface_patch import RECORD as NHANAT_RECORD
from .resting_passive_viscera import _validated_torso_rest_body
from .resting_pleura_proxy import _parse_payload

N27_RECEIPT_SHA = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
N27_PAYLOAD_SHA = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
SOURCE_ARCHIVE_SHA = "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
REGISTRATION_SHA = "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"
RIGID_SHA = "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
QUOTIENT_SHA = "f189f18efcbfa2953374caf5c3b11af90b135f972764713953de3360bbe140da"
QUOTIENT_NPZ_SHA = "02df2dba7d698d8e22ca7b6493d1b0b97d1afa00b06150b39d1e0a7990e3e385"
ROUTE_HASHES = {
    "ancestry": "e72864fb11923322a39e6a924481fc3928e060fe029d7baa68ca1740c82a544a",
    "attribution": "72e3bddc19a9ca404678db3a2a274031b82a0d01981fa9addbcc01776d37e94d",
    "replay": "e0120f894560890b1c3ddf373ff417a23f1ecc1139fd5546fa25516869a0be1f",
    "stationarity": "014f7d41e6e4e1079bc390c365e9af13d1be9f5410deb499821df7847dd457e7",
    "distances": "5f0a714e88691f3c4f230e7e214d0ed34e6014b222ce58a943635ae2d0e38806",
}
BRIDGES = (
    {"member": "FJ3482", "stable_id": 525, "ports": (469, 473),
     "dist": {"10": "FJ3482_distance_to_10_mm", "469": "FJ3482_distance_to_469_mm",
              "473": "FJ3482_distance_to_473_mm"},
     "xyz": "5df0d59d7ab0d83198d798b54c6a31289f61f6a589bf036862901c63a3d475bd",
     "faces": "2fa66bf9e54a16450c710720c99b1cebf17340c0bd51010c4be81e354816885d"},
    {"member": "FJ3583", "stable_id": 526, "ports": (468, 472),
     "dist": {"10": "FJ3583_distance_to_10_mm", "468": "FJ3583_distance_to_468_mm",
              "472": "FJ3583_distance_to_472_mm"},
     "xyz": "72f976a105a01d9670fa2b94c0a148d41c2e1843fefb5d8a38314b7f3de5bd6e",
     "faces": "9147eadfba51c64fbde14f82a6ae606e4c60f7b792e2488b86b6552fcbffbf16"},
)

def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("brachiocephalic common field: " + message)

def quintic_smoothstep(value: Any, np: Any) -> Any:
    t = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return t*t*t*(t*(t*6.0-15.0)+10.0)

def _resolve_sidecar(receipt_path: Path, base_receipt_path: Path, record: dict) -> Path:
    rel = Path(record["path"])
    candidates = [rel] if rel.is_absolute() else [receipt_path.parent / rel,
                                                  base_receipt_path.parent / rel]
    for path in candidates:
        if path.is_file() and _sha_file(path) == record.get("sha256"):
            return path.resolve()
    raise ValueError("brachiocephalic common field: sidecar missing/hash mismatch: " + str(rel))

def _validate_common_field_descriptor(candidate: dict, pinned_base: dict) -> None:
    """Allow only the anatomy rebind and sidecar path relocation on a child."""
    _require(isinstance(candidate, dict) and isinstance(pinned_base, dict),
             "candidate or pinned N27 common-field descriptor is missing")
    child = copy.deepcopy(candidate)
    base = copy.deepcopy(pinned_base)
    _require(isinstance(child.get("anatomy_payload_sha256"), str)
             and isinstance(base.get("anatomy_payload_sha256"), str),
             "common-field anatomy payload binding is malformed")
    child["anatomy_payload_sha256"] = base["anatomy_payload_sha256"]
    for owner in ("map", "polynomials", "domain_boxes"):
        child_record, base_record = child.get(owner), base.get(owner)
        _require(isinstance(child_record, dict) and isinstance(base_record, dict)
                 and isinstance(child_record.get("path"), str)
                 and isinstance(base_record.get("path"), str),
                 "common-field sidecar descriptor is malformed: " + owner)
        child_record["path"] = base_record["path"]
    _require(child == base,
             "candidate common-field descriptor differs from pinned N27 outside anatomy hash/path relocation")


def _source_code_pins() -> dict[str, str]:
    """Bind this preparation to the existing serializer and source owners."""
    source_dir = Path(__file__).resolve().parent
    names = (
        "resting_brachiocephalic_common_field.py",
        "resting_brachiocephalic_source007.py",
        "resting_major_vessels.py",
        "resting_passive_viscera.py",
        "resting_vessel_owner_blend.py",
        "resting_intercostal_mesh_arrangement.py",
        "resting_anatomy_interface_patch.py",
        "resting_pleura_proxy.py",
    )
    return {name: _sha_file(source_dir / name) for name in names}


def _payload_sections(raw: bytes) -> tuple[tuple[bytes, bytes, bytes], tuple]:
    _require(len(raw) >= NHANAT_HEADER.size, "truncated NHANAT payload")
    header = NHANAT_HEADER.unpack_from(raw)
    magic, abi, nrow, nvertex, nindex = header[:5]
    _require(magic == b"NHANAT1\0" and abi == 5, "expected NHANAT ABI5")
    r0 = NHANAT_HEADER.size
    v0 = r0 + nrow * NHANAT_RECORD.size
    i0 = v0 + nvertex * 24
    _require(len(raw) == i0 + nindex * 4, "NHANAT byte length differs from header")
    return (raw[r0:v0], raw[v0:i0], raw[i0:]), (header[0], *map(int, header[1:6]), header[6])

def _validate_n27_parent(receipt: dict, candidate_raw: bytes, base_receipt_path: Path) -> dict:
    ext = receipt.get("provenance", {}).get("passive_brachiocephalic_vein_extension")
    _require(isinstance(ext, dict), "missing source ABI5 vein append provenance")
    _require(Path(ext.get("base_receipt", "")).resolve() == base_receipt_path.resolve(),
             "vein append is not bound directly to pinned N27")
    _require(_sha_file(base_receipt_path) == N27_RECEIPT_SHA
             and ext.get("base_receipt_sha256") == N27_RECEIPT_SHA,
             "pinned N27 receipt SHA changed")
    base_receipt = json.loads(base_receipt_path.read_text())
    base_path = Path(base_receipt["payload"]["path"])
    _require(base_receipt["payload"]["sha256"] == N27_PAYLOAD_SHA
             and _sha_file(base_path) == N27_PAYLOAD_SHA
             and ext.get("base_payload_sha256") == N27_PAYLOAD_SHA,
             "vein append is not bound directly to pinned N27 payload")
    provenance = receipt["provenance"]
    _require(provenance.get("rigid_payload_sha256") == RIGID_SHA,
             "NHRIGID source rest differs from pinned input")
    _require(provenance.get("bodyparts_registration_sha256") == REGISTRATION_SHA,
             "registration differs from pinned input")
    _require(any(row.get("hierarchy") == "part_of" and row.get("sha256") == SOURCE_ARCHIVE_SHA
                 for row in provenance.get("bodyparts3d_archives", [])),
             "BodyParts3D archive differs from pinned input")
    ext = provenance.get("passive_brachiocephalic_vein_extension", {})
    _require(ext.get("additional_physical_mass_kg") == 0
             and ext.get("additional_force_or_state") is False,
             "passive bridge append added physical mass, force, or state")
    registration_path = Path(ext.get("verified_registration_path", ""))
    rigid_path = Path(ext.get("verified_rigid_payload_path", ""))
    _require(registration_path.is_file() and _sha_file(registration_path) == REGISTRATION_SHA
             and rigid_path.is_file() and _sha_file(rigid_path) == RIGID_SHA,
             "verified registration or NHRIGID source-rest bytes changed")
    registration = json.loads(registration_path.read_text())
    body = _validated_torso_rest_body(base_receipt, registration, rigid_path)
    recorded = ext.get("verified_rest_body_context", {})
    _require(recorded.get("validated_against_receipt_torso_rest") is True
             and recorded.get("source_body_id") == body["torso"]["source_body_id"] == 8
             and recorded.get("core_body_index") == body["torso"]["core_body_index"] == 20,
             "recorded torso context differs from the validated NHRIGID source rest")
    for key in ("default_com_position_world_m", "default_inertial_quaternion_world_xyzw"):
        _require(len(recorded.get(key, [])) == len(body["torso"][key])
                 and all(abs(float(a) - float(b)) <= 5e-9
                         for a, b in zip(recorded[key], body["torso"][key])),
                 "recorded torso rest transform differs from NHRIGID for " + key)
    base_raw = base_path.read_bytes()
    base_parts, base_header = _payload_sections(base_raw)
    _child_parts, child_header = _payload_sections(candidate_raw)
    from .resting_pleura_proxy import _parse_payload
    _, _, base_rows = _parse_payload(base_raw)
    _, _, child_rows = _parse_payload(candidate_raw)
    _require(base_header[2] == 523 and len(base_rows) == 523 and max(base_rows) == 524
             and child_header[2] == 525 and child_header[5:] == base_header[5:]
             and set(child_rows) == set(base_rows) | {525, 526},
             "candidate is not the two-row append on the pinned 523-row N27")
    from .resting_intercostal_mesh_arrangement import validate_receipt_composition
    intercostal = validate_receipt_composition(
        base_receipt_path=base_receipt_path,
        base_payload_sha256=base_receipt["payload"]["sha256"],
        receipt=receipt,
        candidate_rows=child_rows,
    )
    added_intercostal_vertices = sum(
        int(row["new_appended_vertex_count"])
        for row in receipt["provenance"]["local_intercostal_mesh_repair"]["rows"]
    )
    _require(child_header[3] == base_header[3] + 651 + added_intercostal_vertices,
             "NHA vertex delta is not exactly bridge append plus mapped intercostal children")
    _require(sum(len(row["faces"]) * 3 for row in child_rows.values()) == child_header[4]
             and sum(len(row["faces"]) * 3 for row in base_rows.values()) == base_header[4],
             "decoded NHA index counts differ from ABI5 headers")
    for stable_id in (525, 526):
        row = child_rows[stable_id]
        _require(int(row["body_index"]) == 20 and int(row["layer"]) == 2
                 and int(row["flags"]) == 0,
                 "appended passive bridge row changed NHA body/layer/flags")
    _require(ext.get("preserved_base_record_bytes_sha256") == _sha_bytes(base_parts[0])
             and ext.get("preserved_base_vertex_bytes_sha256") == _sha_bytes(base_parts[1])
             and ext.get("preserved_base_index_bytes_sha256") == _sha_bytes(base_parts[2]),
             "append receipt does not bind the original N27 source bytes")
    parent_bytes = {
        "receipt_sha256": N27_RECEIPT_SHA, "payload_sha256": N27_PAYLOAD_SHA,
        "records_sha256": _sha_bytes(base_parts[0]),
        "vertices_sha256": _sha_bytes(base_parts[1]),
        "indices_sha256": _sha_bytes(base_parts[2]),
        **intercostal,
        "global_buffer_offsets_may_be_rebased": True,
    }
    return {"base_receipt": base_receipt, "base_path": base_path, "base_raw": base_raw,
            "parent_bytes": parent_bytes}

def _validated_bridge_source_geometry(row: dict, bridge: dict, receipt: dict) -> tuple:
    """Bind a default bridge to 060 or the left bridge to the exact route110 transfer."""
    import numpy as np

    xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
    faces = np.ascontiguousarray(row["faces"], dtype="<u4")
    xyz_sha, faces_sha = _sha_bytes(xyz.tobytes()), _sha_bytes(faces.tobytes())
    if int(bridge["stable_id"]) == 525:
        from .resting_vessel_owner_blend import ROUTE110_PINS

        extension = receipt.get("provenance", {}).get(
            "passive_brachiocephalic_vein_extension", {})
        route = extension.get("route110_reference_field", {})
        source_map = receipt.get("provenance", {}).get("source_id_map", {})
        pointer = (source_map.get("525", {}).get("repair") or {}).get(
            "route110_reference_field", {})
        compiler = route.get("bridge_compiler_binding", {})
        _require(
            route.get("schema") == "numi.human.route110.registered-reference-row-transfer.v1"
            and route.get("candidate_npz_sha256") == ROUTE110_PINS["npz"]
            and route.get("candidate_report_sha256") == ROUTE110_PINS["report"]
            and route.get("candidate_runner_sha256") == ROUTE110_PINS["runner"]
            and route.get("target_preparation_checkpoint_sha256") == ROUTE110_PINS["checkpoint"]
            and route.get("native_or_gpu_run") is False,
            "left bridge route110 transfer is not the pinned offline source candidate",
        )
        _require(
            compiler.get("stable_id") == 525
            and compiler.get("source_member") == bridge["member"] == "FJ3482"
            and compiler.get("existing_compiler_before_xyz_sha256") == bridge["xyz"]
            and compiler.get("existing_compiler_faces_sha256") == bridge["faces"]
            and compiler.get("compiler_output_matches_route110_before_exact") is True,
            "left bridge route110 compiler ancestry does not bind the 060 source row",
        )
        _require(
            pointer.get("stable_id") == 525
            and pointer.get("source_member") == bridge["member"]
            and pointer.get("candidate_npz_sha256") == ROUTE110_PINS["npz"]
            and pointer.get("candidate_report_sha256") == ROUTE110_PINS["report"]
            and pointer.get("candidate_runner_sha256") == ROUTE110_PINS["runner"]
            and pointer.get("target_preparation_checkpoint_sha256") == ROUTE110_PINS["checkpoint"]
            and pointer.get("after_xyz_sha256") == xyz_sha
            and pointer.get("faces_sha256") == faces_sha == bridge["faces"]
            and pointer.get("xyz_changed") is True,
            "left bridge current geometry differs from exact route110 source transfer",
        )
        provenance = {
            "geometry_owner": "pinned route110 source transfer on existing FJ3482 material vertices",
            "baseline_060_xyz_sha256": bridge["xyz"],
            "current_xyz_sha256": xyz_sha,
            "faces_sha256": faces_sha,
            "route110_npz_sha256": ROUTE110_PINS["npz"],
            "route110_source_row_transfer_verified": True,
        }
    else:
        _require(xyz_sha == bridge["xyz"] and faces_sha == bridge["faces"],
                 "candidate source bridge geometry differs for " + bridge["member"])
        provenance = {
            "geometry_owner": "unchanged exact source-060 ABI5 compilation",
            "baseline_060_xyz_sha256": bridge["xyz"],
            "current_xyz_sha256": xyz_sha,
            "faces_sha256": faces_sha,
            "route110_source_row_transfer_verified": False,
        }
    return xyz, faces, provenance


def _route_data(rows: dict, receipt: dict, paths: dict[str, Path], np: Any) -> tuple:
    for key, path in paths.items():
        _require(path.is_file() and _sha_file(path) == ROUTE_HASHES[key],
                 "frozen collar input changed: " + key)
    ancestry = json.loads(paths["ancestry"].read_text())
    attribution = json.loads(paths["attribution"].read_text())
    replay = json.loads(paths["replay"].read_text())
    stationarity = json.loads(paths["stationarity"].read_text())
    _require(ancestry.get("current_nha_sha256") == N27_PAYLOAD_SHA,
             "route collars are not tied to pinned N27")
    candidate107 = attribution.get("candidate107_payload", {})
    _require(candidate107.get("sha256") == ancestry.get("route_candidate107_npz_sha256")
             and candidate107.get("contains_FJ3482") is True
             and candidate107.get("contains_FJ3583") is False,
             "corrected candidate107 left/right attribution differs from route report")
    _require(replay.get("status") == "PASS"
             and stationarity.get("status") == "PASS_SOURCE_PORT_SUPPORT_FIXED",
             "independent collar replay/stationarity did not pass")
    left_support_checks = [c for c in stationarity.get("checks", [])
                           if c.get("member") == "FJ3482"]
    _require(len(left_support_checks) == 3 and all(
        c.get("candidate_status") == "PRESENT_SOURCE_ONLY_NO_POSE_ADMISSION"
        and c.get("bridge_support_delta_bytes_all_zero") is True
        and c.get("target_support_delta_bytes_all_zero") is True
        for c in left_support_checks),
        "left bridge/target collar support is not stationary in candidate107")
    _require(receipt["provenance"].get("rigid_payload_sha256") == RIGID_SHA,
             "candidate body frame differs from route witness frame")
    member_reports = {x["member"]: x for x in ancestry.get("members", [])}
    dist = np.load(paths["distances"], allow_pickle=False)
    output = {}
    try:
        for bridge in BRIDGES:
            member = bridge["member"]
            report = member_reports.get(member)
            _require(report is not None and report.get("proposed_stable_id") == bridge["stable_id"],
                     "route stable ID changed for " + member)
            geo = report.get("candidate107_bridge_geometry", {})
            _require(geo.get("candidate_before_after_xyz_bitwise_equal") is True
                     and geo.get("candidate_source_face_order_bitwise_equal") is True,
                     "bridge geometry is not stationary in frozen route report")
            row = rows.get(bridge["stable_id"])
            _require(row is not None, "missing bridge row " + str(bridge["stable_id"]))
            xyz, faces_u32, geometry_provenance = _validated_bridge_source_geometry(
                row, bridge, receipt)
            faces = faces_u32.astype("<i4")
            identity = receipt["provenance"]["source_id_map"].get(str(bridge["stable_id"]), {})
            _require(identity.get("source_member") == member
                     and int(identity.get("body_index", -1)) == 20
                     and identity.get("vascular_compartment_id") == 4,
                     "source identity/body/CVSim compartment changed")
            _require(report.get("current_attachment_frame") ==
                     "Body20 local; exact retained NHRIGID default body20 pose",
                     "source body frame differs from collar report")
            support = {}
            for pid in (10, *bridge["ports"]):
                port = report.get("ports", {}).get(str(pid))
                _require(port is not None, "missing port " + str(pid))
                witness = port["bridge_witness"]["bridge_vertex_ids_for_each_pair"]
                ids = {int(v) for tri in witness for v in tri}
                _require(bool(ids) and min(ids) >= 0 and max(ids) < len(xyz),
                         "malformed bridge collar witness")
                key = bridge["dist"][str(pid)]
                _require(key in dist.files, "missing source geodesic array " + key)
                frozen_values = np.asarray(dist[key], dtype=np.float32)
                _require(frozen_values.shape == (len(xyz),) and np.isfinite(frozen_values).all()
                         and (frozen_values >= 0).all(), "malformed frozen collar geodesic values")
                zeros = set(map(int, np.flatnonzero(frozen_values == 0.0)))
                _require(zeros == ids, "frozen distance zero set differs from triangle witness " + key)
                support[pid] = ids
            svc_support = support[10]
            distal_support = set().union(*(support[p] for p in bridge["ports"]))
            _require(not (svc_support & distal_support), "SVC and distal collars overlap")
            from .resting_vessel_owner_blend import (
                _geodesic_distance,
                _mesh_components,
                _mesh_graph,
            )
            graph, _ = _mesh_graph(xyz, faces)
            components = _mesh_components(graph)
            source_component = next(
                (component for component in components if svc_support.intersection(component)),
                set())
            _require(len(components) == 1
                     and svc_support.issubset(source_component)
                     and distal_support.issubset(source_component),
                     "material collar witnesses do not share one connected source mesh")
            d_svc = _geodesic_distance(graph, svc_support, source_component)
            d_distal = np.minimum.reduce([
                _geodesic_distance(graph, support[pid], source_component)
                for pid in bridge["ports"]])
            _require(np.all(d_svc[np.asarray(sorted(svc_support), dtype=int)] == 0.0)
                     and np.all(d_distal[np.asarray(sorted(distal_support), dtype=int)] == 0.0),
                     "route-adjusted geodesic distances do not terminate at frozen collar vertices")
            svc_row = rows.get(10)
            _require(svc_row is not None, "missing SVC10 source geometry")
            svc_xyz = np.asarray(svc_row["vertices6"][:, :3], dtype=np.float64)
            svc_faces = np.asarray(svc_row["faces"], dtype=np.int64)
            collar_offsets = []
            for vertex_id in sorted(svc_support):
                point = np.asarray(xyz[vertex_id], dtype=np.float64)
                best_d2 = math.inf
                for face in svc_faces:
                    tri = svc_xyz[np.asarray(face, dtype=np.int64)]
                    closest, _ = _closest_point_on_triangle(point, tri, np)
                    delta = point - np.asarray(closest, dtype=np.float64)
                    best_d2 = min(best_d2, float(np.dot(delta, delta)))
                collar_offsets.append(math.sqrt(best_d2))
            max_collar_offset = max(collar_offsets)
            route_extent = float(np.max(d_svc))
            bound = route_extent + max_collar_offset
            _require(math.isfinite(bound) and bound > 0.0, "invalid source-route bound")
            output[member] = {
                "report": report, "d_svc_m": d_svc, "d_distal_m": d_distal,
                "svc_support": sorted(svc_support), "distal_support": sorted(distal_support),
                "geodesic_metric": "recomputed from candidate Float32 material coordinates; frozen ancestry sets seed collars",
                "geometry_provenance": geometry_provenance,
                "projection_bound_m": bound,
                "route_geodesic_extent_m": route_extent,
                "max_svc_collar_to_svc_surface_m": max_collar_offset,
                "bound_source": (
                    "maximum source-mesh geodesic distance from the exact SVC10 collar witness set "
                    "plus the maximum Euclidean distance of those witness vertices to the SVC10 "
                    "surface; this is computed before bridge projection and bounds every route path "
                    "plus its measured collar offset"
                ),
            }
        return ancestry, attribution, output
    finally:
        dist.close()

def _compile_records(bridge_xyz: Any, d_svc: Any, d_distal: Any, svc_xyz: Any,
                     svc_faces: Any, svc_records: Any, projection_bound_m: float,
                     np: Any) -> tuple:
    xyz = np.asarray(bridge_xyz, dtype=np.float64)
    d_svc = np.asarray(d_svc, dtype=np.float64)
    d_distal = np.asarray(d_distal, dtype=np.float64)
    svc_xyz = np.asarray(svc_xyz, dtype=np.float64)
    faces = np.asarray(svc_faces, dtype=np.int64)
    coeff = np.asarray(svc_records, dtype="<f4").reshape(-1, 7, 4)
    _require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0,
             "malformed bridge coordinates")
    _require(d_svc.shape == d_distal.shape == (len(xyz),)
             and np.isfinite(d_svc).all() and np.isfinite(d_distal).all()
             and (d_svc >= 0).all() and (d_distal >= 0).all(),
             "malformed collar distance arrays")
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
             and int(faces.min()) >= 0 and int(faces.max()) < len(svc_xyz)
             and coeff.shape == (len(svc_xyz), 7, 4)
             and np.isfinite(coeff).all(), "malformed SVC source map")
    _require(np.all(coeff.view("<u4")[:, :, 3] == 0),
             "SVC map float4 W components must be canonical positive zero")
    _require(math.isfinite(projection_bound_m) and projection_bound_m > 0,
             "invalid projection distance bound")
    denom = d_svc + d_distal
    _require(np.all(denom > 0) and not np.any((d_svc == 0) & (d_distal == 0)),
             "SVC and distal collar endpoints overlap")
    param = d_distal / denom
    fade = quintic_smoothstep(param, np)
    distal_zero = d_distal == 0
    exact = {}
    for i, point in enumerate(np.asarray(svc_xyz, dtype="<f4")):
        key = point.tobytes()
        prior = exact.get(key)
        if prior is None:
            exact[key] = i
        elif coeff[prior].tobytes() != coeff[i].tobytes():
            raise ValueError("brachiocephalic common field: duplicate SVC coordinates have unequal coefficients")
    out = np.zeros((len(xyz), 7, 4), dtype="<f4")
    ancestry = []
    for vi, point in enumerate(xyz):
        best_d2, best_face, best_vids, best_bary = math.inf, -1, None, None
        for fi, face in enumerate(faces):
            vids = tuple(map(int, face))
            tri = svc_xyz[np.asarray(vids, dtype=int)]
            closest, bary = _closest_point_on_triangle(point, tri, np)
            delta = point - np.asarray(closest, dtype=np.float64)
            d2 = float(np.dot(delta, delta))
            if d2 < best_d2:
                best_d2, best_face, best_vids = d2, fi, vids
                best_bary = np.asarray(bary, dtype=np.float64)
        distance = math.sqrt(best_d2)
        _require(distance <= projection_bound_m,
                 f"nearest projection {distance:.9g} m exceeds fixed source-route bound {projection_bound_m:.9g} m")
        exact_id = exact.get(np.asarray(point, dtype="<f4").tobytes())
        if exact_id is not None:
            _require(float(fade[vi]) == 1.0,
                     "exact shared SVC point would not retain coefficient bytes")
            out[vi] = coeff[exact_id]
        elif distal_zero[vi]:
            out[vi].fill(np.float32(0.0))
        else:
            values = coeff[np.asarray(best_vids, dtype=int)].astype(np.float64)
            out[vi] = np.asarray(np.einsum("i,ijk->jk", best_bary, values) * float(fade[vi]),
                                 dtype="<f4")
        _require(np.isfinite(out[vi]).all() and
                 (not distal_zero[vi] or np.all(out[vi] == 0.0)),
                 "interpolation is nonfinite or fails distal zero-field collar")
        _require(np.all(out[vi].view("<u4").reshape(7, 4)[:, 3] == 0),
                 "compiled float4 W components must be canonical positive zero")
        ancestry.append({
            "bridge_local_vertex": vi, "svc_local_face": best_face,
            "svc_local_vertices": list(best_vids), "svc_map_record_offsets": list(best_vids),
            "barycentric_weights": [float(x) for x in best_bary],
            "nearest_projection_distance_m": distance,
            "distance_to_svc_collar_m": float(d_svc[vi]),
            "distance_to_distal_collar_m": float(d_distal[vi]),
            "smoothstep_parameter": float(param[vi]), "coefficient_scale": float(fade[vi]),
            "exact_source_coordinate_copy": exact_id is not None,
            "copied_svc_local_vertex": None if exact_id is None else int(exact_id),
            "distal_zero_field_vertex": bool(distal_zero[vi]),
        })
    return out.reshape(len(xyz), 28), ancestry

def prepare_brachiocephalic_common_field(
        *, candidate_receipt_path: Path, output_dir: Path,
        collar_ancestry_path: Path, collar_attribution_path: Path,
        collar_replay_path: Path, collar_stationarity_path: Path,
        port_distances_path: Path) -> dict:
    """Append two passive coefficient ranges to the existing ABI5 common map."""
    import numpy as np

    candidate_receipt_path = Path(candidate_receipt_path).resolve()
    output_dir = Path(output_dir).resolve()
    _require(not output_dir.exists(), "output directory already exists")
    receipt_raw = candidate_receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    payload = receipt["payload"]
    payload_path = Path(payload["path"])
    if not payload_path.is_absolute():
        payload_path = candidate_receipt_path.parent / payload_path
    _require(payload_path.is_file() and _sha_file(payload_path) == payload["sha256"],
             "candidate receipt and anatomy payload differ")
    anatomy_raw = payload_path.read_bytes()
    ext = receipt["provenance"].get("passive_brachiocephalic_vein_extension")
    _require(isinstance(ext, dict), "candidate lacks existing ABI5 source append")
    base_receipt_path = Path(ext["base_receipt"]).resolve()
    parent = _validate_n27_parent(receipt, anatomy_raw, base_receipt_path)

    quotient = ext.get("source_quotient_report", {})
    _require(_sha_file(Path(quotient.get("path", ""))) == quotient.get("sha256") == QUOTIENT_SHA
             and _sha_file(Path(quotient.get("npz_path", ""))) == quotient.get("npz_sha256") == QUOTIENT_NPZ_SHA,
             "pinned 060 quotient inputs differ")
    _, _, rows = _parse_payload(anatomy_raw)
    _, _, base_rows = _parse_payload(parent["base_raw"])
    _require(set(rows) == set(map(int, receipt["provenance"]["source_id_map"]))
             and set(base_rows).issubset(rows) and max(base_rows) == 524 and max(rows) == 526,
             "decoded rows, source map, or max+1 IDs differ")
    _require(all(b["stable_id"] not in base_rows for b in BRIDGES),
             "vein IDs do not append after current N27 max")
    route_paths = {
        "ancestry": Path(collar_ancestry_path), "attribution": Path(collar_attribution_path),
        "replay": Path(collar_replay_path), "stationarity": Path(collar_stationarity_path),
        "distances": Path(port_distances_path)}
    ancestry, attribution, routes = _route_data(rows, receipt, route_paths, np)

    common = receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    base_common = parent["base_receipt"].get("provenance", {}).get(
        "cardiac_geometry_binding", {}).get("common_field")
    _validate_common_field_descriptor(common, base_common)
    _require(common.get("schema") == "numi.human.cardiac_common_field.v1"
             and common.get("anatomy_payload_sha256") in {N27_PAYLOAD_SHA, payload["sha256"]},
             "candidate does not bind the existing common-field map")
    _require(int(common["map"]["record_count"]) == 620718
             and int(common["map"]["record_stride_bytes"]) == 112,
             "common map is not the existing 620718-row 112-byte prefix")
    _require(not any(int(r.get("stable_id", -1)) in {525, 526}
                     for r in common["passive_attachment_ranges"]),
             "one or both vein ranges already exist in the map")
    sidecars = {key: _resolve_sidecar(candidate_receipt_path, base_receipt_path, common[key])
                for key in ("map", "polynomials", "domain_boxes")}
    base_sidecars = {key: _resolve_sidecar(base_receipt_path, base_receipt_path, base_common[key])
                     for key in ("map", "polynomials", "domain_boxes")}
    sidecar_bytes = {}
    for key in ("map", "polynomials", "domain_boxes"):
        expected_sha = base_common[key]["sha256"]
        candidate_bytes = sidecars[key].read_bytes()
        base_bytes = base_sidecars[key].read_bytes()
        _require(_sha_bytes(base_bytes) == expected_sha
                 and _sha_bytes(candidate_bytes) == expected_sha
                 and candidate_bytes == base_bytes,
                 "candidate common-field sidecar differs from pinned N27 bytes: " + key)
        sidecar_bytes[key] = candidate_bytes
    old_map = sidecar_bytes["map"]
    _require(len(old_map) == 620718 * 112 and _sha_bytes(old_map) == common["map"]["sha256"],
             "input map bytes differ from its pinned descriptor")
    old_map_words = np.frombuffer(old_map, dtype="<u4").reshape(-1, 28)
    _require(np.all(old_map_words[:, 3::4] == 0),
             "existing common-map float4 W components must be canonical positive zero")
    polynomials = sidecar_bytes["polynomials"]
    domain_boxes = sidecar_bytes["domain_boxes"]

    svc = rows.get(10)
    svc_range = next((r for r in common["passive_attachment_ranges"]
                      if int(r.get("stable_id", -1)) == 10), None)
    _require(svc is not None and svc_range is not None
             and int(svc_range["vertex_count"]) == len(svc["vertices6"])
             and int(svc_range["body_index"]) == int(svc["body_index"])
             and int(svc_range["semantic"]) == 51011,
             "SVC10 map range differs from existing NHA surface")
    start, count = int(svc_range["first_vertex"]), int(svc_range["vertex_count"])
    map_rows = np.frombuffer(old_map, dtype="<f4").reshape(-1, 28)
    svc_coeff = map_rows[start:start+count]
    _require(len(svc_coeff) == len(svc["vertices6"]), "SVC10 coefficient count differs")
    ranges = []
    bridge_reports = {}
    tail = []
    next_offset = int(common["map"]["record_count"])
    for b in BRIDGES:
        row, route = rows[b["stable_id"]], routes[b["member"]]
        records, ancestry_rows = _compile_records(
            row["vertices6"][:, :3], route["d_svc_m"], route["d_distal_m"],
            svc["vertices6"][:, :3], svc["faces"], svc_coeff,
            route["projection_bound_m"], np)
        tail.append(records.tobytes(order="C"))
        ranges.append({"stable_id": b["stable_id"], "first_vertex": next_offset,
                       "vertex_count": len(row["vertices6"]), "body_index": 20,
                       "semantic": 51011, "vascular_compartment_id": 4})
        bridge_reports[b["member"]] = {
            "stable_id": b["stable_id"],
            "source_local_xyz_sha256": _sha_bytes(np.ascontiguousarray(
                row["vertices6"][:, :3], dtype="<f4").tobytes()),
            "baseline_060_xyz_sha256": b["xyz"],
            "source_faces_sha256": _sha_bytes(np.ascontiguousarray(
                row["faces"], dtype="<u4").tobytes()),
            "geometry_provenance": route["geometry_provenance"],
            "geodesic_metric": route["geodesic_metric"],
            "vertex_count": len(row["vertices6"]),
            "triangle_count": len(row["faces"]), "distal_port_ids": list(b["ports"]),
            "svc_port_support_vertices": route["svc_support"],
            "distal_port_support_vertices": route["distal_support"],
            "projection_distance_bound_m": route["projection_bound_m"],
            "route_geodesic_extent_m": route["route_geodesic_extent_m"],
            "max_svc_collar_to_svc_surface_m": route["max_svc_collar_to_svc_surface_m"],
            "projection_bound_source": route["bound_source"],
            "max_actual_projection_distance_m": max(x["nearest_projection_distance_m"] for x in ancestry_rows),
            "exact_svc_source_coordinate_copies": sum(x["exact_source_coordinate_copy"] for x in ancestry_rows),
            "distal_zero_field_vertices": sum(x["distal_zero_field_vertex"] for x in ancestry_rows),
            "vertex_ancestry": ancestry_rows}
        common["passive_attachment_ranges"].append({
            "body_index": 20, "first_vertex": next_offset, "semantic": 51011,
            "stable_id": b["stable_id"], "vertex_count": len(row["vertices6"])})
        next_offset += len(row["vertices6"])

    new_map = old_map + b"".join(tail)
    _require(new_map[:len(old_map)] == old_map and len(new_map) == next_offset * 112,
             "map append changed its prefix or range size")
    report = {
        "format": "numi.human.brachiocephalic-common-field-source-preparation",
        "method": "Project each bridge vertex to the nearest SVC10 triangle with the established deterministic closest-triangle/barycentric routine, interpolate the seven existing float4 coefficients, and apply quintic C2 fade to the existing zero field at IJV/subclavian collar support.",
        "source_code_sha256": _source_code_pins(),
        "source_preparation_invocation": {
            "entrypoint": "numilab_human.resting_brachiocephalic_common_field.prepare_brachiocephalic_common_field",
            "geometry_predicates_run": False,
            "native_or_gpu_run": False,
        },
        "input_candidate_receipt_path": str(candidate_receipt_path),
        "input_candidate_receipt_sha256": _sha_bytes(receipt_raw),
        "input_candidate_payload_sha256": payload["sha256"],
        "pinned_n27_base_receipt_path": str(base_receipt_path),
        "pinned_n27_base_receipt_sha256": N27_RECEIPT_SHA,
        "pinned_n27_base_payload_sha256": N27_PAYLOAD_SHA,
        "original_523_row_byte_preservation": parent["parent_bytes"],
        "source_060_quotient_report_sha256": QUOTIENT_SHA,
        "source_060_quotient_npz_sha256": QUOTIENT_NPZ_SHA,
        "route_collar_ancestry_sha256": ROUTE_HASHES["ancestry"],
        "route_collar_attribution_sha256": ROUTE_HASHES["attribution"],
        "route_collar_replay_sha256": ROUTE_HASHES["replay"],
        "route_collar_stationarity_sha256": ROUTE_HASHES["stationarity"],
        "route_distance_npz_sha256": ROUTE_HASHES["distances"],
        "route_ancestry_status": ancestry.get("status"),
        "right_side_attribution_scope": attribution.get("scope"),
        "input_common_map_sha256": _sha_bytes(old_map),
        "input_common_map_record_count": 620718,
        "input_common_map_record_stride_bytes": 112,
        "input_common_polynomials_sha256": _sha_bytes(polynomials),
        "input_common_domain_boxes_sha256": _sha_bytes(domain_boxes),
        "coordinate_order": list(common["coordinate_order"]),
        "volume_polynomials_and_domain_boxes_byte_preserved": True,
        "map_prefix_byte_exact": new_map[:len(old_map)] == old_map,
        "appended_record_count": next_offset - 620718,
        "output_map_sha256": _sha_bytes(new_map),
        "ranges": ranges,
        "bridges": bridge_reports,
        "anatomy_payload_modified": False,
        "added_physical_mass_kg": 0,
        "additional_physical_state_or_force": False,
        "existing_cvsim_compartment_id": 4,
        "native_admission": False, "anatomical_admission": False, "all_pose_admission": False,
        "acceptance_gates_remaining": [
            "connected lumen topology and calibre at all three source collars",
            "all-target and nonintersection against adjacent route surfaces",
            "accepted-pose common-field displacement and junction continuity",
            "native common-map loader/runtime qualification"],
    }

    output_dir.mkdir(parents=True)
    for source in sidecars.values():
        shutil.copyfile(source, output_dir / source.name)
    map_path = output_dir / sidecars["map"].name
    map_path.write_bytes(new_map)
    common["anatomy_payload_sha256"] = payload["sha256"]
    common["map"].update(path=map_path.name, sha256=_sha_bytes(new_map),
                         record_count=next_offset, record_stride_bytes=112)
    report_path = output_dir / "brachiocephalic-common-field-preparation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    prepared_receipt = copy.deepcopy(receipt)
    prepared_receipt["provenance"]["cardiac_geometry_binding"]["common_field"] = common
    prepared_receipt["provenance"]["passive_brachiocephalic_common_field_preparation"] = {
        "report_path": str(report_path), "report_sha256": _sha_file(report_path),
        "input_receipt_path": str(candidate_receipt_path),
        "input_receipt_sha256": _sha_bytes(receipt_raw),
        "input_payload_sha256": payload["sha256"],
        "appended_map_sha256": _sha_bytes(new_map), "appended_range_stable_ids": [525, 526],
        "additional_physical_mass_kg": 0, "additional_physical_state_or_force": False,
        "native_admission": False, "anatomical_admission": False, "all_pose_admission": False}
    receipt_path = output_dir / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(prepared_receipt, indent=2, sort_keys=True) + "\n")
    manifest_path = candidate_receipt_path.parent / "resting-anatomy-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        manifest["receipt"] = {"path": str(receipt_path), "sha256": _sha_file(receipt_path)}
        (output_dir / "resting-anatomy-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": "SOURCE_PREPARATION_ONLY_NOT_ADMISSION",
            "output_dir": str(output_dir), "report_path": str(report_path),
            "report_sha256": _sha_file(report_path), "receipt_path": str(receipt_path),
            "receipt_sha256": _sha_file(receipt_path), "input_payload_sha256": payload["sha256"],
            "map_sha256": _sha_file(map_path), "map_record_count": next_offset,
            "map_prefix_sha256": _sha_bytes(old_map),
            "polynomials_sha256": _sha_bytes(polynomials),
            "domain_boxes_sha256": _sha_bytes(domain_boxes)}

