"""Admit one repaired passive taenia through the existing NHANAT1 asset path.

This source-preparation step adds no runtime state, mass, force, or physiology.
It requires exact source-bound audits and explicit colon-interface witnesses.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .resting_anatomy_interface_patch import HEADER, RECORD, signed_volume
from .resting_passive_interfaces import _surface
from .resting_pleura_proxy import _parse_payload, _record_content_bytes

MEMBERS = {454: "FJ2566", 455: "FJ2567", 456: "FJ2568", 457: "FJ2569",
           458: "FJ2570", 459: "FJ2571", 460: "FJ2572"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError("passive taenia reference: " + message)


def _complete_zero(row):
    return (type(row.get("count")) is int and row["count"] == 0
            and row.get("audit_complete", True) and not row.get("count_is_lower_bound", False))


def build_candidate(base_payload: Path, base_receipt: Path, candidate_path: Path,
                    audit_path: Path, shape_path: Path, interface_path: Path,
                    output: Path) -> dict:
    _require(not output.exists(), "retain existing output; choose a new directory")
    raw = base_payload.read_bytes(); source_sha = hashlib.sha256(raw).hexdigest()
    header, records, rows = _parse_payload(raw)
    receipt = json.loads(base_receipt.read_text())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == source_sha
             and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_sha,
             "receipt does not identify the current payload")
    passive = set(receipt["functional_bindings"]["passive_viscera_geometry_binding"]["stable_ids"])
    source_map = receipt["provenance"]["source_id_map"]
    _require(set(MEMBERS).issubset(passive) and all(source_map.get(str(s), {}).get("source_member") == member
             for s, member in MEMBERS.items()), "colon source identities changed")
    _require(rows[457]["body_index"] == 20 and rows[457]["layer"] == 1,
             "taenia lacks the shared passive torso source frame")
    cardiac = receipt["provenance"]["cardiac_geometry_binding"]
    wall = cardiac["ventricular_wall_binding"]
    _require(cardiac.get("output_anatomy_payload_sha256") == source_sha
             and wall.get("output_anatomy_payload_sha256") == source_sha,
             "existing cardiac bindings are stale")
    _require("local_coefficient_refinement" not in wall.get("wall_map", {}),
             "attach source-bound cardiac correction after passive asset preparation")
    digest = _sha(candidate_path)
    audit = json.loads(audit_path.read_text()); shape = json.loads(shape_path.read_text())
    interfaces = json.loads(interface_path.read_text())
    _require(audit.get("source_payload_sha256") == source_sha
             and audit.get("candidate_sha256") == digest and _complete_zero(audit.get("self", {})),
             "candidate lacks a current complete exact self audit")
    ledger = {str(s): hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in passive}
    _require(audit.get("source_record_sha256") == ledger, "passive-neighbor audit sources changed")
    _require(shape.get("source_sha256") == audit.get("source_weld_sha256")
             and audit.get("source_weld_record_sha256") == ledger["457"],
             "shape reference does not bind the original taenia record")
    _require(shape.get("candidate_sha256") == digest and shape.get("sampled_distance_bound_m") == .0005
             and shape.get("relative_volume_bound") == .04,
             "shape evidence is absent or uses different bounds")
    _require(all(0 <= shape[k]["max_m"] <= .0005 for k in ("source_to_candidate", "candidate_to_source")),
             "candidate exceeds the sampled shape bound")
    _require(interfaces.get("candidate_sha256") == digest
             and interfaces.get("audit_sha256") == audit.get("exact_audit_sha256")
             and interfaces.get("predicate_sha256") == audit.get("predicate_sha256"),
             "interface witnesses do not bind the exact candidate audit")
    by_id = {}
    for row in audit.get("pairs", []):
        ids = row.get("ids", [])
        _require(len(ids) == 2 and ids[0] == 457 and ids[1] not in by_id, "invalid or repeated neighbor pair")
        by_id[ids[1]] = row
    _require(set(by_id) == passive - {457}, "audit does not cover every passive neighbor")
    classified = {row["stable_ids"][1]: row for row in interfaces.get("interfaces", [])}
    _require(len(classified) == len(interfaces.get("interfaces", [])), "repeated interface classification")
    positive = set()
    for sid, pair in by_id.items():
        _require(pair.get("audit_complete", True) and not pair.get("invalid_geometry")
                 and not pair.get("count_is_lower_bound", False) and type(pair.get("count")) is int,
                 "neighbor audit is incomplete or invalid")
        if _complete_zero(pair):
            continue
        positive.add(sid); item = classified.get(sid, {})
        _require(sid in MEMBERS and sid != 457 and pair["count"] > 0
                 and pair["count"] == len(pair.get("triangle_pairs", []))
                 and item.get("source_triangle_pairs") == pair["triangle_pairs"]
                 and item.get("source_members") == [MEMBERS[457], MEMBERS[sid]],
                 "crossing lacks its exact source-component interface witnesses")
        if sid in (456, 458):
            span = item.get("distance_from_caudal_ascending_extent_m", [])
            _require(item.get("interface") == "taenia_convergence_at_caudal_ascending_colon"
                     and len(span) == 2 and 0 <= span[0] <= span[1] <= .020,
                     "band crossing lies outside its localized source convergence")
        elif sid == 459:
            span = item.get("distance_from_cranial_rectal_extent_m", [])
            _require(item.get("interface") == "longitudinal_muscle_continuation_at_rectal_entry"
                     and len(span) == 2 and 0 <= span[0] <= span[1] <= .005,
                     "rectal crossing lies outside the source entry")
        else:
            _require(item.get("interface") == "longitudinal_muscle_component_of_colon_wall",
                     "colon wall ownership is not explicit")
    _require(set(classified) == positive, "interface list includes unobserved or missing crossings")
    with np.load(candidate_path, allow_pickle=False) as data:
        vertices, faces, quality = _surface(data["vertices"], data["faces"])
    old_volume = signed_volume(rows[457]["vertices6"][:, :3].astype(float), rows[457]["faces"])
    _require(old_volume > 0 and abs(quality["volume_m3"] / old_volume - 1) <= .04,
             "candidate exceeds the passive volume bound")
    _require(abs(shape["source_volume_m3"] - old_volume) <= 1e-12
             and abs(shape["candidate_volume_m3"] - quality["volume_m3"]) <= 1e-12,
             "shape evidence volumes differ from actual geometry")
    updated = copy.deepcopy(rows); updated[457]["vertices6"] = vertices; updated[457]["faces"] = faces
    chunks, vv, ff, nv, ni = [], [], [], 0, 0
    for record in records:
        sid = record[5]; row = updated[sid]; v, f = row["vertices6"], row["faces"]
        chunks.append(RECORD.pack(row["body_index"], nv, len(v), ni, f.size, sid, row["layer"], row["flags"]))
        vv.append(np.asarray(v, dtype="<f4").tobytes()); ff.append((f.reshape(-1)+nv).astype("<u4").tobytes())
        nv += len(v); ni += f.size
    result = HEADER.pack(header[0], header[1], len(records), nv, ni, header[5], header[6]) + b"".join(chunks+vv+ff)
    _, _, verified = _parse_payload(result)
    _require(all(_record_content_bytes(rows[s]) == _record_content_bytes(verified[s]) for s in rows if s != 457),
             "another anatomical surface changed")
    output_sha = hashlib.sha256(result).hexdigest()
    detail = {"interpretation": "inferred_passive_colon_muscle_reference_surface_not_measured_subject_geometry",
              "compiler_source_sha256": _sha(Path(__file__)), "base_payload_sha256": source_sha,
              "base_receipt_sha256": _sha(base_receipt), "candidate_sha256": digest,
              "audit_sha256": _sha(audit_path), "shape_sha256": _sha(shape_path), "interfaces_sha256": _sha(interface_path),
              "shape": shape, "quality": quality, "source_interfaces": interfaces,
              "additional_physical_mass_kg": 0, "additional_physiological_state": False,
              "all_other_record_geometry_bytes_preserved": True,
              "qualification": "Exact source self and non-colon clearance passed; source colon interfaces are explicitly localized. Accepted native breathing-cycle checks remain required."}
    receipt["provenance"]["passive_taenia_reference"] = detail
    source_map["457"]["reference_geometry_status"] = detail["interpretation"]
    receipt["payload"].update(path=str(output/"resting-thorax.nhanatomy"), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac["output_anatomy_payload_sha256"] = output_sha; wall["output_anatomy_payload_sha256"] = output_sha
    receipt["qualification"]["passive_taenia_reference"] = detail["qualification"]
    output.mkdir(parents=True); (output/"resting-thorax.nhanatomy").write_bytes(result)
    receipt_path = output/"resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"], "qualification": receipt["qualification"],
                "source_surfaces": source_map, "mass_geometry_accounting": receipt["mass_geometry_accounting"]}
    (output/"resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


WALL_REGION_PARENTS = (454, 455, 460)


def face_coordinate_sha256(row):
    """Bind ordered Float32 face corners, independent of render-normal seams."""
    corners = row["vertices6"][row["faces"], :3]
    return hashlib.sha256(np.asarray(corners, dtype="<f4").tobytes()).hexdigest()


def prepare_wall_region(rows: dict, source_map: dict, candidate_path: Path,
                        derivation_path: Path, *, stable_id: int = 456):
    """Check an inferred taenia-libera annotation on existing colon triangles.

    This is an open, nonadditive material-region representation. It does not
    assert a closed muscle solid, preserve the old invalid shell's volume, or
    exempt any parent organ from its own intersection/volume checks.
    """
    from .resting_anatomy_interface_patch import normals, topology_report
    from . import cardiac_cavity_intersections as exact

    _require(stable_id in (456, 457), "only the two source-registered taenia members are supported")
    detail = json.loads(derivation_path.read_text())
    digest = _sha(candidate_path)
    _require(detail.get("representation") == "passive_colon_wall_surface_region_v1",
             "unsupported wall-region representation")
    _require(detail.get("stable_id") == stable_id and detail.get("candidate_sha256") == digest,
             "wall region has a different candidate or anatomical identity")
    required = (stable_id, *WALL_REGION_PARENTS)
    _require(all(s in rows and source_map.get(str(s), {}).get("source_member") == MEMBERS[s]
                 for s in required), "wall-region source component identities changed")
    _require(all(rows[s]["body_index"] == 20 and rows[s]["layer"] == 1 for s in required),
             "wall region and colon do not share the passive torso frame")
    source_record = hashlib.sha256(_record_content_bytes(rows[stable_id])).hexdigest()
    _require(detail.get("source_record_sha256") == source_record,
             "wall-region derivation refers to another source muscle")
    parent_hashes = {str(s): face_coordinate_sha256(rows[s]) for s in WALL_REGION_PARENTS}
    _require(detail.get("parent_face_coordinate_sha256") == parent_hashes,
             "wall-region parent colon geometry changed")
    _require(detail.get("rectal_scope") == "excluded_no_taenia_band",
             "wall-region derivation must distinguish rectal longitudinal muscle")
    shape = detail.get("sampled_shape", {})
    _require(shape.get("distance_is_sampled_not_hausdorff_bound") is True,
             "wall-region shape evidence overstates its sampled scope")
    for name in ("source_to_region", "region_to_source"):
        sample = shape.get(name, {})
        _require(type(sample.get("sample_count")) is int and sample["sample_count"] > 0
                 and isinstance(sample.get("maximum_m"), (int, float))
                 and np.isfinite(sample["maximum_m"]) and sample["maximum_m"] >= 0,
                 "wall-region shape sampling is absent or invalid")
    # These are explicit reference-asset limits, not physiological tolerances.
    if stable_id == 456:
        source_to_limit, region_to_limit, selection_limit = .006, .005, .005
    else:
        # FJ2569's registered shell is substantially larger than its host patch.
        # The coarse derived region is anchored to copied colon triangles and
        # uses explicit sampled (not Hausdorff) limits from this source pairing.
        source_to_limit, region_to_limit, selection_limit = .004, .015, .002
    _require(shape["source_to_region"]["maximum_m"] <= source_to_limit
             and shape["region_to_source"]["maximum_m"] <= region_to_limit
             and shape.get("face_selection_sampled_distance_limit_m") == selection_limit,
             "wall region exceeds the declared coarse reference extent")
    _require(isinstance(detail.get("sensitivity"), list) and len(detail["sensitivity"]) >= 2
             and isinstance(detail.get("references"), list) and detail["references"],
             "wall region lacks sensitivity or anatomical attribution")

    with np.load(candidate_path, allow_pickle=False) as candidate:
        v = candidate["vertices"]
        f = candidate["faces"]
        owners = candidate["colon_owner"]
        parent_faces = candidate["colon_face"]
    _require(v.dtype.kind == "f" and v.dtype.itemsize == 4
             and v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all(),
             "wall region must use finite Float32 positions")
    _require(f.ndim == 2 and f.shape[1] == 3 and len(f) > 0
             and f.dtype.kind in "iu" and f.min() >= 0 and f.max() < len(v),
             "invalid wall-region topology")
    _require(owners.shape == parent_faces.shape == (len(f),)
             and owners.dtype.kind in "iu" and parent_faces.dtype.kind in "iu"
             and set(map(int, owners)) == set(WALL_REGION_PARENTS),
             "wall-region parent-face inventory is incomplete or invalid")
    _require(len(set(zip(map(int, owners), map(int, parent_faces)))) == len(f),
             "wall region repeats a parent face")
    for sid in WALL_REGION_PARENTS:
        mask = owners == sid
        ids = parent_faces[mask]
        parent = rows[sid]
        _require(ids.min() >= 0 and ids.max() < len(parent["faces"]),
                 "wall-region parent face is out of range")
        actual = np.asarray(v[f[mask]], dtype="<f4").tobytes()
        expected = np.asarray(parent["vertices6"][parent["faces"][ids], :3],
                              dtype="<f4").tobytes()
        _require(actual == expected,
                 "wall-region face is not an identically oriented parent triangle")

    # Weld coordinate-identical normal seams before the exact embeddedness test.
    # Float32 -> common 2^149 lattice is lossless, including subnormal values.
    welded, inverse = np.unique(v[f].reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    lattice = []
    for point in welded:
        row = []
        for x in point:
            n, denominator = float(x).as_integer_ratio()
            row.append(n * ((1 << 149) // denominator))
        lattice.append(tuple(row))
    records = exact._records(lattice, faces.tolist())
    self_audit = exact._audit_pair(records, records, same_surface=True)
    _require(self_audit["count"] == 0, "wall region has an exact self intersection")
    topology = topology_report(faces)
    topology.pop("boundary_edges")
    _require(topology["nonmanifold_edge_count"] == 0
             and topology["orientation_error_edge_count"] == 0,
             "wall region has invalid shared edges")
    p = welded[faces].astype(np.float64)
    area = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1) / 2
    n = normals(welded.astype(np.float64), faces)
    vertices6 = np.column_stack((welded, n)).astype("<f4")
    provenance = copy.deepcopy(detail)
    provenance.update({
        "compiler_source_sha256": _sha(Path(__file__)),
        "derivation_sha256": _sha(derivation_path),
        "stable_id": stable_id,
        "source_member": MEMBERS[stable_id],
        "parameter_status": detail.get("parameter_status", "inferred_reference_region_not_measured_subject_geometry"),
        "exact_self": self_audit,
        "predicate_sha256": _sha(Path(exact.__file__)),
        "parent_triangles_preserved_bitwise": True,
        "area_m2": float(area.sum()), "topology": topology,
        "independent_volume_m3": None, "additional_physical_mass_kg": 0,
        "additional_physiological_state": False,
        "boundary_interpretation": "Region extent on intact parent colon, not holes in an organ. Disconnected face sets and point-contacting boundaries are annotations, not a manifold muscle solid.",
        "qualification": (
            "Static region embeddedness and exact parent-face ownership checked. "
            "Parent-organ neighbor and native breathing-cycle checks remain required. "
            "Sampled source distances are not a continuous shape certificate."),
    })
    return vertices6, faces, provenance


def build_wall_region_candidate(base_payload: Path, base_receipt: Path,
                                candidate_path: Path, derivation_path: Path,
                                output: Path, *, stable_id: int = 456) -> dict:
    """Replace one approved taenia row with its nonadditive host region."""
    _require(not output.exists(), "retain existing output; choose a new directory")
    raw = base_payload.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    header, records, rows = _parse_payload(raw)
    receipt = json.loads(base_receipt.read_text())
    _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == source_sha
             and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_sha,
             "receipt does not identify the current payload")
    binding = receipt["functional_bindings"].get("passive_viscera_geometry_binding", {})
    _require(set((stable_id, *WALL_REGION_PARENTS)).issubset(binding.get("stable_ids", [])),
             "wall region lacks the existing passive motion owner")
    cardiac = receipt["provenance"]["cardiac_geometry_binding"]
    wall = cardiac["ventricular_wall_binding"]
    _require(cardiac.get("output_anatomy_payload_sha256") == source_sha
             and wall.get("output_anatomy_payload_sha256") == source_sha,
             "existing cardiac bindings are stale")
    _require("local_coefficient_refinement" not in wall.get("wall_map", {}),
             "attach source-bound cardiac correction after passive asset preparation")
    source_map = receipt["provenance"]["source_id_map"]
    vertices, faces, detail = prepare_wall_region(
        rows, source_map, candidate_path, derivation_path, stable_id=stable_id)
    updated = copy.deepcopy(rows)
    updated[stable_id]["vertices6"], updated[stable_id]["faces"] = vertices, faces
    chunks, vv, ff, nv, ni = [], [], [], 0, 0
    for record in records:
        sid = record[5]; row = updated[sid]; v, f = row["vertices6"], row["faces"]
        chunks.append(RECORD.pack(row["body_index"], nv, len(v), ni, f.size,
                                  sid, row["layer"], row["flags"]))
        vv.append(np.asarray(v, dtype="<f4").tobytes())
        ff.append((f.reshape(-1)+nv).astype("<u4").tobytes())
        nv += len(v); ni += f.size
    result = HEADER.pack(header[0], header[1], len(records), nv, ni, header[5], header[6]) + b"".join(chunks+vv+ff)
    _, _, verified = _parse_payload(result)
    _require(all(_record_content_bytes(rows[s]) == _record_content_bytes(verified[s])
                 for s in rows if s != stable_id), "another anatomical surface changed")
    output_sha = hashlib.sha256(result).hexdigest()
    detail.update(base_payload_sha256=source_sha, base_receipt_sha256=_sha(base_receipt),
                  all_other_record_geometry_bytes_preserved=True)
    if stable_id == 456:
        receipt["provenance"]["passive_taenia_wall_region"] = detail
    regions = receipt["provenance"].setdefault("passive_taenia_wall_regions", {})
    regions[str(stable_id)] = detail
    source_map[str(stable_id)]["reference_geometry_status"] = detail["parameter_status"]
    source_map[str(stable_id)]["passive_representation"] = detail["representation"]
    if stable_id == 457:
        source_map["457"]["source_shell_status"] = "retained_in_provenance_not_rendered_as_independent_solid"
    receipt["payload"].update(path=str(output/"resting-thorax.nhanatomy"), sha256=output_sha,
                              surface_count=len(records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    cardiac["output_anatomy_payload_sha256"] = output_sha
    wall["output_anatomy_payload_sha256"] = output_sha
    receipt["qualification"][f"passive_taenia_wall_region_{stable_id}"] = detail["qualification"]
    output.mkdir(parents=True)
    (output/"resting-thorax.nhanatomy").write_bytes(result)
    receipt_path = output/"resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1", "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"],
                "qualification": receipt["qualification"], "source_surfaces": source_map,
                "mass_geometry_accounting": receipt["mass_geometry_accounting"]}
    (output/"resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest

# Rows explicitly prepared by the passive reference composition; respiratory,
# cardiac, and every unlisted anatomy record remain owned by the target base.
PASSIVE_REFERENCE_TRANSFER_IDS = tuple(sorted({
    2, 3, 13, *range(398, 454), 455, 456, 457, 462,
}))


def compose_passive_rows_onto_respiratory_base(*, base_payload: Path, base_receipt: Path,
                                               passive_payload: Path, passive_receipt: Path,
                                               output: Path) -> dict:
    """Transfer only the reviewed passive rows onto an accepted respiratory source.

    This produces a pre-native assembly asset: the complete-payload cardiac
    binding is deliberately left unchanged and therefore requires re-emission
    by its owner after final scene composition.
    """
    _require(not output.exists(), "retain previous anatomy evidence; choose a new output directory")
    for path in (base_payload, base_receipt, passive_payload, passive_receipt):
        _require(path.is_file(), f"row-transfer input is missing: {path}")

    base_raw = base_payload.read_bytes()
    passive_raw = passive_payload.read_bytes()
    base_sha = hashlib.sha256(base_raw).hexdigest()
    passive_sha = hashlib.sha256(passive_raw).hexdigest()
    base_r = json.loads(base_receipt.read_text())
    passive_r = json.loads(passive_receipt.read_text())
    for receipt, digest, role in ((base_r, base_sha, "respiratory base"),
                                  (passive_r, passive_sha, "passive reference")):
        _require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
                 and receipt.get("payload", {}).get("sha256") == digest
                 and receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == digest,
                 f"{role} receipt does not bind its exact NHANAT payload")

    base_header, base_records, base_rows = _parse_payload(base_raw)
    pass_header, _pass_records, pass_rows = _parse_payload(passive_raw)
    _require(set(base_rows) == set(pass_rows)
             and base_header[5:] == pass_header[5:],
             "source and target anatomy inventories or body registration differ")
    transfer = set(PASSIVE_REFERENCE_TRANSFER_IDS)
    _require(transfer.issubset(base_rows),
             "respiratory base lacks an explicitly reviewed passive row")
    _require(transfer.issubset(pass_rows),
             "passive reference source lacks an explicitly reviewed row")

    base_map = base_r.get("provenance", {}).get("source_id_map", {})
    pass_map = passive_r.get("provenance", {}).get("source_id_map", {})
    for sid in sorted(transfer):
        target_meta = base_map.get(str(sid), {})
        source_meta = pass_map.get(str(sid), {})
        _require(target_meta.get("source_member") == source_meta.get("source_member")
                 and target_meta.get("source_sha256") == source_meta.get("source_sha256")
                 and target_meta.get("source_owner_metadata", {}).get("member_id")
                 == source_meta.get("source_owner_metadata", {}).get("member_id"),
                 f"passive row {sid} changed source identity")
        _require((base_rows[sid]["body_index"], base_rows[sid]["layer"], base_rows[sid]["flags"])
                 == (pass_rows[sid]["body_index"], pass_rows[sid]["layer"], pass_rows[sid]["flags"]),
                 f"passive row {sid} changed body/layer ownership")

    # These two colon owners are the untouched shared frame for the accepted
    # respiratory candidate and the copied taenia host regions.
    for sid in (454, 460):
        _require(sid in base_rows and sid in pass_rows
                 and _record_content_bytes(base_rows[sid]) == _record_content_bytes(pass_rows[sid]),
                 f"required unchanged colon parent {sid} differs")

    updated = copy.deepcopy(base_rows)
    row_hashes = {}
    for sid in sorted(transfer):
        row_hashes[str(sid)] = {
            "base_record_sha256": hashlib.sha256(_record_content_bytes(base_rows[sid])).hexdigest(),
            "passive_record_sha256": hashlib.sha256(_record_content_bytes(pass_rows[sid])).hexdigest(),
            "source_member": pass_map[str(sid)]["source_member"],
            "source_name": pass_map[str(sid)].get("name"),
        }
        updated[sid] = copy.deepcopy(pass_rows[sid])

    chunks, vv, ff, nv, ni = [], [], [], 0, 0
    for record in base_records:
        body_index, _vs, _vc, _is, _ic, sid, layer, flags = map(int, record)
        row = updated[sid]
        vertices = np.asarray(row["vertices6"], dtype="<f4")
        faces = np.asarray(row["faces"], dtype=np.int64).reshape(-1, 3)
        _require(vertices.ndim == 2 and vertices.shape[1] == 6 and np.isfinite(vertices).all()
                 and (len(faces) == 0 or (int(faces.min()) >= 0 and int(faces.max()) < len(vertices))),
                 f"transferred row {sid} is malformed")
        chunks.append(RECORD.pack(body_index, nv, len(vertices), ni, faces.size, sid, layer, flags))
        vv.append(vertices.tobytes())
        ff.append((faces.reshape(-1) + nv).astype("<u4").tobytes())
        nv += len(vertices); ni += faces.size
    result = HEADER.pack(base_header[0], base_header[1], len(base_records), nv, ni,
                         base_header[5], base_header[6]) + b"".join(chunks + vv + ff)
    _, _, verified = _parse_payload(result)
    for sid in base_rows:
        expected = pass_rows[sid] if sid in transfer else base_rows[sid]
        _require(_record_content_bytes(expected) == _record_content_bytes(verified[sid]),
                 f"row transfer changed unexpected or failed to replace stable ID {sid}")

    output.mkdir(parents=True)
    payload_path = output / "resting-thorax.nhanatomy"
    payload_path.write_bytes(result)
    output_sha = hashlib.sha256(result).hexdigest()
    receipt = copy.deepcopy(base_r)
    receipt["payload"].update(path=str(payload_path), sha256=output_sha,
                              surface_count=len(base_records), vertex_count=nv, index_count=ni)
    receipt["functional_bindings"]["anatomy_payload_sha256"] = output_sha
    for sid in sorted(transfer):
        receipt["provenance"]["source_id_map"][str(sid)] = copy.deepcopy(pass_map[str(sid)])
    source_evidence = {
        key: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        for key, value in passive_r.get("provenance", {}).items()
        if key in {"passive_bowel_reference_composition", "pylorus_stomach_arrangement",
                   "passive_taenia_wall_region", "passive_taenia_wall_regions",
                   "passive_component_identity_refresh"}
    }
    transfer_detail = {
        "schema": "numi.human.passive_reference_row_transfer.v1",
        "owner": "numilab_human.resting_taenia_reference.compose_passive_rows_onto_respiratory_base",
        "target_payload_sha256": base_sha,
        "target_receipt_sha256": _sha(base_receipt),
        "passive_source_payload_sha256": passive_sha,
        "passive_source_receipt_sha256": _sha(passive_receipt),
        "transferred_stable_ids": sorted(transfer),
        "transferred_source_rows": row_hashes,
        "unchanged_parent_ids_checked": [454, 460],
        "source_passive_provenance_entry_sha256": source_evidence,
        "all_unlisted_geometry_bytes_preserved": True,
        "passive_mass_or_physiology_added": False,
        "native_readiness": "not_ready_pending_final_cardiac_binding_reemit_and_integrated_cycle_audit",
        "cardiac_binding_note": "The target cardiac binding is preserved unchanged and remains bound to the pre-transfer complete anatomy payload; its existing owner must re-emit the binding after final assembly.",
    }
    receipt.setdefault("provenance", {})["passive_reference_row_transfer"] = transfer_detail
    receipt.setdefault("qualification", {})["passive_reference_row_transfer"] = (
        "Preparatory row assembly only; target respiratory rows and unlisted records are preserved. "
        "The complete-payload cardiac binding is intentionally stale until its existing owner re-emits it, "
        "and integrated native cycle qualification remains pending.")
    receipt_path = output / "resting-anatomy-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest = {"schema": "numi.human.resting-anatomy-manifest.v1",
                "payload": receipt["payload"],
                "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
                "functional_bindings": receipt["functional_bindings"],
                "qualification": receipt["qualification"],
                "source_surfaces": receipt["provenance"]["source_id_map"],
                "passive_reference_row_transfer": transfer_detail,
                "mass_geometry_accounting": receipt["mass_geometry_accounting"]}
    (output / "resting-anatomy-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
