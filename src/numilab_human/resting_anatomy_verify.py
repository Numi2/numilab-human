"""Shared exact geometry gates for accepted resting-human captures.

This oracle consumes existing MRVPACK2 and accepted-render receipts. It neither
changes meshes nor advances simulation state. Selected geometry is not whole-body
or physiological qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import cardiac_cavity_intersections as ci
from .cardiac_cavity_geometry import analyze_topology
from .model import ImportError as HumanImportError
from .torso_anatomy_audit import _pack_sections

# Native appendTorsoAnatomyGeometry uses this layer-indexed semantic table.
# In particular, a nerve's stable ID must not be looked up as an organ ID.
NHA_LAYER_SEMANTICS = (51010, 51011, 51012, 51020, 51021, 51022, 51023,
                       51024, 51025, 51026, 51027, 51028, 51029, 51030, 51031)
ANATOMY_SEMANTICS = frozenset((51004, 51005, 51006, 51007, 51009,
                              *NHA_LAYER_SEMANTICS))


def compiled_bone_identity_map(by_body, body_order):
    """Join source-decoded bone IDs to actual Core owners, never numeric guesses."""
    if (not isinstance(body_order, list) or not body_order
            or any(not isinstance(name, str) or not name for name in body_order)
            or len(set(body_order)) != len(body_order)):
        raise ValueError("bone identity requires unique runtime body names")
    result = {}
    for body, rows in by_body.items():
        if type(body) is not int or not 0 <= body < len(body_order):
            raise ValueError("bone identity has an invalid Core body index")
        name = body_order[body]
        laterality = "right" if name.endswith("_r") else "left" if name.endswith("_l") else None
        for row in rows:
            sid = row.get("stable_id")
            member = row.get("member_id")
            if type(sid) is not int or sid <= 0 or sid in result:
                raise ValueError("bone identity has a duplicate or invalid stable ID")
            if not isinstance(member, str) or not member:
                raise ValueError("bone identity lacks its source member")
            result[sid] = {"typed_identity": [51004, sid], "core_body_index": body,
                              "core_body_name": name, "laterality": laterality,
                              "source_member_id": member}
    if not result:
        raise ValueError("bone identity table is empty")
    return result


def load_compiled_bone_identities(bone_artifact, body_manifest, *, capture=None):
    """Use the existing NHBONES owner reader and its bound runtime manifest."""
    from . import model
    artifact, runtime_path = Path(bone_artifact), Path(body_manifest)
    manifest_path = artifact / "bodyparts3d-myosim-major-bones.manifest.json"
    bone_manifest = model.read_json(manifest_path)
    runtime = model.read_json(runtime_path)
    expected = bone_manifest["source"]["runtime_reference"]["manifest"]["sha256"]
    if model.sha256(runtime_path) != expected:
        raise ValueError("bone identity runtime manifest hash differs from compiled source")
    source_sha = runtime["source"]["archive_sha256"]
    by_body, descriptor, payload = model._numi_human_bone_envelope_surfaces(artifact, source_sha)
    identities = compiled_bone_identity_map(by_body, runtime["core_tree"]["body_order"])
    proof = {
        "bone_payload": {"path": str(payload.resolve()), "sha256": descriptor["sha256"]},
        "bone_manifest": {"path": str(manifest_path.resolve()), "sha256": model.sha256(manifest_path)},
        "runtime_manifest": {"path": str(runtime_path.resolve()), "sha256": expected},
        "source_archive_sha256": source_sha}
    if capture is not None:
        invocation_path = Path(capture).parent.parent / "invocation.json"
        invocation = model.read_json(invocation_path)
        actual_bones = Path(invocation["argv"][3])
        if model.sha256(actual_bones) != descriptor["sha256"]:
            raise ValueError("compiled bone artifact differs from native capture input")
        proof["native_invocation"] = {
            "path": str(invocation_path.resolve()), "sha256": model.sha256(invocation_path)}
    return identities, proof


class SurfaceTopologyError(ValueError):
    """A source surface failed indexed or exact-coordinate topology checks."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")))


def _surface_topology_diagnostic(key, stage, topology, triangles,
                                 input_faces, topology_to_input_vertex,
                                 input_source_vertex_ids, input_positions):
    degenerate = [int(value) for value in topology["degenerate_face_ids"]]
    duplicate_ids = {int(value) for value in topology["duplicate_face_ids"]}
    if not degenerate and not duplicate_ids:
        return None
    groups = {}
    for face_id, face in enumerate(triangles):
        groups.setdefault(tuple(sorted(int(value) for value in face)), []).append(face_id)
    duplicate_groups = [face_ids for face_ids in groups.values() if len(face_ids) > 1]
    duplicate_groups.sort(key=lambda face_ids: face_ids[0])
    witness_face_ids = set(degenerate)
    for group in duplicate_groups:
        if duplicate_ids.intersection(group):
            witness_face_ids.update(group)
    reported_duplicate_groups = [group for group in duplicate_groups
                                 if duplicate_ids.intersection(group)]
    witness_faces = []
    for face_id in sorted(witness_face_ids)[:24]:
        topology_vertices = [int(value) for value in triangles[face_id]]
        input_vertices = [int(value) for value in input_faces[face_id]]
        representative_vertices = [int(topology_to_input_vertex[value])
                                   for value in topology_vertices]
        witness_faces.append({
            "source_face_index": int(face_id),
            "topology_vertex_ids": topology_vertices,
            "input_local_vertex_indices": input_vertices,
            "representative_input_local_vertex_indices": representative_vertices,
            "input_source_vertex_ids": [int(input_source_vertex_ids[value]) for value in input_vertices],
            "input_float32_xyz_bits": [
                [int(value) for value in input_positions[index].view(np.uint32)]
                for index in input_vertices
            ],
        })
    return {
        "typed_identity": list(key),
        "topology_stage": stage,
        "degenerate_face_count": len(degenerate),
        "degenerate_face_ids": degenerate[:24],
        "duplicate_face_count": len(duplicate_ids),
        "duplicate_face_ids": sorted(duplicate_ids)[:24],
        "duplicate_face_group_count": len(reported_duplicate_groups),
        "duplicate_face_groups": reported_duplicate_groups[:12],
        "witness_faces": witness_faces,
        "witness_truncated": (len(degenerate) > 24 or len(duplicate_ids) > 24
                              or len(reported_duplicate_groups) > 12
                              or len(witness_face_ids) > 24),
        "coordinate_rule": "exact emitted Float32 coordinates; no tolerance welding",
    }


def source_surface_key(stable_id, layer, *, payload_kind):
    """Resolve typed source IDs using the existing native layer convention."""
    if type(stable_id) is not int or stable_id <= 0 or type(layer) is not int:
        raise ValueError("source surface identity and layer must be integers")
    if payload_kind == "nhanatomy":
        if not 1 <= layer <= len(NHA_LAYER_SEMANTICS):
            raise ValueError("invalid native anatomy layer")
        semantic = NHA_LAYER_SEMANTICS[layer - 1]
    elif payload_kind == "nhtissue":
        if layer not in (1, 2, 3):
            raise ValueError("invalid native tissue layer")
        semantic = {1: 51005, 2: 51006, 3: 51004}[layer]
    else:
        raise ValueError("unsupported native surface payload kind")
    return typed_key((semantic, stable_id))


def typed_key(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("surface identity requires (semantic, stable_id)")
    if any(type(v) is not int for v in value):
        raise ValueError("surface identity must contain integers")
    if value[0] not in ANATOMY_SEMANTICS or value[1] <= 0:
        raise ValueError("invalid typed anatomy identity")
    return tuple(value)


def gid_index_map(gids):
    """Index unique integer GIDs without assuming their source order is sorted."""
    values = np.asarray(gids)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("GIDs must be a one-dimensional integer array")
    result = {}
    for index, value in enumerate(values):
        gid = int(value)
        if gid in result:
            raise ValueError(f"duplicate GID: {gid}")
        result[gid] = index
    return result


def keyed_row_indices(source_gids, requested_gids):
    """Return source row indices in requested GID order, failing on missing IDs."""
    source_index = gid_index_map(source_gids)
    requested = np.asarray(requested_gids)
    if requested.ndim != 1 or requested.dtype.kind not in "iu":
        raise ValueError("requested GIDs must be a one-dimensional integer array")
    missing = [int(gid) for gid in requested if int(gid) not in source_index]
    if missing:
        raise ValueError("missing GIDs: " + str(missing[:8]))
    return np.fromiter((source_index[int(gid)] for gid in requested),
                       dtype=np.intp, count=len(requested))


def _compact_topology(topology):
    keys = ("vertex_count", "face_count", "face_component_count",
            "boundary_edge_count", "closed_oriented_manifold_candidate")
    result = {k: topology[k] for k in keys}
    for k in ("nonmanifold_edges", "orientation_defect_edges",
              "degenerate_face_ids", "repeated_vertex_face_ids",
              "duplicate_face_ids", "vertex_manifold_defect_ids",
              "unused_vertex_ids", "boundary_branch_vertex_ids"):
        if k in topology:
            result[k + "_count"] = len(topology[k])
    return result


@dataclass
class PreparedSurface:
    key: tuple[int, int]
    positions: np.ndarray
    faces: np.ndarray
    source_vertex_ids: np.ndarray
    normals: np.ndarray | None
    raw_topology: dict
    quotient_topology: dict
    quotient_positions: np.ndarray
    quotient_faces: np.ndarray
    source_vertex_to_quotient: np.ndarray
    records: tuple
    index: object
    source_face_lineage: tuple | None = field(default=None, repr=False)
    source_vertex_lineage: dict | None = field(default=None, repr=False)
    ordered_core_body_indices: tuple | None = field(default=None, repr=False)
    core_weights_by_gid: dict | None = field(default=None, repr=False)
    rigid_owner_proof: dict | None = field(default=None, repr=False)
    declared_core_body_indices: tuple | None = field(default=None, repr=False, kw_only=True)
    _self_result: dict | None = field(default=None, repr=False)
    _meter_index: object | None = field(default=None, repr=False)
    _face_records: dict | None = field(default=None, repr=False)
    _vertex_indices: dict | None = field(default=None, repr=False)

    def face_record(self, face_id):
        if self._face_records is None:
            self._face_records = {}
            for record in self.records:
                self._face_records.setdefault(int(record[3]), record)
        return self._face_records.get(face_id)

    def _required_face_record(self, face_id):
        if type(face_id) is not int or not 0 <= face_id < len(self.faces):
            raise ValueError("face identity must be an in-range integer")
        record = self.face_record(face_id)
        if record is None:
            raise ValueError("face identity has no prepared record")
        return record

    def triangle_meters(self, face_id):
        """Exact rational metres, retaining emitted Float32 coordinate values.

        Predicate records use an integer lattice with one unit equal to
        2**-149 metres. Do not mix those records with metric lineage points.
        """
        record = self._required_face_record(face_id)
        return tuple(tuple(Fraction(value, 1 << 149) for value in point)
                     for point in record[0])

    def intersection_points_meters(self, other, first_face, second_face):
        """Return sorted unique geometric contacts in exact rational metres.

        The predicate can repeat endpoints while enumerating triangle edges.
        Deduplication affects contact classification only, never mesh bytes.
        """
        if not isinstance(other, PreparedSurface):
            raise TypeError("intersection requires another prepared surface")
        first = self._required_face_record(first_face)
        second = other._required_face_record(second_face)
        points = ci.triangle_intersection_points(first[0], second[0])
        return tuple(sorted({tuple(Fraction(value) / (1 << 149)
                                   for value in point) for point in points}))

    def vertex_index(self, gid):
        if self._vertex_indices is None:
            self._vertex_indices = gid_index_map(self.source_vertex_ids)
        return self._vertex_indices.get(gid)

    def self_audit(self):
        if self._self_result is None:
            self._self_result = ci._audit_pair_prepared_first(
                self.index, self.records, same_surface=True)
        return self._self_result

    def self_audit_since(self, parent):
        """Check every changed face against this complete surface exactly.

        The parent must have an exact zero-self result from the same oracle.
        Unchanged ordered triangles retain the parent's pair proof; quotient
        vertex numbers may differ, because shared-point identity is geometric.
        This reuses an in-memory proof, never an unverified imported result.
        Source ancestry, material bindings and cross-surface gates remain
        separate requirements.
        """
        if not isinstance(parent, PreparedSurface) or parent.key != self.key:
            raise ValueError("incremental self audit requires the same typed parent surface")
        if not np.array_equal(parent.source_vertex_ids, self.source_vertex_ids):
            raise ValueError("incremental self audit requires unchanged ordered vertex identities")
        parent_result = parent.self_audit()
        if parent_result["count"]:
            raise ValueError("incremental self audit requires a zero-self parent")
        if self._self_result is not None:
            return self._self_result
        parent_faces = {record[3]: record[0] for record in parent.records}
        changed = [int(record[3]) for record in self.records
                   if parent_faces.get(record[3]) != record[0]]
        result = ci._audit_self_pairs_touching_faces(self.records, changed)
        parent_diagnostics = parent.diagnostics()
        result["incremental_proof"] = {
            "parent_positions_sha256": parent_diagnostics["positions_sha256"],
            "parent_faces_sha256": parent_diagnostics["faces_sha256"],
            "parent_self_result_sha256": _canonical_json_sha256(parent_result),
            "changed_face_ids": changed,
            "retained_unchanged_face_count": len(self.records) - len(changed),
            "rule": ("Unchanged exact ordered triangles reuse the zero-self parent; "
                     "all pairs incident on changed faces use the complete current "
                     "record set and the existing exact predicate."),
        }
        self._self_result = result
        return result

    @property
    def closed(self):
        # Unreferenced source slots carry ancestry, not a boundary of the
        # rendered solid. Preserve their bytes and diagnose them separately.
        topology = self.quotient_topology
        return not any(topology[key] for key in (
            "boundary_edges", "nonmanifold_edges", "orientation_defect_edges",
            "degenerate_face_ids", "repeated_vertex_face_ids", "duplicate_face_ids",
            "vertex_manifold_defect_ids", "boundary_branch_vertex_ids"))

    def meter_index(self):
        if not self.closed or self.self_audit()["count"]:
            raise ValueError("containment requires a closed embedded target")
        if self._meter_index is None:
            self._meter_index = ci.prepare_float32_point_location_meters(
                self.quotient_positions.tolist(), self.quotient_faces.tolist())
        return self._meter_index

    def diagnostics(self):
        return {
            "identity": list(self.key),
            "positions_sha256": hashlib.sha256(self.positions.tobytes()).hexdigest(),
            "faces_sha256": hashlib.sha256(self.faces.tobytes()).hexdigest(),
            "exact_duplicate_vertex_count": len(self.positions) - len(self.quotient_positions),
            "raw_topology": _compact_topology(self.raw_topology),
            "exact_coordinate_quotient_topology": _compact_topology(self.quotient_topology),
            "referenced_surface_closed": self.closed,
            "declared_material_owner_order": (None if self.declared_core_body_indices is None
                                               else list(self.declared_core_body_indices)),
            "unused_slot_policy": "Unreferenced slots retain source ancestry; they do not create a surface boundary.",
            "seam_policy": "Exact packed coordinate identity for predicates only; source faces and mesh bytes retained.",
            "rigid_owner_proof": None if self.rigid_owner_proof is None else {
                "body_index": self.rigid_owner_proof["body_index"],
                "capture_receipt_sha256": self.rigid_owner_proof["capture_receipt_sha256"],
                "position_f32_sha256": hashlib.sha256(
                    self.rigid_owner_proof["position_f32_bytes"]).hexdigest(),
                "quaternion_f32_sha256": hashlib.sha256(
                    self.rigid_owner_proof["quaternion_f32_bytes"]).hexdigest(),
            },
        }


def prepare_surface(key, positions, faces, *, normals=None, source_vertex_ids=None,
                    source_face_lineage=None, source_vertex_lineage=None,
                    ordered_core_body_indices=None, core_weight_gids=None,
                    core_weights_float64=None, rigid_owner_proof=None,
                    expected_ordered_core_body_indices=None):
    key = typed_key(key)
    p = np.asarray(positions)
    f = np.asarray(faces)
    if p.dtype != np.dtype("float32") or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("positions must be emitted Float32 XYZ in metres")
    if len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("empty or nonfinite positions")
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu" or len(f) == 0:
        raise ValueError("faces must be nonempty integer local triangles")
    if np.any(f < 0) or np.any(f >= len(p)):
        raise ValueError("invalid local face index")
    if normals is not None:
        normals = np.asarray(normals)
        if normals.dtype != np.dtype("float32") or normals.shape != p.shape or not np.isfinite(normals).all():
            raise ValueError("invalid emitted Float32 normals")
        # Only vertices referenced by the emitted triangle surface need a
        # display normal. Allocated but unused source slots retain their
        # original finite normal bytes, which may be zero.
        referenced_vertices = np.unique(f.reshape(-1))
        referenced_normals = normals[referenced_vertices].astype(float)
        if np.any(np.abs(np.linalg.norm(referenced_normals, axis=1) - 1) > 2e-3):
            raise ValueError("referenced emitted normals are not unit")
    if source_vertex_ids is None:
        source_vertex_ids = np.arange(len(p), dtype=np.uint64)
    source_vertex_ids = np.asarray(source_vertex_ids)
    if source_vertex_ids.shape != (len(p),) or source_vertex_ids.dtype.kind not in "iu":
        raise ValueError("invalid source vertex ancestry")
    if len(np.unique(source_vertex_ids)) != len(p):
        raise ValueError("duplicate source vertex ancestry")
    source_metadata_present = (source_face_lineage is not None
                               or source_vertex_lineage is not None)
    core_metadata = (ordered_core_body_indices, core_weight_gids, core_weights_float64)
    has_core_metadata = any(value is not None for value in core_metadata)
    if source_metadata_present and (source_face_lineage is None or source_vertex_lineage is None):
        raise ValueError("shared-interface proof requires both source face and vertex lineage")
    if has_core_metadata and any(value is None for value in core_metadata):
        raise ValueError("shared-interface proof requires complete ordered core material lineage")
    if source_metadata_present and not has_core_metadata and rigid_owner_proof is None:
        raise ValueError("shared-interface proof requires ordered core rows or an accepted rigid owner")
    if rigid_owner_proof is not None and not source_metadata_present:
        raise ValueError("rigid owner proof requires source face and vertex lineage")
    if has_core_metadata and rigid_owner_proof is not None:
        raise ValueError("choose ordered core material rows or a rigid owner proof")
    # The historical forearm contract remains strictly positive with 21 owners.
    # Other existing native parent rows must explicitly pin their complete support
    # inventory; structural zero coefficients are retained, never replaced by floors.
    declared_cores = None
    if expected_ordered_core_body_indices is not None:
        declared_cores = _declared_material_inventory(expected_ordered_core_body_indices)
        if not has_core_metadata or not source_metadata_present:
            raise ValueError("declared material inventory requires complete source and actual owner rows")
    face_lineage = None
    vertex_lineage = None
    core_body_indices = None
    weights_by_gid = None
    normalized_rigid_owner = (None if rigid_owner_proof is None
                              else _normalize_rigid_owner_proof(rigid_owner_proof))
    if source_metadata_present:
        face_lineage = tuple(source_face_lineage)
        if len(face_lineage) != len(f) or any(not isinstance(row, dict) for row in face_lineage):
            raise ValueError("source face lineage must map every emitted face")
        vertex_lineage = {}
        for gid, row in source_vertex_lineage.items():
            if type(gid) is not int or not isinstance(row, dict):
                raise ValueError("source vertex lineage must use integer GIDs and records")
            vertex_lineage[gid] = row
    if has_core_metadata:
        core_body_indices = tuple(ordered_core_body_indices)
        if declared_cores is None:
            if (len(core_body_indices) != 21 or any(type(value) is not int for value in core_body_indices)
                    or len(set(core_body_indices)) != 21):
                raise ValueError("material owner order must contain 21 unique integer body indices")
        elif core_body_indices != declared_cores:
            raise ValueError("actual material owner order differs from the complete declared inventory")
        weight_gids = np.asarray(core_weight_gids)
        weights = np.asarray(core_weights_float64)
        if (weight_gids.ndim != 1 or weight_gids.dtype.kind not in "iu"
                or weights.shape != (len(weight_gids), len(core_body_indices))
                or weights.dtype != np.dtype("float64") or not np.isfinite(weights).all()):
            raise ValueError("material weights must be finite Float64 rows keyed by GID")
        weight_index = gid_index_map(weight_gids)
        weights_by_gid = {gid: np.array(weights[index], dtype=np.float64, copy=True)
                          for gid, index in weight_index.items()}
        referenced_gids = {int(source_vertex_ids[int(local)]) for face in f for local in face}
        missing = sorted(referenced_gids - weights_by_gid.keys())
        if missing:
            raise ValueError("missing ordered material rows for referenced GIDs: " + str(missing[:8]))
        bad = [gid for gid in referenced_gids
               if np.any(weights_by_gid[gid] < 0 if declared_cores is not None
                         else weights_by_gid[gid] <= 0)
               or abs(float(weights_by_gid[gid].sum()) - 1.0) > 1e-12]
        if bad:
            policy = "nonnegative" if declared_cores is not None else "strictly positive"
            raise ValueError("referenced material rows must be " + policy + " and normalized: " + str(bad[:8]))
    # Numeric coordinate identity treats +0 and -0 as the same exact point.
    keys = [ci.float32_point_lattice_key(row) for row in p.tolist()]
    identities = {}
    keep, mapping = [], []
    for i, point in enumerate(keys):
        if point not in identities:
            identities[point] = len(keep)
            keep.append(i)
        mapping.append(identities[point])
    mapping = np.asarray(mapping, dtype=np.uint32)
    qp = np.ascontiguousarray(p[keep])
    qf = np.ascontiguousarray(mapping[f], dtype=np.uint32)
    raw = analyze_topology(p.tolist(), f.tolist())
    quotient = analyze_topology(qp.tolist(), qf.tolist())
    topology_inputs = (
        ("indexed", raw, f, np.arange(len(p), dtype=np.int64)),
        ("exact_float32_coordinate_quotient", quotient, qf,
         np.asarray(keep, dtype=np.int64)),
    )
    for stage, topology, triangles, topology_to_input_vertex in topology_inputs:
        diagnostic = _surface_topology_diagnostic(
            key, stage, topology, triangles, f, topology_to_input_vertex,
            source_vertex_ids, p)
        if diagnostic is not None:
            raise SurfaceTopologyError(diagnostic)
    if quotient["nonmanifold_edges"] or quotient["orientation_defect_edges"]:
        raise ValueError("exact seam quotient is nonmanifold or inconsistently oriented")
    records = tuple(ci._records([keys[i] for i in keep], qf.tolist()))
    return PreparedSurface(key, np.array(p, copy=True, order="C"), np.array(f, dtype=np.uint32, copy=True, order="C"),
                           source_vertex_ids.copy(), normals, raw, quotient, qp, qf, mapping,
                           records, ci._prepare_surface_aabb(records), face_lineage,
                           vertex_lineage, core_body_indices, weights_by_gid,
                           normalized_rigid_owner, declared_core_body_indices=declared_cores)


def _declared_material_inventory(value):
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= 192:
        raise ValueError("declared material inventory must contain 1..192 ordered owners")
    if (any(type(core) is not int or not 0 <= core < 192 for core in value)
            or len(set(value)) != len(value)):
        raise ValueError("declared material inventory requires unique nonnegative integer owners")
    return tuple(value)


def _normalize_rigid_owner_proof(value):
    """Canonicalize an already-validated accepted rigid-pose owner record."""
    if not isinstance(value, dict):
        raise TypeError("rigid owner proof must be a mapping")
    expected = {"body_index", "capture_receipt_sha256", "position_m", "quaternion_xyzw"}
    if set(value) != expected:
        raise ValueError("rigid owner proof fields differ from the accepted-pose contract")
    body = value.get("body_index")
    receipt_sha = value.get("capture_receipt_sha256")
    if type(body) is not int or body < 0:
        raise ValueError("rigid owner proof body index must be a nonnegative integer")
    if (not isinstance(receipt_sha, str) or len(receipt_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in receipt_sha)):
        raise ValueError("rigid owner proof capture receipt SHA-256 is invalid")
    arrays = []
    for key, length in (("position_m", 3), ("quaternion_xyzw", 4)):
        raw = value.get(key)
        if (not isinstance(raw, (list, tuple)) or len(raw) != length
                or any(type(item) not in (int, float) or not math.isfinite(float(item))
                       for item in raw)):
            raise ValueError("rigid owner proof has invalid " + key)
        array = np.asarray(raw, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError("rigid owner proof has nonfinite Float32 " + key)
        arrays.append(array)
    if abs(float(np.linalg.norm(arrays[1].astype(np.float64))) - 1.0) > 2e-3:
        raise ValueError("rigid owner quaternion is not normalized")
    return {"body_index": body, "capture_receipt_sha256": receipt_sha,
            "position_f32_bytes": arrays[0].tobytes(),
            "quaternion_f32_bytes": arrays[1].tobytes()}


def _overlap(first, second):
    return bool(np.all(first.positions.min(axis=0) <= second.positions.max(axis=0)) and
                np.all(second.positions.min(axis=0) <= first.positions.max(axis=0)))


def _contained_components(source, target, *, inspect_all_faces=False):
    if not target.closed:
        return {"status": "not_checked_open_target", "inside_component_witnesses": [],
                "outside_component_witnesses": [], "target_preparation": target.diagnostics()}
    self_result = target.self_audit()
    if self_result["count"]:
        return {"status": "not_checked_nonembedded_target", "inside_component_witnesses": [],
                "outside_component_witnesses": [], "target_self_audit": self_result,
                "target_preparation": target.diagnostics()}
    prepared = target.meter_index()
    inside_witnesses, outside_witnesses = [], []
    component_rows = []
    for component_index, component in enumerate(source.quotient_topology["face_components"]):
        inside = outside = None
        boundary_count = 0
        face_ids = component if inspect_all_faces else component[:1]
        for face_id in face_ids:
            triangle = source.quotient_positions[source.quotient_faces[int(face_id)]]
            point = tuple(sum((Fraction.from_float(float(row[axis])) for row in triangle),
                              Fraction(0)) / 3 for axis in range(3))
            result = ci.point_location_prepared_indexed(point, prepared)
            location = result["location"]
            if location == "boundary":
                boundary_count += 1
                continue
            if location not in ("inside", "outside"):
                return {"status": "unresolved_exact_point_location",
                        "inside_component_witnesses": inside_witnesses,
                        "outside_component_witnesses": outside_witnesses,
                        "component_checks": component_rows}
            witness = {"component_index": component_index, "source_face_id": int(face_id),
                       "sample": "exact_triangle_centroid_off_target_boundary",
                       "point_m": [float(v) for v in point]}
            if location == "inside" and inside is None:
                inside = witness
            elif location == "outside" and outside is None:
                outside = witness
        if inside is None and outside is None:
            return {"status": "unresolved_all_component_samples_on_boundary",
                    "inside_component_witnesses": inside_witnesses,
                    "outside_component_witnesses": outside_witnesses,
                    "component_checks": component_rows}
        if inside is not None:
            inside_witnesses.append(inside)
        if outside is not None:
            outside_witnesses.append(outside)
        component_rows.append({
            "component_index": component_index,
            "face_count": len(component),
            "sampled_face_count": len(face_ids),
            "sampled_off_boundary_face_count": len(face_ids) - boundary_count,
            "boundary_centroid_face_count": boundary_count,
            "has_inside_face_interior": inside is not None,
            "has_outside_face_interior": outside is not None,
        })
    return {"status": "checked_exact_rational_metres",
            "inside_component_witnesses": inside_witnesses,
            "outside_component_witnesses": outside_witnesses,
            "component_checks": component_rows}


def _intended_enclosures(value):
    """Ordered enclosed-surface/container identities, never a boundary exemption."""
    pairs = set()
    for pair in value:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("intended enclosure requires enclosed and container identities")
        inside, container = (typed_key(key) for key in pair)
        if inside == container:
            raise ValueError("surface cannot be its own intended enclosure")
        pairs.add((inside, container))
    return pairs


def _lineage_value_present(value):
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _has_source_face_lineage(record):
    if not isinstance(record, dict):
        return False
    if any(_lineage_value_present(record.get(key)) for key in (
            "source_parent_face", "source_parent_faces_component_union",
            "source_face_support", "source_face_support_provenance")):
        return True
    parent = record.get("parent_054_face_ref")
    if isinstance(parent, dict) and any(_lineage_value_present(parent.get(key)) for key in (
            "parent_original_source_face_index", "parent_candidate_face_index", "row_local_face_index")):
        return True
    return any(
        type(row.get("source_face")) is int and row["source_face"] >= 0
        and isinstance(row.get("source_member"), str) and bool(row["source_member"])
        and isinstance(row.get("source_member_sha256"), str)
        and len(row["source_member_sha256"]) == 64
        and all(ch in "0123456789abcdef" for ch in row["source_member_sha256"])
        for row in _source_face_lineage_chain(record))


def _has_source_vertex_lineage(record):
    if not isinstance(record, dict):
        return False
    if any(_lineage_value_present(record.get(key)) for key in (
            "source_gid", "source_vertex_id", "source_parent_gids",
            "source_parent_faces", "source_face_support", "source_face_support_provenance",
            "alias_of", "contraction_of")):
        return True
    return (record.get("material_mapping_kind") == 3
            and record.get("material_binding") == "independent_existing_core_carrier"
            and 1 <= len(record.get("core_body_indices", ())) <= 192
            and len(record.get("core_weights", ())) == len(record.get("core_body_indices", ()))
            or _valid_source_point_crosswalk(record.get("source_point_crosswalk")))


def _source_point_crosswalk_digest(value):
    payload = {key: item for key, item in value.items()
               if key != "source_capture_crosswalk_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _cyclic_source_face_orientation(actual, canonical, *, reversed_order=False):
    actual = list(actual)
    expected = list(canonical)
    if reversed_order:
        expected = [expected[0], expected[2], expected[1]]
    return any(actual == expected[shift:] + expected[:shift] for shift in range(3))


def _valid_source_point_crosswalk(value):
    if not isinstance(value, dict) or set(value) != {
            "schema", "source_capture_crosswalk_sha256", "seam", "first", "second"}:
        return False
    if value.get("schema") != "numi.human.shared-source-seam-endpoint-crosswalk.v4":
        return False
    digest = value.get("source_capture_crosswalk_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or digest != _source_point_crosswalk_digest(value)):
        return False

    def valid_sha(item):
        return (isinstance(item, str) and len(item) == 64
                and all(ch in "0123456789abcdef" for ch in item))

    seam = value.get("seam")
    if not isinstance(seam, dict) or set(seam) != {
            "source_member", "source_member_sha256", "source_faces",
            "endpoint_source_vertex_ids"}:
        return False
    if (not isinstance(seam.get("source_member"), str) or not seam["source_member"]
            or not valid_sha(seam.get("source_member_sha256"))):
        return False
    faces = seam.get("source_faces")
    endpoints = seam.get("endpoint_source_vertex_ids")
    if (not isinstance(faces, list) or len(faces) != 2
            or not isinstance(endpoints, list) or not endpoints
            or any(type(v) is not int or v < 0 for v in endpoints)
            or len(set(endpoints)) != len(endpoints)):
        return False
    face_ids = []
    for row in faces:
        if not isinstance(row, dict) or set(row) != {"source_face", "source_vertex_ids"}:
            return False
        face_id, ids = row.get("source_face"), row.get("source_vertex_ids")
        if (type(face_id) is not int or face_id < 0 or not isinstance(ids, list) or len(ids) != 3
                or any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != 3):
            return False
        face_ids.append(face_id)
    if (face_ids[0] == face_ids[1]
            or not set(endpoints) <= (set(faces[0]["source_vertex_ids"])
                                      & set(faces[1]["source_vertex_ids"]))):
        return False

    side_fields = {"identity", "source_row_payload_sha256", "seam_faces", "vertex_bindings"}
    for side_name in ("first", "second"):
        side = value.get(side_name)
        if not isinstance(side, dict) or set(side) != side_fields:
            return False
        identity = side.get("identity")
        if (not isinstance(identity, list) or len(identity) != 2
                or any(type(v) is not int for v in identity)
                or not valid_sha(side.get("source_row_payload_sha256"))):
            return False
        seam_rows = side.get("seam_faces")
        if not isinstance(seam_rows, list) or len(seam_rows) != 2:
            return False
        expected_bindings = set()
        for source, row in zip(faces, seam_rows):
            if not isinstance(row, dict) or set(row) != {
                    "face_index", "source_face", "source_row_face",
                    "source_row_payload_sha256", "ordered_source_vertex_ids",
                    "ordered_capture_gids"}:
                return False
            ids = row.get("ordered_source_vertex_ids")
            gids = row.get("ordered_capture_gids")
            canonical = source["source_vertex_ids"]
            if (type(row.get("face_index")) is not int or row["face_index"] < 0
                    or row.get("source_face") != source["source_face"]
                    or type(row.get("source_face")) is not int
                    or type(row.get("source_row_face")) is not int or row["source_row_face"] < 0
                    or not valid_sha(row.get("source_row_payload_sha256"))
                    or row["source_row_payload_sha256"] != side["source_row_payload_sha256"]
                    or not isinstance(ids, list) or len(ids) != 3
                    or any(type(v) is not int for v in ids) or set(ids) != set(canonical)
                    or not _cyclic_source_face_orientation(
                        ids, canonical, reversed_order=(side_name == "second"))
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(v) is not int or v < 0 for v in gids)
                    or len(set(gids)) != 3):
                return False
            expected_bindings.update((source_id, gid)
                                     for source_id, gid in zip(ids, gids)
                                     if source_id in endpoints)
        bindings = side.get("vertex_bindings")
        if not isinstance(bindings, list):
            return False
        observed = set()
        for binding in bindings:
            if (not isinstance(binding, dict) or set(binding) != {"capture_gid", "source_vertex_id"}
                    or type(binding.get("capture_gid")) is not int or binding["capture_gid"] < 0
                    or type(binding.get("source_vertex_id")) is not int
                    or binding["source_vertex_id"] not in endpoints):
                return False
            pair = (binding["source_vertex_id"], binding["capture_gid"])
            if pair in observed:
                return False
            observed.add(pair)
        if observed != expected_bindings:
            return False
    return value["first"]["identity"] != value["second"]["identity"]

def _source_face_lineage_chain(record):
    chain, seen = [], set()
    current = record
    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        parent = current.get("parent_row_face_lineage")
        if not isinstance(parent, dict):
            parent = current.get("parent_054_face_ref")
        current = parent
    return chain


def _source_face_lineage_matches(record, expected):
    return any(all(row.get(key) == value for key, value in expected.items())
               for row in _source_face_lineage_chain(record))


def _source_patch_ancestry_matches(record, expected):
    """Match a source patch frame through a linked generated-face ancestry chain."""
    chain = _source_face_lineage_chain(record)
    if not chain:
        return False
    member = expected.get("source_member")
    member_sha = expected.get("source_member_sha256")
    if (not any(row.get("source_member") == member for row in chain)
            or not any(row.get("source_member") == member
                       and row.get("source_member_sha256") == member_sha for row in chain)):
        return False
    required_frame = {key: value for key, value in expected.items()
                      if key not in {"source_member_sha256", "owner_stable_id"}}
    frame_found = False
    for row in chain:
        if all(row.get(key) == value for key, value in required_frame.items()):
            frame_found = True
            break
    if not frame_found:
        return False
    owner = expected.get("owner_stable_id")
    return owner is None or any(row.get("owner_stable_id") == owner for row in chain)


def _exact_source_corner_ids_from_lineage(
        record, source_vertex_ids, *, expected_source_face=None,
        expected_source_member=None, expected_source_member_sha256=None,
        expected_source_row_face=None, expected_source_row_payload_sha256=None):
    chain = _source_face_lineage_chain(record)
    expected = None
    if expected_source_face is not None:
        expected = {
            "source_member": expected_source_member,
            "source_member_sha256": expected_source_member_sha256,
            "source_face": expected_source_face,
        }
        if expected_source_row_face is not None:
            expected["source_row_face"] = expected_source_row_face
        if expected_source_row_payload_sha256 is not None:
            expected["source_row_payload_sha256"] = expected_source_row_payload_sha256
        if not _source_patch_ancestry_matches(record, expected):
            return None
    for row_index, row in enumerate(chain):
        bary = row.get("corner_barycentric_exact")
        if not isinstance(bary, list) or len(bary) != 3:
            continue
        if expected is not None:
            linked_expected_frame = any(
                all(candidate.get(key) == value for key, value in expected.items()
                    if key != "source_member_sha256")
                and (candidate is row or row.get("source_face") == candidate.get("output_face"))
                for candidate in chain[row_index:])
            if not linked_expected_frame:
                continue
        result = []
        valid = True
        for weights in bary:
            if not isinstance(weights, list) or len(weights) != 3:
                valid = False
                break
            values = []
            for pair in weights:
                if (not isinstance(pair, list) or len(pair) != 2
                        or not all(isinstance(part, str) for part in pair)):
                    valid = False
                    break
                try:
                    values.append(Fraction(int(pair[0]), int(pair[1])))
                except (ValueError, ZeroDivisionError):
                    valid = False
                    break
            if not valid or sum(values, Fraction(0)) != 1:
                valid = False
                break
            ones = [index for index, value in enumerate(values) if value == 1]
            if len(ones) != 1 or any(value not in (0, 1) for value in values):
                valid = False
                break
            result.append(int(source_vertex_ids[ones[0]]))
        if valid and len(set(result)) == 3:
            return result
    return None


def _shared_source_crosswalk_point_contact(
        first, second, first_face, second_face, feature_gid_pairs,
        source_point_crosswalk, source_seam_face_pairs,
        source_seam_endpoint_source_vertex_id):
    """Authenticate one exact point at a pinned reciprocal source-seam endpoint."""
    if (not isinstance(feature_gid_pairs, (list, tuple)) or len(feature_gid_pairs) != 1
            or not isinstance(feature_gid_pairs[0], (list, tuple))
            or len(feature_gid_pairs[0]) != 2
            or any(type(gid) is not int for gid in feature_gid_pairs[0])):
        return False, "crosswalk_feature_must_be_one_typed_vertex_pair"
    if type(source_seam_endpoint_source_vertex_id) is not int:
        return False, "source_seam_endpoint_id_must_be_a_typed_integer"
    if not _valid_source_point_crosswalk(source_point_crosswalk):
        return False, "source_point_crosswalk_is_missing_or_invalid"
    if (first.rigid_owner_proof is None or second.rigid_owner_proof is None
            or first.rigid_owner_proof != second.rigid_owner_proof):
        return False, "accepted_rigid_owner_pose_proofs_differ_or_are_missing"
    if (first.source_face_lineage is None or second.source_face_lineage is None
            or first.source_vertex_lineage is None or second.source_vertex_lineage is None):
        return False, "source_lineage_is_missing"
    if (not isinstance(first_face, int) or isinstance(first_face, bool)
            or not isinstance(second_face, int) or isinstance(second_face, bool)
            or not 0 <= first_face < len(first.faces)
            or not 0 <= second_face < len(second.faces)):
        return False, "incident_face_is_out_of_range"
    if not isinstance(source_seam_face_pairs, (list, tuple)) or len(source_seam_face_pairs) != 2:
        return False, "both_reciprocal_source_seam_face_pairs_are_required"

    seam = source_point_crosswalk["seam"]
    endpoint = source_seam_endpoint_source_vertex_id
    if endpoint not in seam["endpoint_source_vertex_ids"]:
        return False, "source_seam_endpoint_is_not_in_authenticated_face_pair"
    sides = (source_point_crosswalk["first"], source_point_crosswalk["second"])
    surfaces = (first, second)
    incident_faces = (first_face, second_face)
    feature_gids = tuple(feature_gid_pairs[0])
    if (sides[0]["identity"] != list(first.key)
            or sides[1]["identity"] != list(second.key)):
        return False, "crosswalk_surface_identity_differs"

    pairs = []
    for pair in source_seam_face_pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(face) is not int for face in pair)):
            return False, "source_seam_face_pair_is_invalid"
        if (not 0 <= pair[0] < len(first.faces) or not 0 <= pair[1] < len(second.faces)):
            return False, "source_seam_face_pair_is_out_of_range"
        pairs.append(tuple(pair))
    if len(set(pairs)) != 2:
        return False, "duplicate_source_seam_face_pair"

    for pair_index, (source_face, pair) in enumerate(zip(seam["source_faces"], pairs)):
        rows = (sides[0]["seam_faces"][pair_index], sides[1]["seam_faces"][pair_index])
        if (rows[0]["face_index"], rows[1]["face_index"]) != pair:
            return False, "declared_face_pairs_do_not_match_ordered_seam_rows"
        for side, surface, row, face_index in zip(sides, surfaces, rows, pair):
            lineage = surface.source_face_lineage[face_index]
            if not _has_source_face_lineage(lineage):
                return False, "seam_face_has_no_source_lineage"
            expected = {
                "source_member": seam["source_member"],
                "source_member_sha256": seam["source_member_sha256"],
                "source_face": source_face["source_face"],
                "source_row_face": row["source_row_face"],
                "source_row_payload_sha256": row["source_row_payload_sha256"],
            }
            if not _source_face_lineage_matches(lineage, expected):
                return False, "seam_face_ancestry_does_not_match_authenticated_source"
            actual_gids = [int(surface.source_vertex_ids[int(local)])
                           for local in surface.faces[face_index]]
            if actual_gids != row["ordered_capture_gids"]:
                return False, "ordered_capture_GIDs_do_not_match_the_authenticated_source_face_map"

    for surface, side, face_index, gid in zip(surfaces, sides, incident_faces, feature_gids):
        if surface.vertex_index(gid) is None:
            return False, "crosswalk_vertex_is_absent_from_prepared_surface"
        incident_gids = [int(surface.source_vertex_ids[int(local)])
                         for local in surface.faces[face_index]]
        if gid not in incident_gids:
            return False, "feature_vertex_is_not_incident_to_contact_face"
        bindings = {(row["source_vertex_id"], row["capture_gid"])
                    for row in side["vertex_bindings"]}
        if (endpoint, gid) not in bindings:
            return False, "feature_GID_lacks_authenticated_endpoint_binding"

        incident_source_corner = None
        for source_face, row in zip(seam["source_faces"], side["seam_faces"]):
            if row["face_index"] == face_index:
                incident_source_corner = dict(zip(
                    row["ordered_capture_gids"], row["ordered_source_vertex_ids"]))
                break
        if incident_source_corner is None:
            lineage = surface.source_face_lineage[face_index]
            for source_face in seam["source_faces"]:
                expected = {
                    "source_member": seam["source_member"],
                    "source_member_sha256": seam["source_member_sha256"],
                    "source_face": source_face["source_face"],
                }
                if _source_face_lineage_matches(lineage, expected):
                    recovered = _exact_source_corner_ids_from_lineage(
                        lineage, source_face["source_vertex_ids"])
                    if recovered is not None:
                        incident_source_corner = dict(zip(incident_gids, recovered))
                        break
        if (incident_source_corner is not None
                and incident_source_corner.get(gid) != endpoint):
            return False, "incident_face_corner_does_not_map_to_authenticated_source_endpoint"
        if (incident_source_corner is None
                and {source_id for source_id, bound_gid in bindings if bound_gid == gid}
                    != {endpoint}):
            return False, "ambiguous_endpoint_GID_lacks_incident_source_corner_ancestry"

    p0 = first.positions[first.vertex_index(feature_gids[0])]
    p1 = second.positions[second.vertex_index(feature_gids[1])]
    if ci.float32_point_lattice_key(p0.tolist()) != ci.float32_point_lattice_key(p1.tolist()):
        return False, "crosswalk_vertex_emitted_float32_points_differ"
    expected_point = tuple(Fraction(value, 1 << 149)
                           for value in ci.float32_point_lattice_key(p0.tolist()))
    if first.intersection_points_meters(second, first_face, second_face) != (expected_point,):
        return False, "exact_intersection_exceeds_authenticated_source_seam_endpoint"
    return True, "exact_singleton_at_authenticated_two_face_source_seam_endpoint"

def _shared_feature_contact(first, second, first_face, second_face, feature_gids):
    """Fail-closed proof that a cross-surface hit is only a supported shared GID feature."""
    if len(feature_gids) not in (1, 2) or len(set(feature_gids)) != len(feature_gids):
        return False, "feature_must_be_one_vertex_or_one_edge"
    if any(type(gid) is not int for gid in feature_gids):
        return False, "feature_GIDs_must_be_integers"
    if (first.source_face_lineage is None or second.source_face_lineage is None
            or first.source_vertex_lineage is None or second.source_vertex_lineage is None
            or first.core_weights_by_gid is None or second.core_weights_by_gid is None
            or first.ordered_core_body_indices is None or second.ordered_core_body_indices is None):
        return False, "missing_source_or_actual_owner_material_proof"
    if first.declared_core_body_indices != second.declared_core_body_indices:
        return False, "material_inventory_policies_differ"
    if first.ordered_core_body_indices != second.ordered_core_body_indices:
        return False, "ordered_core_owner_lists_differ"
    if not _has_source_face_lineage(first.source_face_lineage[first_face]):
        return False, "first_face_has_no_source_lineage"
    if not _has_source_face_lineage(second.source_face_lineage[second_face]):
        return False, "second_face_has_no_source_lineage"
    face_a = [int(first.source_vertex_ids[int(i)]) for i in first.faces[first_face]]
    face_b = [int(second.source_vertex_ids[int(i)]) for i in second.faces[second_face]]
    if not set(feature_gids).issubset(face_a) or not set(feature_gids).issubset(face_b):
        return False, "declared_feature_GIDs_not_shared_by_both_faces"
    expected = set()
    for gid in feature_gids:
        if gid not in first.source_vertex_lineage or gid not in second.source_vertex_lineage:
            return False, "feature_GID_has_no_birth_or_alias_lineage"
        lineage_a = first.source_vertex_lineage[gid]
        lineage_b = second.source_vertex_lineage[gid]
        if not _has_source_vertex_lineage(lineage_a) or not _has_source_vertex_lineage(lineage_b):
            return False, "feature_GID_lineage_has_no_source_support"
        if json.dumps(lineage_a, sort_keys=True, separators=(",", ":")) != json.dumps(
                lineage_b, sort_keys=True, separators=(",", ":")):
            return False, "feature_GID_birth_lineage_differs_between_owners"
        ix_a = first.vertex_index(gid)
        ix_b = second.vertex_index(gid)
        pa, pb = first.positions[ix_a], second.positions[ix_b]
        if pa.tobytes() != pb.tobytes():
            return False, "feature_GID_emitted_Float32_positions_differ"
        wa = first.core_weights_by_gid.get(gid)
        wb = second.core_weights_by_gid.get(gid)
        if wa is None or wb is None:
            return False, "feature_GID_has_no_actual_ordered_21_core_row"
        invalid_support = np.any(wa < 0) if first.declared_core_body_indices is not None else np.any(wa <= 0)
        if (wa.dtype != np.dtype("float64") or wb.dtype != np.dtype("float64")
                or wa.shape != (len(first.ordered_core_body_indices),)
                or wb.shape != wa.shape or not np.isfinite(wa).all() or not np.isfinite(wb).all()
                or not np.array_equal(wa, wb) or invalid_support
                or abs(float(wa.sum()) - 1.0) > 1e-12):
            return False, "feature_GID_ordered_21_core_rows_differ_or_are_invalid"
        for lineage, owner_weights in ((lineage_a, wa), (lineage_b, wb)):
            if lineage.get("material_mapping_kind") == 3:
                try:
                    lineage_bodies = tuple(lineage["core_body_indices"])
                    lineage_weights = np.asarray(lineage["core_weights"], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    return False, "kind3_feature_lineage_lacks_ordered_owner_rows"
                if (lineage.get("material_binding") != "independent_existing_core_carrier"
                        or lineage_bodies != first.ordered_core_body_indices
                        or lineage_weights.shape != (len(first.ordered_core_body_indices),)
                        or not np.array_equal(lineage_weights, owner_weights)):
                    return False, "kind3_feature_lineage_differs_from_actual_owner_row"
        expected.add(tuple(Fraction(int(value)) for value in ci.float32_point_lattice_key(pa.tolist())))
    record_a = first.face_record(first_face)
    record_b = second.face_record(second_face)
    if record_a is None or record_b is None:
        return False, "face_record_missing"
    actual = set(ci.triangle_intersection_points(record_a[0], record_b[0]))
    if actual != expected:
        return False, "exact_intersection_exceeds_declared_shared_feature"
    return True, "exact_shared_GID_feature_with_source_and_owner_proof"


def _canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _surface_source_hashes(surface):
    gids = np.asarray(surface.source_vertex_ids)
    if gids.ndim != 1 or gids.dtype.kind not in "iu":
        raise ValueError("source-bound overlap requires integer source GIDs")
    canonical_gids = np.ascontiguousarray(gids.astype("<u8", copy=False))
    return {
        "positions_sha256": hashlib.sha256(surface.positions.tobytes()).hexdigest(),
        "faces_sha256": hashlib.sha256(surface.faces.tobytes()).hexdigest(),
        "gids_sha256": hashlib.sha256(canonical_gids.tobytes()).hexdigest(),
    }


def _parse_source_rational_xyz(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + " must be a three-coordinate rational point")
    result = []
    for coordinate in value:
        if not isinstance(coordinate, str):
            raise TypeError(label + " coordinates must be exact rational strings")
        try:
            result.append(Fraction(coordinate))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(label + " contains an invalid rational coordinate") from exc
    return tuple(result)


def _proof_surface_descriptor(proof, key):
    descriptors = proof.get("surfaces")
    if not isinstance(descriptors, dict):
        raise TypeError("source proof has no surface descriptors")
    descriptor = descriptors.get(str(tuple(key)))
    if not isinstance(descriptor, dict):
        raise TypeError("source proof lacks a typed surface descriptor")
    return descriptor


def audit_source_bound_connected_overlap(
        first, second, *, source_proof_bytes, expected_source_proof_sha256,
        source_binding, expected_source_binding_sha256, expected_source_map_sha256):
    """Verify one source-proven passive overlap without waiving generic containment.

    ``source_proof_bytes`` is the frozen pair report. ``source_binding`` is the
    caller's already-verified source-map binding for these exact prepared rows;
    its hashes are compared to both the report and the actual surfaces here.
    The function independently rechecks exact intersections and the reported
    binary64 interior witness with the existing P6 predicates. It accepts only
    the scoped connected-overlap conclusion; it makes no flow or clinical claim.
    """
    if not isinstance(first, PreparedSurface) or not isinstance(second, PreparedSurface):
        raise TypeError("source-bound overlap requires prepared surfaces")
    first_key, second_key = typed_key(first.key), typed_key(second.key)
    if first_key == second_key:
        raise ValueError("source-bound overlap requires two distinct typed identities")
    if not isinstance(source_proof_bytes, bytes):
        raise TypeError("source proof must be frozen bytes")
    if (not isinstance(expected_source_proof_sha256, str)
            or len(expected_source_proof_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_source_proof_sha256)):
        raise ValueError("source proof SHA-256 pin is invalid")
    proof_sha256 = hashlib.sha256(source_proof_bytes).hexdigest()
    if proof_sha256 != expected_source_proof_sha256:
        raise ValueError("source proof bytes do not match their SHA-256 pin")
    if (not isinstance(expected_source_map_sha256, str)
            or len(expected_source_map_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_source_map_sha256)):
        raise ValueError("source map SHA-256 pin is invalid")
    if not isinstance(source_binding, dict):
        raise TypeError("verified source binding must be a mapping")
    if (not isinstance(expected_source_binding_sha256, str)
            or len(expected_source_binding_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_source_binding_sha256)
            or _canonical_json_sha256(source_binding) != expected_source_binding_sha256):
        raise ValueError("source binding bytes do not match their SHA-256 pin")
    if (source_binding.get("source_proof_sha256") != proof_sha256
            or source_binding.get("source_map_sha256") != expected_source_map_sha256):
        raise ValueError("source binding is not pinned to the supplied proof and map")
    try:
        proof = json.loads(source_proof_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source proof is not valid UTF-8 JSON") from exc
    if (not isinstance(proof, dict)
            or proof.get("schema") != "numi.human.head-csf-pair-continuity.v1"
            or proof.get("status") != "complete"
            or proof.get("not_admission") is not True):
        raise ValueError("source proof is not the expected frozen CSF pair report")
    expected_scope = (
        "only captured typed rows (51010,323) fourth ventricle and "
        "(51010,326) cerebral aqueduct; no broad target scan and no native admission")
    if (source_binding.get("surface_identities") != [list(first_key), list(second_key)]
            or proof.get("scope") != expected_scope
            or (first_key, second_key) != ((51010, 323), (51010, 326))):
        raise ValueError("source proof and source binding identities or scope differ")
    binding_rows = source_binding.get("surfaces")
    if not isinstance(binding_rows, list) or len(binding_rows) != 2:
        raise ValueError("source binding must contain exactly two surface rows")
    bound = {}
    surfaces_by_key = {first_key: first, second_key: second}
    for row in binding_rows:
        if not isinstance(row, dict):
            raise TypeError("source binding surface row must be a mapping")
        key = typed_key(row.get("identity"))
        if key not in surfaces_by_key:
            raise ValueError("source binding includes an unrelated surface identity")
        if key in bound:
            raise ValueError("duplicate typed identity in source binding")
        if row.get("identity") != list(key):
            raise ValueError("source binding identity must use canonical integer list")
        actual = _surface_source_hashes(surfaces_by_key[key])
        bound[key] = row
        for hash_name, actual_hash in actual.items():
            value = row.get(hash_name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)
                    or value != actual_hash):
                raise ValueError("source binding does not match prepared surface " + hash_name)
        for hash_name in ("source_face_lineage_sha256", "source_vertex_lineage_sha256"):
            value = row.get(hash_name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError("verified source binding lacks " + hash_name)
    if set(bound) != {first_key, second_key}:
        raise ValueError("source binding omits one typed surface")
    for surface in (first, second):
        descriptor = _proof_surface_descriptor(proof, surface.key)
        actual = _surface_source_hashes(surface)
        if (descriptor.get("position_sha256") != actual["positions_sha256"]
                or descriptor.get("face_sha256") != actual["faces_sha256"]):
            raise ValueError("frozen source proof geometry does not match the prepared surface")
        if not surface.closed or descriptor.get("closed") is not True:
            return {"passed": False, "reason": "surface_not_closed",
                    "identity": list(surface.key)}
        self_result = surface.self_audit()
        if (type(descriptor.get("exact_self_count")) is not int
                or descriptor.get("exact_self_count") != 0
                or self_result.get("count") != 0):
            return {"passed": False, "reason": "surface_self_intersection",
                    "identity": list(surface.key), "self_audit": self_result}
        if surface.quotient_topology.get("face_component_count") != 1:
            return {"passed": False, "reason": "surface_not_single_component",
                    "identity": list(surface.key),
                    "face_component_count": surface.quotient_topology.get("face_component_count")}
    connection = proof.get("closed_solid_connection")
    exact_gate = proof.get("exact_pair_gate")
    witnesses = proof.get("contact_witnesses")
    if not isinstance(connection, dict) or not isinstance(exact_gate, dict) or not isinstance(witnesses, list):
        raise TypeError("source proof lacks exact contact and solid witness records")
    inside = connection.get("exact_inside_both_witness")
    if not isinstance(inside, dict) or inside.get("method") != "intersection_volume_centroid":
        raise ValueError("source proof lacks the declared exact inside-both witness")
    inside_point = inside.get("point_world_m")
    if (not isinstance(inside_point, list) or len(inside_point) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in inside_point)):
        raise ValueError("source inside witness must be a finite binary64 point")
    exact_inside_point = tuple(Fraction.from_float(float(value)) for value in inside_point)
    first_inside = ci.point_location_prepared_indexed(exact_inside_point, first.meter_index())
    second_inside = ci.point_location_prepared_indexed(exact_inside_point, second.meter_index())
    if first_inside.get("location") != "inside" or second_inside.get("location") != "inside":
        return {"passed": False, "reason": "frozen_overlap_witness_not_inside_both",
                "point_m": [str(value) for value in exact_inside_point],
                "first_location": first_inside.get("location"),
                "second_location": second_inside.get("location")}
    if (inside.get("first_location") != "inside" or inside.get("second_location") != "inside"):
        raise ValueError("source proof does not describe an exact inside-both witness")
    expected_pairs_raw = exact_gate.get("exact_face_pairs")
    if not isinstance(expected_pairs_raw, list):
        raise TypeError("source proof exact face-pair set is missing")
    expected_pairs = []
    for pair in expected_pairs_raw:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(face) is not int or face < 0 for face in pair)):
            raise ValueError("source proof face pair must contain nonnegative integer indices")
        expected_pairs.append(tuple(pair))
    if len(set(expected_pairs)) != len(expected_pairs):
        raise ValueError("source proof contains duplicate exact face pairs")
    expected_contacts = {}
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise TypeError("source contact witness must be a mapping")
        pair_raw = witness.get("face_pair")
        if (not isinstance(pair_raw, list) or len(pair_raw) != 2
                or any(type(face) is not int or face < 0 for face in pair_raw)):
            raise ValueError("source contact witness face pair is invalid")
        pair = tuple(pair_raw)
        if pair in expected_contacts:
            raise ValueError("source proof repeats an exact contact pair")
        points = witness.get("points_world_m")
        if not isinstance(points, list) or not points:
            raise ValueError("source contact witness has no exact rational points")
        parsed = tuple(sorted({_parse_source_rational_xyz(point, "source contact point")
                               for point in points}))
        if len(parsed) != len(points):
            raise ValueError("source proof repeats a rational contact point")
        expected_contacts[pair] = parsed
    if set(expected_contacts) != set(expected_pairs):
        raise ValueError("source proof face-pair and contact-witness sets differ")
    exact_result = ci._audit_pair_prepared_first(first.index, second.records, same_surface=False)
    observed_pairs = tuple(sorted(tuple(pair) for pair in exact_result["triangle_pairs"]))
    if set(observed_pairs) != set(expected_pairs):
        return {"passed": False, "reason": "observed_exact_face_pairs_differ_from_source_proof",
                "expected_pair_count": len(expected_pairs),
                "observed_pair_count": len(observed_pairs),
                "unexpected_pairs": [list(pair) for pair in sorted(set(observed_pairs) - set(expected_pairs))],
                "missing_pairs": [list(pair) for pair in sorted(set(expected_pairs) - set(observed_pairs))]}
    for first_face, second_face in observed_pairs:
        actual_points = first.intersection_points_meters(second, first_face, second_face)
        if actual_points != expected_contacts[(first_face, second_face)]:
            return {"passed": False, "reason": "exact_contact_points_differ_from_source_proof",
                    "face_pair": [first_face, second_face],
                    "expected_points_m": [[str(v) for v in p]
                                           for p in expected_contacts[(first_face, second_face)]],
                    "observed_points_m": [[str(v) for v in p] for p in actual_points]}
    return {
        "passed": True,
        "scope": "source-bound passive geometric overlap only",
        "identities": [list(first_key), list(second_key)],
        "source_proof_sha256": proof_sha256,
        "source_map_sha256": expected_source_map_sha256,
        "source_binding_sha256": expected_source_binding_sha256,
        "surface_bindings": [bound[key] for key in (first_key, second_key)],
        "surface_checks": [
            {"identity": list(surface.key), "closed": True, "self_intersections": 0,
             "connected_face_components": 1}
            for surface in (first, second)],
        "exact_face_pair_count": len(observed_pairs),
        "exact_rational_contact_pair_set_matches_source_proof": True,
        "exact_rational_contact_points_match_source_proof": True,
        "exact_inside_both_witness": {
            "point_m": [str(value) for value in exact_inside_point],
            "first_location": first_inside["location"],
            "second_location": second_inside["location"]},
        "connected_union_reason": (
            "Both closed embedded surfaces are each one connected component, and the "
            "rechecked exact point is in the interior of both; their union is connected."),
        "claim_limit": "No CSF flow, caliber, pressure, clinical, or native-motion claim.",
    }


_SOURCE_PATCH_COMPLEX_SCHEMA = "numi.human.reciprocal-source-patch-complex.v1"
_SOURCE_PATCH_FRAGMENT_COMPLEX_SCHEMA = "numi.human.reciprocal-source-patch-fragment-complex.v1"


def _source_patch_lineage_hashes(surface):
    if surface.source_face_lineage is None or surface.source_vertex_lineage is None:
        raise ValueError("source patch complex requires complete face and vertex ancestry")
    face_sha = _canonical_json_sha256(list(surface.source_face_lineage))
    vertices = [[int(gid), lineage]
                for gid, lineage in sorted(surface.source_vertex_lineage.items(),
                                           key=lambda item: int(item[0]))]
    vertex_sha = _canonical_json_sha256(vertices)
    return {"source_face_lineage_sha256": face_sha,
            "source_vertex_lineage_sha256": vertex_sha}


def _source_patch_surface_hashes(surface):
    return {**_surface_source_hashes(surface), **_source_patch_lineage_hashes(surface)}


def _source_patch_edges(source_faces):
    edges = set()
    for row in source_faces:
        ids = row["source_vertex_ids"]
        for index in range(3):
            edges.add(tuple(sorted((ids[index], ids[(index + 1) % 3]))))
    return edges




def _patch_fraction(value, *, hexadecimal=False):
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(item, str) for item in value)):
        raise ValueError("source patch rational must be a numerator/denominator pair")
    def integer(text):
        return int(text, 16) if hexadecimal or text.startswith(("0x", "-0x")) else int(text, 10)
    numerator, denominator = integer(value[0]), integer(value[1])
    if denominator <= 0:
        raise ValueError("source patch rational denominator must be positive")
    return Fraction(numerator, denominator)


def _patch_float32_bits(point):
    return [int(value) for value in np.asarray(point, dtype=np.float32).view(np.uint32)]


def _patch_round_fraction_float32_bits(value):
    """Return IEEE-754 binary32 round-to-nearest-even bits for an exact rational."""
    value = Fraction(value)
    sign = 0x80000000 if value < 0 else 0
    numerator, denominator = abs(value.numerator), value.denominator
    if numerator == 0:
        return sign

    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif (numerator << -exponent) < denominator:
        exponent -= 1

    def rounded_quotient(top, bottom):
        quotient, remainder = divmod(top, bottom)
        twice = remainder << 1
        if twice > bottom or (twice == bottom and quotient & 1):
            quotient += 1
        return quotient

    if exponent < -126:
        significand = rounded_quotient(numerator << 149, denominator)
        if significand == 0:
            return sign
        if significand >= (1 << 23):
            return sign | (1 << 23)
        return sign | significand

    shift = 23 - exponent
    if shift >= 0:
        significand = rounded_quotient(numerator << shift, denominator)
    else:
        significand = rounded_quotient(numerator, denominator << -shift)
    if significand == (1 << 24):
        significand >>= 1
        exponent += 1
    if exponent > 127:
        raise ValueError("source patch rational rounds outside finite Float32 range")
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def _patch_float32_exact_fraction(bits):
    """Decode finite IEEE-754 binary32 bits as an exact Fraction."""
    if type(bits) is not int or not 0 <= bits <= 0xffffffff:
        raise ValueError("source patch Float32 bits are malformed")
    sign = -1 if bits & 0x80000000 else 1
    exponent = (bits >> 23) & 0xff
    significand = bits & 0x7fffff
    if exponent == 0xff:
        raise ValueError("source patch Float32 bits are not finite")
    if exponent == 0:
        power = -149
    else:
        significand |= 1 << 23
        power = exponent - 127 - 23
    value = Fraction(sign * significand, 1)
    return value * (1 << power) if power >= 0 else value / (1 << -power)


def _patch_rational_float32_bits(point):
    return [_patch_round_fraction_float32_bits(value) for value in point]


def _patch_barycentric(triangle, point):
    origin = triangle[0]
    first = tuple(triangle[1][i] - origin[i] for i in range(3))
    second = tuple(triangle[2][i] - origin[i] for i in range(3))
    delta = tuple(point[i] - origin[i] for i in range(3))
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        determinant = first[axis_a] * second[axis_b] - first[axis_b] * second[axis_a]
        if determinant:
            weight_b = (delta[axis_a] * second[axis_b] - delta[axis_b] * second[axis_a]) / determinant
            weight_c = (first[axis_a] * delta[axis_b] - first[axis_b] * delta[axis_a]) / determinant
            return (Fraction(1) - weight_b - weight_c, weight_b, weight_c)
    raise ValueError("source patch parent triangle is degenerate")


def _prepare_source_patch_fragment_complex(first, second, value):
    """Validate a subdivided reciprocal patch through exact parent-face barycentrics."""
    fields = {"schema", "source_patch_complex_sha256", "identities", "source_member",
              "source_member_sha256", "input_pins", "points", "source_faces", "first", "second"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("source patch fragment-complex fields are malformed")
    digest = value.get("source_patch_complex_sha256")
    payload = {key: item for key, item in value.items() if key != "source_patch_complex_sha256"}
    if (value.get("schema") != _SOURCE_PATCH_FRAGMENT_COMPLEX_SCHEMA
            or not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or digest != _canonical_json_sha256(payload)):
        raise ValueError("source patch fragment-complex seal is invalid")
    if value.get("identities") != [list(first.key), list(second.key)]:
        raise ValueError("source patch fragment-complex ordered identities differ")
    member, member_sha = value.get("source_member"), value.get("source_member_sha256")
    if (not isinstance(member, str) or not member or not isinstance(member_sha, str)
            or len(member_sha) != 64 or any(ch not in "0123456789abcdef" for ch in member_sha)):
        raise ValueError("source patch fragment-complex source member pin is invalid")
    pins = value.get("input_pins")
    pin_names = {"current_payload_sha256", "parent_payload_sha256", "intermediate_payload_sha256",
                 "arrangement_sha256", "arrangement_certificate_sha256", "ancestry_first_sha256",
                 "ancestry_second_sha256", "capture_pack_sha256", "capture_receipt_sha256",
                 "source_obj_sha256"}
    if not isinstance(pins, dict) or set(pins) != pin_names:
        raise ValueError("source patch fragment-complex input pin fields differ")
    for item in pins.values():
        if (not isinstance(item, str) or len(item) != 64
                or any(ch not in "0123456789abcdef" for ch in item)):
            raise ValueError("source patch fragment-complex input SHA is invalid")
    if (first.rigid_owner_proof is None or second.rigid_owner_proof is None
            or first.rigid_owner_proof != second.rigid_owner_proof):
        raise ValueError("source patch fragments do not share the accepted rigid-owner proof")

    point_rows = value.get("points")
    if not isinstance(point_rows, list) or len(point_rows) != 4:
        raise ValueError("source patch fragment-complex must bind four generated points")
    point_exact, local_bits = {}, {}
    for row in point_rows:
        if not isinstance(row, dict) or set(row) != {"point_id", "local_xyz_exact", "local_float32_bits"}:
            raise ValueError("source patch generated point record is malformed")
        point_id, xyz, bits = row.get("point_id"), row.get("local_xyz_exact"), row.get("local_float32_bits")
        if (not isinstance(point_id, str) or not point_id or point_id in point_exact
                or not isinstance(xyz, list) or len(xyz) != 3
                or not isinstance(bits, list) or len(bits) != 3
                or any(type(bit) is not int or not 0 <= bit <= 0xffffffff for bit in bits)):
            raise ValueError("source patch generated point values are malformed")
        exact = tuple(_patch_fraction(pair) for pair in xyz)
        if _patch_float32_bits([float(component) for component in exact]) != bits:
            raise ValueError("source patch exact point does not quantize to pinned local Float32")
        point_exact[point_id] = exact
        local_bits[point_id] = tuple(bits)
    if len(set(local_bits.values())) != 4:
        raise ValueError("source patch generated local points collapse after Float32 quantization")

    source_faces = value.get("source_faces")
    if not isinstance(source_faces, list) or len(source_faces) != 2:
        raise ValueError("source patch must bind two authenticated parent faces")
    source_face_ids, parsed_sources, source_point_ids = [], [], set()
    for face in source_faces:
        required = {"source_face", "source_vertex_ids", "source_row_face",
                    "source_row_payload_sha256", "parent_ordered_source_vertex_ids",
                    "parent_vertices_exact", "arrangement_record_index",
                    "arrangement_record_sha256", "arrangement_record",
                    "arrangement_point_ids", "barycentric_exact"}
        if not isinstance(face, dict) or set(face) != required:
            raise ValueError("source patch authenticated parent face fields are malformed")
        face_id, source_ids = face.get("source_face"), face.get("source_vertex_ids")
        row_face, parent_ids = face.get("source_row_face"), face.get("parent_ordered_source_vertex_ids")
        if (type(face_id) is not int or face_id < 0 or type(row_face) is not int or row_face < 0
                or not isinstance(source_ids, list) or len(source_ids) != 3
                or any(type(item) is not int or item < 0 for item in source_ids)
                or len(set(source_ids)) != 3 or not isinstance(parent_ids, list) or len(parent_ids) != 3
                or any(type(item) is not int or item < 0 for item in parent_ids)
                or set(parent_ids) != set(source_ids)
                or not _cyclic_source_face_orientation(parent_ids, source_ids)):
            raise ValueError("source patch parent face identity or corner order is invalid")
        if face.get("source_row_payload_sha256") != pins["parent_payload_sha256"]:
            raise ValueError("source patch parent row payload differs from pinned source")
        parent_coords = face.get("parent_vertices_exact")
        if not isinstance(parent_coords, list) or len(parent_coords) != 3:
            raise ValueError("source patch parent source coordinates are malformed")
        parent_triangle = []
        for point in parent_coords:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError("source patch parent source point is malformed")
            parent_triangle.append(tuple(_patch_fraction(pair) for pair in point))
        record, record_index = face.get("arrangement_record"), face.get("arrangement_record_index")
        record_sha = face.get("arrangement_record_sha256")
        if (type(record_index) is not int or record_index < 0
                or not isinstance(record_sha, str) or len(record_sha) != 64
                or not isinstance(record, dict)
                or record_sha != _canonical_json_sha256(record)
                or record.get("source") != "medulla343"
                or record.get("source_face") != row_face
                or record.get("other_location") != "inside"):
            raise ValueError("source patch exact arrangement record is stale or misidentified")
        arrangement_vertices = record.get("vertices")
        point_ids, bary_rows = face.get("arrangement_point_ids"), face.get("barycentric_exact")
        if (not isinstance(arrangement_vertices, list) or len(arrangement_vertices) != 3
                or not isinstance(point_ids, list) or len(point_ids) != 3
                or not isinstance(bary_rows, list) or len(bary_rows) != 3):
            raise ValueError("source patch arrangement triangle is malformed")
        for point_id, encoded_xyz, encoded_bary in zip(point_ids, arrangement_vertices, bary_rows):
            if point_id not in point_exact:
                raise ValueError("source patch arrangement references an unknown point")
            if not isinstance(encoded_xyz, list) or len(encoded_xyz) != 3:
                raise ValueError("source patch arrangement vertex coordinates are malformed")
            xyz = tuple(_patch_fraction(pair, hexadecimal=True) for pair in encoded_xyz)
            if xyz != point_exact[point_id]:
                raise ValueError("source patch point differs from its exact arrangement coordinate")
            if not isinstance(encoded_bary, list) or len(encoded_bary) != 3:
                raise ValueError("source patch barycentric map is malformed")
            declared = tuple(_patch_fraction(pair) for pair in encoded_bary)
            calculated = _patch_barycentric(parent_triangle, xyz)
            if (declared != calculated or sum(declared, Fraction(0)) != 1
                    or any(weight < 0 for weight in declared)):
                raise ValueError("source patch nested exact barycentric map is invalid")
            rebuilt = tuple(sum((declared[i] * parent_triangle[i][axis]
                                 for i in range(3)), Fraction(0)) for axis in range(3))
            if rebuilt != xyz:
                raise ValueError("source patch barycentrics do not reconstruct exact source point")
            if point_id in source_point_ids and point_exact[point_id] != xyz:
                raise ValueError("shared source patch point differs between parent-face frames")
            source_point_ids.add(point_id)
        source_face_ids.append(face_id)
        parsed_sources.append((face, tuple(parent_triangle), tuple(point_ids)))
    if (source_face_ids[0] == source_face_ids[1]
            or len(set(source_faces[0]["source_vertex_ids"])
                   & set(source_faces[1]["source_vertex_ids"])) != 2
            or source_point_ids != set(point_exact)):
        raise ValueError("source patch parents or generated point coverage are not a two-face disk")

    sides = (value.get("first"), value.get("second"))
    surfaces = (first, second)
    side_fields = {"identity", "source_row_payload_sha256", "surface_hashes",
                   "point_bindings", "patch_faces", "support_faces"}
    hash_fields = {"positions_sha256", "faces_sha256", "gids_sha256",
                   "source_face_lineage_sha256", "source_vertex_lineage_sha256"}
    support_maps, point_maps, coordinate_maps, patch_face_maps = [], [], [], []
    for side_index, (side, surface) in enumerate(zip(sides, surfaces)):
        if not isinstance(side, dict) or set(side) != side_fields or side.get("identity") != list(surface.key):
            raise ValueError("source patch fragment side identity or fields differ")
        if side.get("source_row_payload_sha256") != pins["current_payload_sha256"]:
            raise ValueError("source patch side is not bound to pinned current payload")
        if surface.source_face_lineage is None or surface.source_vertex_lineage is None:
            raise ValueError("source patch fragments require face and vertex lineage arrays")
        hashes = side.get("surface_hashes")
        if not isinstance(hashes, dict) or set(hashes) != hash_fields:
            raise ValueError("source patch fragment surface hashes are malformed")
        if hashes != _source_patch_surface_hashes(surface):
            raise ValueError("source patch fragment surface or ancestry hash is stale")

        bindings = side.get("point_bindings")
        if not isinstance(bindings, list) or len(bindings) != 4:
            raise ValueError("source patch world point binding coverage is incomplete")
        point_map, coordinate_map = {}, {}
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {
                    "point_id", "capture_gids", "world_float32_bits"}:
                raise ValueError("source patch world point binding fields are malformed")
            point_id, gids, bits = binding.get("point_id"), binding.get("capture_gids"), binding.get("world_float32_bits")
            if (point_id not in point_exact or point_id in point_map
                    or not isinstance(gids, list) or not gids
                    or any(type(gid) is not int or gid < 0 for gid in gids)
                    or len(set(gids)) != len(gids)
                    or not isinstance(bits, list) or len(bits) != 3
                    or any(type(bit) is not int or not 0 <= bit <= 0xffffffff for bit in bits)):
                raise ValueError("source patch world point binding values are invalid")
            coords = set()
            for gid in gids:
                vi = surface.vertex_index(gid)
                if vi is None or _patch_float32_bits(surface.positions[vi]) != bits:
                    raise ValueError("source patch Float32 point differs from actual captured GID")
                coords.add(tuple(Fraction(item, 1 << 149)
                                 for item in ci.float32_point_lattice_key(surface.positions[vi].tolist())))
            if len(coords) != 1:
                raise ValueError("source patch coordinate alias GIDs disagree")
            point_map[point_id] = set(gids)
            coordinate_map[point_id] = next(iter(coords))
        if set(point_map) != set(point_exact) or len(set(coordinate_map.values())) != 4:
            raise ValueError("source patch world map does not uniquely cover four generated points")
        for point_id, point_gids in point_map.items():
            point_key = tuple(ci.float32_point_lattice_key(
                surface.positions[surface.vertex_index(gid)].tolist())
                for gid in sorted(point_gids))
            if len(set(point_key)) != 1:
                raise ValueError("source patch coordinate alias GIDs disagree")
            expected_aliases = {
                int(surface.source_vertex_ids[index])
                for index, position in enumerate(surface.positions)
                if tuple(ci.float32_point_lattice_key(position.tolist())) == point_key[0]
            }
            if point_gids != expected_aliases:
                raise ValueError("source patch world map omits or invents exact-coordinate alias GIDs")

        support_rows = side.get("support_faces")
        if not isinstance(support_rows, list) or not support_rows:
            raise ValueError("source patch incident support faces are missing")
        support_map = {}
        for support in support_rows:
            required = {"face_index", "source_member", "source_member_sha256", "source_face",
                        "source_row_face", "source_row_payload_sha256", "ordered_capture_gids"}
            if not isinstance(support, dict) or set(support) != required:
                raise ValueError("source patch support-face fields are malformed")
            fi, gids = support.get("face_index"), support.get("ordered_capture_gids")
            if (type(fi) is not int or not 0 <= fi < len(surface.faces) or fi in support_map
                    or not isinstance(support.get("source_member"), str) or not support["source_member"]
                    or not isinstance(support.get("source_member_sha256"), str)
                    or len(support["source_member_sha256"]) != 64
                    or any(ch not in "0123456789abcdef" for ch in support["source_member_sha256"])
                    or type(support.get("source_face")) is not int or support["source_face"] < 0
                    or type(support.get("source_row_face")) is not int or support["source_row_face"] < 0
                    or support.get("source_row_payload_sha256") != pins["parent_payload_sha256"]
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(gid) is not int or gid < 0 for gid in gids) or len(set(gids)) != 3):
                raise ValueError("source patch support-face values are invalid")
            actual = [int(surface.source_vertex_ids[int(local)]) for local in surface.faces[fi]]
            if actual != gids:
                raise ValueError("source patch support GIDs differ from actual face")
            expected = {key: support[key] for key in (
                "source_member", "source_member_sha256", "source_face", "source_row_face",
                "source_row_payload_sha256")}
            expected["owner_stable_id"] = surface.key[1]
            if not _source_patch_ancestry_matches(surface.source_face_lineage[fi], expected):
                raise ValueError("source patch support ancestry differs from pinned parent frame")
            support_map[fi] = support
        lattice = {tuple(ci.float32_point_lattice_key(
            surface.positions[surface.vertex_index(gid)].tolist()))
                   for gids in point_map.values() for gid in gids}
        actual_support = {fi for fi, face in enumerate(surface.faces)
                          if any(tuple(ci.float32_point_lattice_key(
                              surface.positions[int(local)].tolist())) in lattice for local in face)}
        if set(support_map) != actual_support:
            raise ValueError("source patch support does not cover exact coordinate-alias incident stars")

        patch_rows = side.get("patch_faces")
        if not isinstance(patch_rows, list) or len(patch_rows) != 2:
            raise ValueError("source patch side must bind both generated face fragments")
        patch_map = {}
        for source, patch in zip(parsed_sources, patch_rows):
            source_face, row_face, arrangement_point_ids = (
                source[0]["source_face"], source[0]["source_row_face"], source[2])
            required = {"face_index", "source_face", "source_row_face", "lineage_face_id",
                        "ordered_point_ids", "ordered_capture_gids"}
            if not isinstance(patch, dict) or set(patch) != required:
                raise ValueError("source patch generated-face fields are malformed")
            fi, point_ids, gids = patch.get("face_index"), patch.get("ordered_point_ids"), patch.get("ordered_capture_gids")
            if (type(fi) is not int or fi not in support_map or fi in patch_map
                    or patch.get("source_face") != source_face or patch.get("source_row_face") != row_face
                    or type(patch.get("lineage_face_id")) is not int or patch["lineage_face_id"] < 0
                    or not isinstance(point_ids, list) or len(point_ids) != 3
                    or set(point_ids) != set(arrangement_point_ids)
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(gid) is not int for gid in gids)):
                raise ValueError("source patch generated-face mapping is invalid")
            if not _cyclic_source_face_orientation(
                    point_ids, arrangement_point_ids, reversed_order=(side_index == 1)):
                raise ValueError("source patch generated-face winding is not reciprocal")
            actual = [int(surface.source_vertex_ids[int(local)]) for local in surface.faces[fi]]
            if actual != gids or any(gid not in point_map[pid] for gid, pid in zip(gids, point_ids)):
                raise ValueError("source patch generated-face point/GID mapping differs")
            expected = {"source_member": member, "source_member_sha256": member_sha,
                        "source_face": source_face, "source_row_face": row_face,
                        "source_row_payload_sha256": pins["parent_payload_sha256"],
                        "owner_stable_id": surface.key[1]}
            lineage = surface.source_face_lineage[fi]
            if not _source_patch_ancestry_matches(lineage, expected):
                raise ValueError("source patch generated face lacks authenticated source-face ancestry")
            if side_index == 0:
                identity = [[["1", "1"], ["0", "1"], ["0", "1"]],
                            [["0", "1"], ["1", "1"], ["0", "1"]],
                            [["0", "1"], ["0", "1"], ["1", "1"]]]
                if (lineage.get("operation") != "exact_arrangement_medulla_fragment_outside_csf"
                        or lineage.get("source_face") != patch["lineage_face_id"]
                        or lineage.get("corner_barycentric_exact") != identity):
                    raise ValueError("medulla fragment does not map exactly to its pinned intermediate face")
            elif (lineage.get("operation") != "reversed_medulla343_priority_interface"
                  or lineage.get("source_face") != source_face):
                raise ValueError("reciprocal pons fragment is not source-bound")
            patch_map[fi] = patch
        support_maps.append(support_map)
        point_maps.append(point_map)
        coordinate_maps.append(coordinate_map)
        patch_face_maps.append(patch_map)

    patch_face_pairs, patch_edges = [], set()
    for source in parsed_sources:
        source_face = source[0]["source_face"]
        rows = [next((item for item in face_map.values() if item["source_face"] == source_face), None)
                for face_map in patch_face_maps]
        if any(row is None for row in rows):
            raise ValueError("reciprocal source face is missing from one emitted region")
        ids_a, ids_b = rows[0]["ordered_point_ids"], rows[1]["ordered_point_ids"]
        if (set(ids_a) != set(ids_b)
                or not _cyclic_source_face_orientation(ids_a, ids_b, reversed_order=True)
                or any(coordinate_maps[0][pid] != coordinate_maps[1][pid] for pid in ids_a)):
            raise ValueError("reciprocal source patch fragments differ in actual-capture coordinates")
        patch_face_pairs.append((rows[0]["face_index"], rows[1]["face_index"], source_face))
        for i in range(3):
            patch_edges.add(tuple(sorted((ids_a[i], ids_a[(i + 1) % 3]))))
    return {"source_patch_complex_sha256": digest, "source_member": member,
            "source_member_sha256": member_sha,
            "source_faces": [{"source_face": row[0]["source_face"], "source_vertex_ids": row[2]}
                             for row in parsed_sources],
            "patch_vertices": tuple(point_exact), "patch_edges": patch_edges,
            "support_faces": (set(support_maps[0]), set(support_maps[1])),
            "vertex_maps": point_maps, "coordinate_maps": coordinate_maps,
            "patch_face_pairs": tuple(patch_face_pairs), "fragment_complex": True}


def _prepare_source_patch_complex(first, second, value):
    """Validate a source-bound reciprocal patch and its incident stars."""
    if isinstance(value, dict) and "carrier_faces" in value:
        return _prepare_source_patch_carrier_complex(first, second, value)
    if isinstance(value, dict) and value.get("schema") == _SOURCE_PATCH_FRAGMENT_COMPLEX_SCHEMA:
        return _prepare_source_patch_fragment_complex(first, second, value)
    fields = {"schema", "source_patch_complex_sha256", "identities",
              "source_member", "source_member_sha256", "source_faces",
              "patch_vertex_source_ids", "first", "second"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("source patch complex fields are malformed")
    if value.get("schema") != _SOURCE_PATCH_COMPLEX_SCHEMA:
        raise ValueError("source patch complex schema is unsupported")
    proof_sha = value.get("source_patch_complex_sha256")
    payload = {key: item for key, item in value.items()
               if key != "source_patch_complex_sha256"}
    if (not isinstance(proof_sha, str) or len(proof_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in proof_sha)
            or proof_sha != _canonical_json_sha256(payload)):
        raise ValueError("source patch complex digest is stale or malformed")
    identities = value.get("identities")
    if (not isinstance(identities, list) or len(identities) != 2
            or identities != [list(first.key), list(second.key)]):
        raise ValueError("source patch complex ordered identities differ")
    member, member_sha = value.get("source_member"), value.get("source_member_sha256")
    if (not isinstance(member, str) or not member
            or not isinstance(member_sha, str) or len(member_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in member_sha)):
        raise ValueError("source patch complex member identity is invalid")
    source_faces = value.get("source_faces")
    if not isinstance(source_faces, list) or len(source_faces) != 2:
        raise ValueError("source patch complex must contain exactly two reciprocal source faces")
    source_face_ids = []
    for row in source_faces:
        if not isinstance(row, dict) or set(row) != {"source_face", "source_vertex_ids"}:
            raise ValueError("source patch source-face record is malformed")
        face_id, ids = row.get("source_face"), row.get("source_vertex_ids")
        if (type(face_id) is not int or face_id < 0 or not isinstance(ids, list) or len(ids) != 3
                or any(type(item) is not int or item < 0 for item in ids)
                or len(set(ids)) != 3):
            raise ValueError("source patch source-face IDs are invalid")
        source_face_ids.append(face_id)
    if source_face_ids[0] == source_face_ids[1]:
        raise ValueError("source patch repeats a source face")
    common = set(source_faces[0]["source_vertex_ids"]) & set(source_faces[1]["source_vertex_ids"])
    patch_vertices = value.get("patch_vertex_source_ids")
    if (len(common) != 2 or not isinstance(patch_vertices, list)
            or any(type(item) is not int or item < 0 for item in patch_vertices)
            or len(set(patch_vertices)) != len(patch_vertices)
            or set(patch_vertices) != set().union(
                *(set(row["source_vertex_ids"]) for row in source_faces))):
        raise ValueError("source patch must be the four-corner complex of two faces sharing one edge")
    # The two source triangles must form a consistently oriented disk.
    shared_orientations = []
    for row in source_faces:
        ids = row["source_vertex_ids"]
        shared_orientations.append(next(
            (ids[index], ids[(index + 1) % 3])
            for index in range(3)
            if {ids[index], ids[(index + 1) % 3]} == common))
    if shared_orientations[0] != tuple(reversed(shared_orientations[1])):
        raise ValueError("source patch faces do not have reciprocal orientation")

    sides = (value.get("first"), value.get("second"))
    surfaces = (first, second)
    support_maps, vertex_maps, coordinate_maps = [], [], []
    side_fields = {"identity", "source_row_payload_sha256", "surface_hashes",
                   "vertex_bindings", "patch_faces", "support_faces"}
    surface_hash_fields = {"positions_sha256", "faces_sha256", "gids_sha256",
                           "source_face_lineage_sha256", "source_vertex_lineage_sha256"}
    for side_index, (side, surface) in enumerate(zip(sides, surfaces)):
        if not isinstance(side, dict) or set(side) != side_fields:
            raise ValueError("source patch side record is malformed")
        if side.get("identity") != list(surface.key):
            raise ValueError("source patch side identity differs from prepared surface")
        row_sha = side.get("source_row_payload_sha256")
        if (not isinstance(row_sha, str) or len(row_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in row_sha)):
            raise ValueError("source patch row payload SHA-256 is invalid")
        hashes = side.get("surface_hashes")
        if not isinstance(hashes, dict) or set(hashes) != surface_hash_fields:
            raise ValueError("source patch surface hash record is malformed")
        if hashes != _source_patch_surface_hashes(surface):
            raise ValueError("source patch surface or ancestry hashes are stale")
        bindings = side.get("vertex_bindings")
        if not isinstance(bindings, list) or len(bindings) != len(patch_vertices):
            raise ValueError("source patch vertex binding coverage is incomplete")
        vertex_map = {}
        for binding in bindings:
            if (not isinstance(binding, dict) or set(binding) != {
                    "source_vertex_id", "capture_gid"}):
                raise ValueError("source patch vertex binding is malformed")
            source_id, gid = binding.get("source_vertex_id"), binding.get("capture_gid")
            if (type(source_id) is not int or type(gid) is not int or source_id < 0 or gid < 0
                    or source_id in vertex_map or surface.vertex_index(gid) is None):
                raise ValueError("source patch vertex binding is invalid or duplicated")
            lineage = surface.source_vertex_lineage.get(gid)
            if (not _has_source_vertex_lineage(lineage)
                    or lineage.get("source_vertex_id") != source_id
                    or lineage.get("source_member") != member
                    or lineage.get("source_member_sha256") != member_sha):
                raise ValueError("source patch corner vertex ancestry does not match its authenticated binding")
            vertex_map[source_id] = gid
        if set(vertex_map) != set(patch_vertices):
            raise ValueError("source patch vertex bindings do not cover its exact corners")

        patch_rows = side.get("patch_faces")
        support_rows = side.get("support_faces")
        if not isinstance(patch_rows, list) or len(patch_rows) != 2:
            raise ValueError("each source patch side must bind both reciprocal faces")
        if not isinstance(support_rows, list) or len(support_rows) < 2:
            raise ValueError("source patch incident face support is missing")
        support_map = {}
        for support in support_rows:
            required = {"face_index", "source_member", "source_member_sha256",
                        "source_face", "source_row_face", "source_row_payload_sha256",
                        "ordered_capture_gids"}
            if not isinstance(support, dict) or set(support) != required:
                raise ValueError("source patch incident face support record is malformed")
            face_index = support.get("face_index")
            gids = support.get("ordered_capture_gids")
            if (type(face_index) is not int or not 0 <= face_index < len(surface.faces)
                    or face_index in support_map
                    or type(support.get("source_face")) is not int or support["source_face"] < 0
                    or type(support.get("source_row_face")) is not int or support["source_row_face"] < 0
                    or not isinstance(support.get("source_member"), str)
                    or not support["source_member"]
                    or not isinstance(support.get("source_member_sha256"), str)
                    or len(support["source_member_sha256"]) != 64
                    or any(ch not in "0123456789abcdef"
                           for ch in support["source_member_sha256"])
                    or support.get("source_row_payload_sha256") != row_sha
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(gid) is not int or gid < 0 for gid in gids)
                    or len(set(gids)) != 3):
                raise ValueError("source patch incident face support values are invalid")
            actual_gids = [int(surface.source_vertex_ids[int(local)])
                           for local in surface.faces[face_index]]
            if actual_gids != gids:
                raise ValueError("source patch incident face GIDs differ from prepared face")
            lineage = surface.source_face_lineage[face_index]
            expected = {
                "source_member": support["source_member"],
                "source_member_sha256": support["source_member_sha256"],
                "source_face": support["source_face"],
                "source_row_face": support["source_row_face"],
                "source_row_payload_sha256": row_sha,
                "owner_stable_id": surface.key[1],
            }
            if (not _has_source_face_lineage(lineage)
                    or not _source_patch_ancestry_matches(lineage, expected)):
                raise ValueError("source patch incident face ancestry does not match its binding")
            bound_patch_gids = set(vertex_map.values())
            if not (set(gids) & bound_patch_gids):
                raise ValueError("source patch support face lacks an indexed authenticated patch corner")
            support_patch_ids = {source_id for source_id, gid in vertex_map.items()
                                 if gid in gids}
            for source_id in support_patch_ids:
                vertex_record = surface.source_vertex_lineage.get(vertex_map[source_id])
                if (not _has_source_vertex_lineage(vertex_record)
                        or vertex_record.get("source_vertex_id") != source_id
                        or vertex_record.get("source_member") != member
                        or vertex_record.get("source_member_sha256") != member_sha):
                    raise ValueError("source patch support corner lineage differs from its exact binding")
            support_map[face_index] = support
        patch_points = {
            ci.float32_point_lattice_key(
                surface.positions[surface.vertex_index(gid)].tolist())
            for gid in vertex_map.values()
        }
        actual_support = {
            face_index for face_index, face in enumerate(surface.faces)
            if any(ci.float32_point_lattice_key(surface.positions[int(local)].tolist())
                   in patch_points for local in face)
        }
        if set(support_map) != actual_support:
            raise ValueError("source patch incident face support does not cover the exact corner and alias stars")

        patch_indices = set()
        for source_face, patch_row in zip(source_faces, patch_rows):
            required = {"face_index", "source_face", "source_row_face",
                        "source_row_payload_sha256", "ordered_source_vertex_ids",
                        "ordered_capture_gids"}
            if not isinstance(patch_row, dict) or set(patch_row) != required:
                raise ValueError("source patch face binding is malformed")
            face_index = patch_row.get("face_index")
            source_ids = patch_row.get("ordered_source_vertex_ids")
            gids = patch_row.get("ordered_capture_gids")
            if (type(face_index) is not int or face_index not in support_map
                    or face_index in patch_indices
                    or patch_row.get("source_face") != source_face["source_face"]
                    or type(patch_row.get("source_row_face")) is not int
                    or patch_row["source_row_face"] < 0
                    or patch_row.get("source_row_payload_sha256") != row_sha
                    or not isinstance(source_ids, list) or len(source_ids) != 3
                    or any(type(item) is not int for item in source_ids)
                    or set(source_ids) != set(source_face["source_vertex_ids"])
                    or not _cyclic_source_face_orientation(
                        source_ids, source_face["source_vertex_ids"],
                        reversed_order=(side_index == 1))
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(gid) is not int for gid in gids)
                    or gids != [vertex_map[item] for item in source_ids]):
                raise ValueError("source patch face mapping or orientation is invalid")
            support = support_map[face_index]
            if (support["source_face"] != patch_row["source_face"]
                    or support["source_row_face"] != patch_row["source_row_face"]
                    or support["source_member"] != member
                    or support["source_member_sha256"] != member_sha):
                raise ValueError("source patch face ancestry differs from authenticated member")
            lineage = surface.source_face_lineage[face_index]
            expected = {
                "source_member": member,
                "source_member_sha256": member_sha,
                "source_face": source_face["source_face"],
                "source_row_face": patch_row["source_row_face"],
                "source_row_payload_sha256": row_sha,
                "owner_stable_id": surface.key[1],
            }
            if not _source_patch_ancestry_matches(lineage, expected):
                raise ValueError("source patch face ancestry does not match its authenticated source frame")
            recovered = _exact_source_corner_ids_from_lineage(
                lineage, source_face["source_vertex_ids"],
                expected_source_face=source_face["source_face"],
                expected_source_member=member,
                expected_source_member_sha256=member_sha,
                expected_source_row_face=patch_row["source_row_face"],
                expected_source_row_payload_sha256=row_sha)
            if recovered is not None and recovered != source_ids:
                raise ValueError("source patch face corner ancestry is not exact")
            if (recovered is None and any(
                    isinstance(row.get("corner_barycentric_exact"), list)
                    for row in _source_face_lineage_chain(lineage))):
                raise ValueError("source patch face corner ancestry is not linked to its authenticated frame")
            patch_indices.add(face_index)
        coordinate_map = {}
        for source_id, gid in vertex_map.items():
            position = surface.positions[surface.vertex_index(gid)]
            coordinate_map[source_id] = tuple(
                Fraction(value, 1 << 149)
                for value in ci.float32_point_lattice_key(position.tolist()))
        support_maps.append(support_map)
        vertex_maps.append(vertex_map)
        coordinate_maps.append(coordinate_map)

    if (first.rigid_owner_proof is None or second.rigid_owner_proof is None
            or first.rigid_owner_proof != second.rigid_owner_proof):
        raise ValueError("source patch surfaces do not share the same accepted rigid-owner proof")
    for source_id in patch_vertices:
        if coordinate_maps[0][source_id] != coordinate_maps[1][source_id]:
            raise ValueError("reciprocal source patch corner coordinates differ")
    patch_face_indices = []
    for source_face in source_faces:
        face_indices = []
        for side in sides:
            row = next((item for item in side["patch_faces"]
                        if item["source_face"] == source_face["source_face"]), None)
            if row is None:
                raise ValueError("source patch face is missing from one side")
            face_indices.append(row["face_index"])
        patch_face_indices.append((face_indices[0], face_indices[1],
                                   source_face["source_face"]))
    return {
        "source_patch_complex_sha256": proof_sha,
        "source_member": member,
        "source_member_sha256": member_sha,
        "source_faces": source_faces,
        "patch_vertices": tuple(patch_vertices),
        "patch_edges": _source_patch_edges(source_faces),
        "support_faces": (set(support_maps[0]), set(support_maps[1])),
        "vertex_maps": vertex_maps,
        "coordinate_maps": coordinate_maps,
        "patch_face_pairs": tuple(patch_face_indices),
    }


def _exact_point_on_source_patch_edge(point, start, end):
    direction = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    cross = (direction[1] * offset[2] - direction[2] * offset[1],
             direction[2] * offset[0] - direction[0] * offset[2],
             direction[0] * offset[1] - direction[1] * offset[0])
    return (cross == (0, 0, 0)
            and all(min(start[index], end[index]) <= point[index] <=
                    max(start[index], end[index]) for index in range(3)))


def _source_patch_complex_contact(first, second, first_face, second_face, patch):
    if (first_face not in patch["support_faces"][0]
            or second_face not in patch["support_faces"][1]):
        return False, "contact_faces_are_outside_authenticated_incident_support"
    points = first.intersection_points_meters(second, first_face, second_face)
    if patch.get("carrier_complex"):
        direct = patch["carrier_face_pair_lookup"].get((first_face, second_face))
        if direct is not None:
            if len(points) == 3 and set(points) == set(direct[1]):
                return True, "exact_reciprocal_source_carrier_face"
            return False, "reciprocal_source_carrier_face_intersection_is_not_exact"
        if len(points) == 1 and _cached_exact_point_on_carrier(points[0], patch):
            return True, "exact_point_contact_on_authenticated_carrier"
        if len(points) == 2 and _cached_exact_segment_on_carrier(
                points[0], points[1], patch):
            return True, "exact_segment_contact_fully_covered_by_authenticated_carrier"
        return False, "contact_is_not_fully_covered_by_authenticated_carrier"
    direct_source_face = next((source_face for first_index, second_index, source_face
                               in patch["patch_face_pairs"]
                               if first_index == first_face and second_index == second_face), None)
    if direct_source_face is not None:
        source_face = next(row for row in patch["source_faces"]
                           if row["source_face"] == direct_source_face)
        triangle_points = {patch["coordinate_maps"][0][source_id]
                           for source_id in source_face["source_vertex_ids"]}
        if set(points) == triangle_points and len(points) == 3:
            return True, "exact_reciprocal_source_patch_face"
        return False, "reciprocal_source_patch_face_intersection_is_not_the_exact_triangle"

    if len(points) not in (1, 2):
        return False, "contact_is_not_an_exact_patch_corner_or_edge"
    gids_a = {int(first.source_vertex_ids[int(local)])
              for local in first.faces[first_face]}
    gids_b = {int(second.source_vertex_ids[int(local)])
              for local in second.faces[second_face]}
    source_to_gid_a, source_to_gid_b = patch["vertex_maps"]
    for point in points:
        matching_vertices = [source_id for source_id, xyz in patch["coordinate_maps"][0].items()
                             if point == xyz]
        if len(matching_vertices) == 1:
            source_id = matching_vertices[0]
            mapped_a, mapped_b = source_to_gid_a[source_id], source_to_gid_b[source_id]
            incident_a = bool(mapped_a & gids_a) if isinstance(mapped_a, set) else mapped_a in gids_a
            incident_b = bool(mapped_b & gids_b) if isinstance(mapped_b, set) else mapped_b in gids_b
            if incident_a and incident_b:
                continue
            return False, "patch_corner_is_not_incident_to_both_contact_faces"
        matching_edges = [edge for edge in patch["patch_edges"]
                          if _exact_point_on_source_patch_edge(
                              point, patch["coordinate_maps"][0][edge[0]],
                              patch["coordinate_maps"][0][edge[1]])]
        if len(matching_edges) != 1:
            return False, "contact_point_is_not_on_one_authenticated_patch_edge"
        edge = matching_edges[0]
        def incident(source_map, source_id, gids):
            mapped = source_map[source_id]
            return bool(mapped & gids) if isinstance(mapped, set) else mapped in gids
        if not all((incident(source_to_gid_a, source_id, gids_a)
                    and incident(source_to_gid_b, source_id, gids_b)) for source_id in edge):
            return False, "patch_edge_is_not_incident_to_both_contact_faces"
    if len(points) == 2:
        common_edges = [edge for edge in patch["patch_edges"]
                        if all(_exact_point_on_source_patch_edge(
                            point, patch["coordinate_maps"][0][edge[0]],
                            patch["coordinate_maps"][0][edge[1]]) for point in points)]
        if len(common_edges) != 1:
            return False, "segment_is_not_contained_in_one_authenticated_patch_edge"
    return True, "exact_contact_contained_in_supported_source_patch_corner_or_edge"


def _exact_triangle_normal(triangle):
    first = tuple(triangle[1][i] - triangle[0][i] for i in range(3))
    second = tuple(triangle[2][i] - triangle[0][i] for i in range(3))
    return (first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0])


def _exact_point_on_carrier(point, triangles):
    for triangle in triangles:
        normal = _exact_triangle_normal(triangle)
        if normal == (0, 0, 0):
            continue
        delta = tuple(point[i] - triangle[0][i] for i in range(3))
        if sum((normal[i] * delta[i] for i in range(3)), Fraction(0)) != 0:
            continue
        try:
            if all(weight >= 0 for weight in _patch_barycentric(triangle, point)):
                return True
        except ValueError:
            continue
    return False


def _exact_segment_triangle_interval(start, end, triangle):
    """Return the exact closed parameter interval where a segment meets a triangle."""
    normal = _exact_triangle_normal(triangle)
    if normal == (0, 0, 0):
        return None
    d0 = sum((normal[i] * (start[i] - triangle[0][i]) for i in range(3)), Fraction(0))
    d1 = sum((normal[i] * (end[i] - triangle[0][i]) for i in range(3)), Fraction(0))
    if d0 == 0 and d1 == 0:
        try:
            first = _patch_barycentric(triangle, start)
            last = _patch_barycentric(triangle, end)
        except ValueError:
            return None
        lower, upper = Fraction(0), Fraction(1)
        for a, b in zip(first, last):
            slope = b - a
            if slope == 0:
                if a < 0:
                    return None
                continue
            crossing = -a / slope
            if slope > 0:
                lower = max(lower, crossing)
            else:
                upper = min(upper, crossing)
        if lower > upper or upper < 0 or lower > 1:
            return None
        return max(Fraction(0), lower), min(Fraction(1), upper)
    denominator = d0 - d1
    if denominator == 0:
        return None
    parameter = d0 / denominator
    if not 0 <= parameter <= 1:
        return None
    point = tuple(start[i] + parameter * (end[i] - start[i]) for i in range(3))
    return (parameter, parameter) if _exact_point_on_carrier(point, (triangle,)) else None


def _exact_segment_on_carrier(start, end, triangles):
    """Require the full segment, not samples, to lie in the carrier triangle union."""
    intervals = [interval for triangle in triangles
                 if (interval := _exact_segment_triangle_interval(start, end, triangle)) is not None]
    if not intervals:
        return False
    intervals.sort()
    if intervals[0][0] > 0:
        return False
    covered = intervals[0][1]
    for lower, upper in intervals[1:]:
        if lower > covered:
            return False
        covered = max(covered, upper)
        if covered >= 1:
            return True
    return covered >= 1


_CARRIER_QUERY_CACHE_LIMIT = 65536


def _exact_triangle_aabb(triangle):
    return (
        tuple(min(point[axis] for point in triangle) for axis in range(3)),
        tuple(max(point[axis] for point in triangle) for axis in range(3)),
    )


def _aabb_overlaps(first, second):
    return all(first[0][axis] <= second[1][axis]
               and second[0][axis] <= first[1][axis] for axis in range(3))


def _cached_exact_point_on_carrier(point, patch):
    """Exact carrier membership with conservative exact-AABB pruning and bounded memoization."""
    point = tuple(point)
    cache = patch["_carrier_query_cache"]["point"]
    if point in cache:
        return cache[point]
    candidates = [
        triangle for triangle, (lower, upper) in zip(
            patch["carrier_triangles"], patch["carrier_triangle_aabbs"])
        if all(lower[axis] <= point[axis] <= upper[axis] for axis in range(3))
    ]
    result = _exact_point_on_carrier(point, candidates)
    if len(cache) < _CARRIER_QUERY_CACHE_LIMIT:
        cache[point] = result
    return result


def _cached_exact_segment_on_carrier(start, end, patch):
    """Exact full-segment coverage with conservative AABB pruning and canonical cache keys."""
    endpoints = tuple(sorted((tuple(start), tuple(end))))
    if len(endpoints) != 2:
        raise ValueError("carrier segment must have exactly two endpoints")
    cache = patch["_carrier_query_cache"]["segment"]
    if endpoints in cache:
        return cache[endpoints]
    first, last = endpoints
    segment_bounds = (
        tuple(min(first[axis], last[axis]) for axis in range(3)),
        tuple(max(first[axis], last[axis]) for axis in range(3)),
    )
    candidates = [
        triangle for triangle, triangle_bounds in zip(
            patch["carrier_triangles"], patch["carrier_triangle_aabbs"])
        if _aabb_overlaps(segment_bounds, triangle_bounds)
    ]
    result = _exact_segment_on_carrier(first, last, candidates)
    if len(cache) < _CARRIER_QUERY_CACHE_LIMIT:
        cache[endpoints] = result
    return result


def _carrier_topology(point_ids, carrier_faces, expected_components):
    """Validate the carrier as an oriented, manifold triangular 2-complex."""
    edge_rows, adjacency, incident = {}, [set() for _ in carrier_faces], {}
    seen_faces = set()
    for face_index, face in enumerate(carrier_faces):
        ids = tuple(face)
        if len(set(ids)) != 3 or any(point_id not in point_ids for point_id in ids):
            raise ValueError("source patch carrier face has invalid point references")
        face_key = tuple(sorted(ids))
        if face_key in seen_faces:
            raise ValueError("source patch carrier repeats a triangle")
        seen_faces.add(face_key)
        if _exact_triangle_normal([point_ids[item] for item in ids]) == (0, 0, 0):
            raise ValueError("source patch carrier contains a degenerate triangle")
        for point_id in ids:
            incident.setdefault(point_id, []).append(face_index)
        for index in range(3):
            start, end = ids[index], ids[(index + 1) % 3]
            edge_rows.setdefault(tuple(sorted((start, end))), []).append((face_index, start, end))
    for uses in edge_rows.values():
        if len(uses) > 2:
            raise ValueError("source patch carrier has a nonmanifold edge")
        if len(uses) == 2:
            (a, a0, a1), (b, b0, b1) = uses
            if (a0, a1) != (b1, b0):
                raise ValueError("source patch carrier internal edge winding is inconsistent")
            adjacency[a].add(b)
            adjacency[b].add(a)
    components, unseen = [], set(range(len(carrier_faces)))
    while unseen:
        seed = min(unseen)
        stack, component = [seed], set()
        while stack:
            face_index = stack.pop()
            if face_index in component:
                continue
            component.add(face_index)
            unseen.discard(face_index)
            stack.extend(adjacency[face_index] - component)
        components.append(component)
    if len(components) != expected_components:
        raise ValueError("source patch carrier component count differs from declared source topology")
    for point_id, faces in incident.items():
        link = {}
        for face_index in faces:
            opposite = [item for item in carrier_faces[face_index] if item != point_id]
            if len(opposite) != 2:
                raise ValueError("source patch carrier vertex link is malformed")
            a, b = opposite
            link.setdefault(a, set()).add(b)
            link.setdefault(b, set()).add(a)
        if not link or any(len(neighbors) > 2 for neighbors in link.values()):
            raise ValueError("source patch carrier has a nonmanifold vertex link")
        reached, stack = set(), [next(iter(link))]
        while stack:
            item = stack.pop()
            if item in reached:
                continue
            reached.add(item)
            stack.extend(link[item] - reached)
        degrees = [len(neighbors) for neighbors in link.values()]
        if (reached != set(link) or degrees.count(1) not in (0, 2)
                or any(degree not in (1, 2) for degree in degrees)):
            raise ValueError("source patch carrier has an invalid boundary vertex link")
    return {"component_count": len(components),
            "component_face_counts": sorted(len(component) for component in components),
            "edge_count": len(edge_rows),
            "boundary_edge_count": sum(len(uses) == 1 for uses in edge_rows.values())}


def _prepare_source_patch_carrier_complex(first, second, value):
    """Validate a sealed multi-face carrier using the existing source-patch owner."""
    fields = {"schema", "source_patch_complex_sha256", "identities", "source_member",
              "source_member_sha256", "input_pins", "expected_component_count", "points",
              "source_faces", "carrier_faces", "first", "second"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("source patch carrier fields are malformed")
    payload = {key: item for key, item in value.items() if key != "source_patch_complex_sha256"}
    digest = value.get("source_patch_complex_sha256")
    if (value.get("schema") != _SOURCE_PATCH_COMPLEX_SCHEMA
            or not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or digest != _canonical_json_sha256(payload)):
        raise ValueError("source patch carrier seal is invalid")
    if value.get("identities") != [list(first.key), list(second.key)]:
        raise ValueError("source patch carrier ordered identities differ")
    member, member_sha = value.get("source_member"), value.get("source_member_sha256")
    if (not isinstance(member, str) or not member
            or not isinstance(member_sha, str) or len(member_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in member_sha)):
        raise ValueError("source patch carrier source identity is invalid")
    pins = value.get("input_pins")
    pin_fields = {"source_obj_sha256", "arrangement_sha256", "arrangement_certificate_sha256",
                  "first_payload_sha256", "second_payload_sha256", "capture_pack_sha256",
                  "capture_receipt_sha256"}
    if (not isinstance(pins, dict) or set(pins) != pin_fields
            or any(not isinstance(item, str) or len(item) != 64
                   or any(ch not in "0123456789abcdef" for ch in item)
                   for item in pins.values())):
        raise ValueError("source patch carrier input pins are malformed")
    expected_components = value.get("expected_component_count")
    if type(expected_components) is not int or expected_components < 1:
        raise ValueError("source patch carrier expected component count is invalid")
    if (first.rigid_owner_proof is None or second.rigid_owner_proof is None
            or first.rigid_owner_proof != second.rigid_owner_proof):
        raise ValueError("source patch carrier surfaces do not share accepted owner/rest-frame proof")

    point_rows = value.get("points")
    if not isinstance(point_rows, list) or len(point_rows) < 3:
        raise ValueError("source patch carrier point inventory is incomplete")
    local_points, capture_points, local_bits, capture_bits = {}, {}, {}, {}
    for row in point_rows:
        required = {"point_id", "local_xyz_exact", "capture_xyz_exact",
                    "local_float32_bits", "capture_float32_bits"}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("source patch carrier point record is malformed")
        point_id = row["point_id"]
        if not isinstance(point_id, str) or not point_id or point_id in local_points:
            raise ValueError("source patch carrier point identity is invalid or duplicated")
        local = tuple(_patch_fraction(pair) for pair in row["local_xyz_exact"])
        captured = tuple(_patch_fraction(pair) for pair in row["capture_xyz_exact"])
        if len(local) != 3 or len(captured) != 3:
            raise ValueError("source patch carrier point coordinates are malformed")
        for name, xyz, bits_map in (("local", local, local_bits), ("capture", captured, capture_bits)):
            raw = row[name + "_float32_bits"]
            if (not isinstance(raw, list) or len(raw) != 3
                    or any(type(bit) is not int or not 0 <= bit <= 0xffffffff for bit in raw)
                    or _patch_rational_float32_bits(xyz) != raw):
                raise ValueError("source patch carrier point bits differ from exact rational Float32 rounding")
            if name == "capture" and tuple(
                    _patch_float32_exact_fraction(bit) for bit in raw) != xyz:
                raise ValueError("source patch carrier capture point is not its exact Float32 lattice value")
            bits_map[point_id] = tuple(raw)
        local_points[point_id], capture_points[point_id] = local, captured
    if (len(set(local_bits.values())) != len(local_points)
            or len(set(capture_bits.values())) != len(local_points)):
        raise ValueError("source patch carrier points collapse after Float32 quantization")

    parent_rows = value.get("source_faces")
    if not isinstance(parent_rows, list) or not parent_rows:
        raise ValueError("source patch carrier has no authenticated source faces")
    parents = {}
    for row in parent_rows:
        if not isinstance(row, dict) or set(row) != {
                "source_face", "source_vertex_ids", "source_vertices_exact"}:
            raise ValueError("source patch carrier parent face is malformed")
        face_id, vertex_ids, xyz_rows = row["source_face"], row["source_vertex_ids"], row["source_vertices_exact"]
        if (type(face_id) is not int or face_id < 0 or face_id in parents
                or not isinstance(vertex_ids, list) or len(vertex_ids) != 3
                or any(type(item) is not int or item < 0 for item in vertex_ids)
                or len(set(vertex_ids)) != 3 or not isinstance(xyz_rows, list) or len(xyz_rows) != 3):
            raise ValueError("source patch carrier parent face IDs are invalid")
        triangle = tuple(tuple(_patch_fraction(pair) for pair in point) for point in xyz_rows)
        if any(len(point) != 3 for point in triangle) or _exact_triangle_normal(triangle) == (0, 0, 0):
            raise ValueError("source patch carrier parent triangle is malformed or degenerate")
        parents[face_id] = (tuple(vertex_ids), triangle)

    carrier_rows = value.get("carrier_faces")
    if not isinstance(carrier_rows, list) or not carrier_rows:
        raise ValueError("source patch carrier has no faces")
    carrier_ids, carrier_faces, carrier_capture_triangles = [], [], []
    for row in carrier_rows:
        required = {"carrier_face_id", "source_face", "arrangement_record_index",
                    "arrangement_record_sha256", "arrangement_record", "point_ids",
                    "barycentric_exact"}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("source patch carrier face record is malformed")
        carrier_id, source_face = row["carrier_face_id"], row["source_face"]
        point_ids, bary_rows = row["point_ids"], row["barycentric_exact"]
        record, record_sha = row["arrangement_record"], row["arrangement_record_sha256"]
        record_index = row["arrangement_record_index"]
        if (not isinstance(carrier_id, str) or not carrier_id or carrier_id in carrier_ids
                or type(source_face) is not int or source_face not in parents
                or type(record_index) is not int or record_index < 0
                or not isinstance(record, dict) or not isinstance(record_sha, str)
                or len(record_sha) != 64 or record_sha != _canonical_json_sha256(record)
                or not isinstance(point_ids, list) or len(point_ids) != 3
                or any(point_id not in local_points for point_id in point_ids)
                or len(set(point_ids)) != 3 or not isinstance(bary_rows, list) or len(bary_rows) != 3):
            raise ValueError("source patch carrier face values are invalid")
        if (record.get("carrier_face_id") != carrier_id or record.get("source_member") != member
                or record.get("source_member_sha256") != member_sha
                or record.get("source_face") != source_face or record.get("point_ids") != point_ids
                or record.get("barycentric_exact") != bary_rows):
            raise ValueError("source patch carrier arrangement record is stale or misidentified")
        parent_triangle = parents[source_face][1]
        for point_id, encoded_bary in zip(point_ids, bary_rows):
            if not isinstance(encoded_bary, list) or len(encoded_bary) != 3:
                raise ValueError("source patch carrier barycentric map is malformed")
            bary = tuple(_patch_fraction(pair) for pair in encoded_bary)
            if sum(bary, Fraction(0)) != 1 or any(weight < 0 for weight in bary):
                raise ValueError("source patch carrier barycentric weights are invalid")
            rebuilt = tuple(sum((bary[index] * parent_triangle[index][axis]
                                 for index in range(3)), Fraction(0)) for axis in range(3))
            if rebuilt != local_points[point_id]:
                raise ValueError("source patch carrier ancestry does not reconstruct its point")
        expected_xyz = [[[str(value.numerator), str(value.denominator)] for value in local_points[pid]]
                        for pid in point_ids]
        if record.get("local_xyz_exact") != expected_xyz:
            raise ValueError("source patch carrier record coordinates differ from barycentric points")
        carrier_ids.append(carrier_id)
        carrier_faces.append(tuple(point_ids))
        carrier_capture_triangles.append(tuple(capture_points[pid] for pid in point_ids))
    if set(pid for face in carrier_faces for pid in face) != set(local_points):
        raise ValueError("source patch carrier point inventory has unused or missing points")
    topology = _carrier_topology(local_points, carrier_faces, expected_components)

    sides = (value.get("first"), value.get("second"))
    surfaces = (first, second)
    side_fields = {"identity", "source_row_payload_sha256", "surface_hashes",
                   "point_bindings", "patch_faces", "support_faces"}
    support_maps, point_maps, coordinate_maps, patch_maps = [], [], [], []
    for side_index, (side, surface, pin_name) in enumerate(zip(
            sides, surfaces, ("first_payload_sha256", "second_payload_sha256"))):
        if not isinstance(side, dict) or set(side) != side_fields or side.get("identity") != list(surface.key):
            raise ValueError("source patch carrier side identity or fields differ")
        row_sha = side.get("source_row_payload_sha256")
        if row_sha != pins[pin_name]:
            raise ValueError("source patch carrier side payload differs from pinned row")
        if surface.source_face_lineage is None or surface.source_vertex_lineage is None:
            raise ValueError("source patch carrier requires complete face and vertex lineage")
        if side.get("surface_hashes") != _source_patch_surface_hashes(surface):
            raise ValueError("source patch carrier surface or ancestry hash is stale")
        bindings = side.get("point_bindings")
        if not isinstance(bindings, list) or len(bindings) != len(local_points):
            raise ValueError("source patch carrier point binding coverage is incomplete")
        point_map, coordinate_map = {}, {}
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {
                    "point_id", "capture_gids", "world_float32_bits"}:
                raise ValueError("source patch carrier point binding is malformed")
            point_id, gids, bits = binding["point_id"], binding["capture_gids"], binding["world_float32_bits"]
            if (point_id not in local_points or point_id in point_map or not isinstance(gids, list) or not gids
                    or any(type(gid) is not int or gid < 0 for gid in gids)
                    or len(set(gids)) != len(gids) or bits != list(capture_bits[point_id])):
                raise ValueError("source patch carrier point binding values are invalid")
            coords = set()
            for gid in gids:
                vertex = surface.vertex_index(gid)
                if vertex is None or _patch_float32_bits(surface.positions[vertex]) != bits:
                    raise ValueError("source patch carrier point differs from captured GID")
                coords.add(tuple(Fraction(item, 1 << 149) for item in
                                 ci.float32_point_lattice_key(surface.positions[vertex].tolist())))
            if len(coords) != 1 or next(iter(coords)) != capture_points[point_id]:
                raise ValueError("source patch carrier coordinate aliases disagree with exact point")
            vertex = surface.vertex_index(gids[0])
            point_key = tuple(ci.float32_point_lattice_key(surface.positions[vertex].tolist()))
            expected_aliases = {
                int(surface.source_vertex_ids[index]) for index, position in enumerate(surface.positions)
                if tuple(ci.float32_point_lattice_key(position.tolist())) == point_key}
            if set(gids) != expected_aliases:
                raise ValueError("source patch carrier omits or invents exact-coordinate alias GIDs")
            point_map[point_id], coordinate_map[point_id] = set(gids), next(iter(coords))
        if set(point_map) != set(local_points) or len(set(coordinate_map.values())) != len(local_points):
            raise ValueError("source patch carrier point map is incomplete or nonunique")

        support_rows = side.get("support_faces")
        if not isinstance(support_rows, list) or not support_rows:
            raise ValueError("source patch carrier incident face support is missing")
        support_map = {}
        for support in support_rows:
            required = {"face_index", "source_member", "source_member_sha256", "source_face",
                        "source_row_face", "source_row_payload_sha256", "ordered_capture_gids"}
            if not isinstance(support, dict) or set(support) != required:
                raise ValueError("source patch carrier support-face record is malformed")
            fi, gids = support["face_index"], support["ordered_capture_gids"]
            if (type(fi) is not int or not 0 <= fi < len(surface.faces) or fi in support_map
                    or type(support["source_face"]) is not int or support["source_face"] < 0
                    or type(support["source_row_face"]) is not int or support["source_row_face"] < 0
                    or support["source_row_payload_sha256"] != row_sha
                    or not isinstance(support["source_member"], str) or not support["source_member"]
                    or not isinstance(support["source_member_sha256"], str)
                    or len(support["source_member_sha256"]) != 64
                    or any(ch not in "0123456789abcdef" for ch in support["source_member_sha256"])
                    or not isinstance(gids, list) or len(gids) != 3
                    or any(type(gid) is not int for gid in gids) or len(set(gids)) != 3):
                raise ValueError("source patch carrier support-face values are invalid")
            actual_gids = [int(surface.source_vertex_ids[int(local)]) for local in surface.faces[fi]]
            if actual_gids != gids:
                raise ValueError("source patch carrier support GIDs differ from actual face")
            expected = {key: support[key] for key in (
                "source_member", "source_member_sha256", "source_face", "source_row_face",
                "source_row_payload_sha256")}
            expected["owner_stable_id"] = surface.key[1]
            if not _source_patch_ancestry_matches(surface.source_face_lineage[fi], expected):
                raise ValueError("source patch carrier support ancestry differs from bound face")
            support_map[fi] = support
        point_lattice = {tuple(ci.float32_point_lattice_key(
            surface.positions[surface.vertex_index(gid)].tolist()))
                         for gids in point_map.values() for gid in gids}
        actual_support = {fi for fi, face in enumerate(surface.faces)
                          if any(tuple(ci.float32_point_lattice_key(surface.positions[int(local)].tolist()))
                                 in point_lattice for local in face)}
        if set(support_map) != actual_support:
            raise ValueError("source patch carrier support omits or invents exact incident stars")

        patch_rows = side.get("patch_faces")
        if not isinstance(patch_rows, list) or len(patch_rows) != len(carrier_faces):
            raise ValueError("source patch carrier face mapping is incomplete")
        patch_map = {}
        used_face_indices = set()
        for row in patch_rows:
            required = {"carrier_face_id", "face_index", "source_member", "source_member_sha256",
                        "source_face", "source_row_face", "lineage_face_id", "ordered_point_ids",
                        "ordered_capture_gids"}
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("source patch carrier emitted-face record is malformed")
            cid, fi = row["carrier_face_id"], row["face_index"]
            point_ids, gids = row["ordered_point_ids"], row["ordered_capture_gids"]
            if (cid not in carrier_ids or cid in patch_map or type(fi) is not int
                    or fi in used_face_indices
                    or fi not in support_map or type(row["source_face"]) is not int
                    or type(row["source_row_face"]) is not int or type(row["lineage_face_id"]) is not int
                    or row["lineage_face_id"] < 0 or not isinstance(point_ids, list) or len(point_ids) != 3
                    or not isinstance(gids, list) or len(gids) != 3):
                raise ValueError("source patch carrier emitted-face values are invalid")
            carrier_index = carrier_ids.index(cid)
            carrier_row = carrier_rows[carrier_index]
            if (set(point_ids) != set(carrier_row["point_ids"])
                    or not _cyclic_source_face_orientation(
                        point_ids, carrier_row["point_ids"], reversed_order=(side_index == 1))):
                raise ValueError("source patch carrier face does not preserve reciprocal winding")
            actual_gids = [int(surface.source_vertex_ids[int(local)]) for local in surface.faces[fi]]
            if actual_gids != gids or any(gid not in point_map[pid] for gid, pid in zip(gids, point_ids)):
                raise ValueError("source patch carrier face GIDs differ from point crosswalk")
            support = support_map[fi]
            if any(row[key] != support[key] for key in (
                    "source_member", "source_member_sha256", "source_face", "source_row_face")):
                raise ValueError("source patch carrier emitted-face ancestry differs from support")
            lineage = surface.source_face_lineage[fi]
            if not _source_patch_ancestry_matches(lineage, {
                    "source_member": row["source_member"],
                    "source_member_sha256": row["source_member_sha256"],
                    "source_face": row["source_face"], "source_row_face": row["source_row_face"],
                    "source_row_payload_sha256": row_sha, "owner_stable_id": surface.key[1]}):
                raise ValueError("source patch carrier direct source ancestry is invalid")
            if not _source_patch_ancestry_matches(lineage, {
                    "source_member": member, "source_member_sha256": member_sha,
                    "source_face": carrier_row["source_face"], "owner_stable_id": surface.key[1]}):
                raise ValueError("source patch carrier common-parent ancestry is invalid")
            patch_map[cid] = (fi, tuple(point_ids), tuple(gids))
            used_face_indices.add(fi)
        if set(patch_map) != set(carrier_ids):
            raise ValueError("source patch carrier emitted-face IDs do not exactly cover carrier")
        support_maps.append(support_map)
        point_maps.append(point_map)
        coordinate_maps.append(coordinate_map)
        patch_maps.append(patch_map)

    patch_pairs = []
    patch_face_pair_lookup = {}
    for index, carrier_id in enumerate(carrier_ids):
        a, b = patch_maps[0][carrier_id], patch_maps[1][carrier_id]
        different_points = [(point_id, coordinate_maps[0].get(point_id),
                             coordinate_maps[1].get(point_id)) for point_id in a[1]
                            if coordinate_maps[0].get(point_id) != coordinate_maps[1].get(point_id)]
        if (set(a[1]) != set(b[1]) or not _cyclic_source_face_orientation(
                a[1], b[1], reversed_order=True) or different_points):
            raise ValueError("source patch carrier reciprocal faces differ in owner-frame coordinates")
        carrier_triangle = tuple(capture_points[point_id] for point_id in carrier_faces[index])
        patch_pairs.append((a[0], b[0], carrier_id, carrier_triangle))
        patch_face_pair_lookup.setdefault((a[0], b[0]), (carrier_id, carrier_triangle))
    carrier_triangles = tuple(carrier_capture_triangles)
    return {"source_patch_complex_sha256": digest, "source_member": member,
            "source_member_sha256": member_sha, "support_faces": tuple(map(set, support_maps)),
            "vertex_maps": point_maps, "coordinate_maps": coordinate_maps,
            "patch_face_pairs": tuple(patch_pairs),
            "carrier_face_pair_lookup": patch_face_pair_lookup, "carrier_complex": True,
            "carrier_triangles": carrier_triangles,
            "carrier_triangle_aabbs": tuple(_exact_triangle_aabb(triangle)
                                            for triangle in carrier_triangles),
            "_carrier_query_cache": {"point": {}, "segment": {}},
            "carrier_topology": topology}


def audit_surface_pair(first, second, *, intended_face_pairs=(), intended_enclosures=(),
                       intended_shared_feature_pairs=(), intended_source_patch_complexes=()):
    allowed = {tuple(pair) for pair in intended_face_pairs}
    enclosures = _intended_enclosures(intended_enclosures)
    if any(set(pair) != {first.key, second.key} for pair in enclosures):
        raise ValueError("intended enclosure must identify this surface pair")
    patch_contexts = [
        _prepare_source_patch_complex(first, second, value)
        for value in intended_source_patch_complexes
    ]
    if not _overlap(first, second):
        # Disjoint solids are safe ordinary neighbors, but cannot satisfy a
        # declared anatomical enclosure. Retain that distinction in the cheap
        # path instead of bypassing the outside-enclosure rejection below.
        requires_carrier_contact = any(context.get("carrier_complex") for context in patch_contexts)
        return {"identities": [list(first.key), list(second.key)],
                "passed": not enclosures and not requires_carrier_contact,
                "aabb_disjoint": True, "triangle_pair_count": 0,
                "declared_enclosure_outside_aabb": bool(enclosures),
                "declared_carrier_outside_aabb": requires_carrier_contact,
                "intended_enclosures": [[list(a), list(b)] for a, b in sorted(enclosures)]}
    result = ci._audit_pair_prepared_first(first.index, second.records, same_surface=False)
    pairs = result["triangle_pairs"]
    pair_set = {tuple(pair) for pair in pairs}
    accepted_features = {}
    rejected_features = []
    declared = set()
    for row in intended_shared_feature_pairs:
        if not isinstance(row, dict):
            raise TypeError("shared feature declarations must be mappings")
        try:
            declared_first = typed_key(row.get("first_identity"))
            declared_second = typed_key(row.get("second_identity"))
        except ValueError as exc:
            raise ValueError("shared feature declaration requires ordered typed surface identities") from exc
        if (declared_first, declared_second) != (first.key, second.key):
            raise ValueError("shared feature declaration identities do not match audit arguments")
        first_face = row.get("first_face_index")
        second_face = row.get("second_face_index")
        feature_gids = row.get("feature_gids")
        feature_gid_pairs = row.get("feature_gid_pairs")
        if type(first_face) is not int or type(second_face) is not int:
            raise ValueError("shared feature face indices must be integers")
        if not (0 <= first_face < len(first.faces) and 0 <= second_face < len(second.faces)):
            raise ValueError("shared feature face index out of range")
        pair = (first_face, second_face)
        if pair in declared:
            raise ValueError("duplicate shared feature face pair")
        declared.add(pair)
        if (feature_gids is None) == (feature_gid_pairs is None):
            raise ValueError("shared feature declaration requires exactly one GID or crosswalk form")
        if pair not in pair_set:
            rejected_features.append({"face_pair": list(pair),
                                      "feature_gids": feature_gids,
                                      "feature_gid_pairs": feature_gid_pairs,
                                      "reason": "declared_pair_has_no_exact_intersection"})
            continue
        if feature_gid_pairs is not None:
            source_point_crosswalk = row.get("source_point_crosswalk")
            source_seam_face_pairs = row.get("source_seam_face_pairs")
            endpoint_source_vertex_id = row.get("source_seam_endpoint_source_vertex_id")
            passed, reason = _shared_source_crosswalk_point_contact(
                first, second, first_face, second_face, feature_gid_pairs,
                source_point_crosswalk, source_seam_face_pairs, endpoint_source_vertex_id)
            if passed:
                accepted_features[pair] = {
                    "identities": [list(first.key), list(second.key)],
                    "face_pair": list(pair), "feature_gid_pairs": feature_gid_pairs,
                    "feature_kind": "source_crosswalk_point",
                    "source_point_crosswalk": source_point_crosswalk,
                    "source_seam_face_pairs": source_seam_face_pairs,
                    "source_seam_endpoint_source_vertex_id": endpoint_source_vertex_id,
                    "proof": reason}
            else:
                rejected_features.append({"face_pair": list(pair),
                                          "feature_gid_pairs": feature_gid_pairs,
                                          "source_point_crosswalk": source_point_crosswalk,
                                          "reason": reason})
            continue
        if not isinstance(feature_gids, (list, tuple)):
            raise TypeError("shared feature GIDs must be a list or tuple")
        passed, reason = _shared_feature_contact(first, second, first_face, second_face, feature_gids)
        if passed:
            accepted_features[pair] = {
                "identities": [list(first.key), list(second.key)],
                "face_pair": list(pair), "feature_gids": list(feature_gids),
                "feature_kind": "vertex" if len(feature_gids) == 1 else "edge",
                "proof": reason}
        else:
            rejected_features.append({"face_pair": list(pair), "feature_gids": list(feature_gids),
                                      "reason": reason})
    accepted_patches = {}
    rejected_patches = []
    expected_carrier_pairs = {
        (row[0], row[1]) for context in patch_contexts if context.get("carrier_complex")
        for row in context["patch_face_pairs"]
    }
    for raw_pair in pairs:
        pair = tuple(raw_pair)
        if pair in allowed or pair in accepted_features:
            continue
        matching_contexts = [
            context for context in patch_contexts
            if (pair[0] in context["support_faces"][0]
                and pair[1] in context["support_faces"][1])
        ]
        if len(matching_contexts) > 1:
            raise ValueError("one contact pair is covered by multiple source patch complexes")
        if not matching_contexts:
            continue
        passed, reason = _source_patch_complex_contact(
            first, second, pair[0], pair[1], matching_contexts[0])
        if passed:
            accepted_patches[pair] = {
                "identities": [list(first.key), list(second.key)],
                "face_pair": list(pair),
                "source_patch_complex_sha256":
                    matching_contexts[0]["source_patch_complex_sha256"],
                "proof": reason,
            }
        else:
            rejected_patches.append({"face_pair": list(pair), "reason": reason})
    unexpected = [pair for pair in pairs
                  if tuple(pair) not in allowed and tuple(pair) not in accepted_features
                  and tuple(pair) not in accepted_patches]
    observed_pairs = {tuple(pair) for pair in pairs}
    missing_carrier_pairs = sorted(expected_carrier_pairs - observed_pairs)
    for pair in missing_carrier_pairs:
        rejected_patches.append({"face_pair": list(pair),
                                 "reason": "declared_carrier_face_pair_not_observed"})
    # Shared boundary features can divide a connected surface component into
    # regions on opposite sides. Check every triangle interior centroid rather
    # than assuming one sample represents the whole connected component.
    containment = []
    if not unexpected:
        inspect_all_faces = bool(pairs)
        containment = [
            _contained_components(first, second, inspect_all_faces=inspect_all_faces),
            _contained_components(second, first, inspect_all_faces=inspect_all_faces),
        ]
    for row, (source, target) in zip(containment, ((first, second), (second, first))):
        row["source_identity"] = list(source.key)
        row["target_identity"] = list(target.key)
        row["declared_intended_enclosure"] = (source.key, target.key) in enclosures
    inside = any(row["inside_component_witnesses"] and
                 not row["declared_intended_enclosure"] for row in containment)
    outside_declared_enclosure = any(
        row["outside_component_witnesses"] and row["declared_intended_enclosure"]
        for row in containment)
    unresolved = any(row["status"] in ("not_checked_nonembedded_target",
                                     "unresolved_exact_point_location",
                                     "unresolved_all_component_samples_on_boundary") for row in containment)
    requires_carrier_solids = any(context.get("carrier_complex") for context in patch_contexts)
    carrier_solid_interior_checked = (
        not requires_carrier_solids or
        (first.closed and second.closed and len(containment) == 2
         and all(row["status"] == "checked_exact_rational_metres" for row in containment)))
    return {"identities": [list(first.key), list(second.key)],
            "passed": (not unexpected and not inside and not outside_declared_enclosure
                       and not unresolved and not missing_carrier_pairs
                       and carrier_solid_interior_checked),
            "triangle_pair_count": len(pairs), "unexpected_triangle_pairs": unexpected,
            "declared_interface_triangle_pairs": [p for p in pairs if tuple(p) in allowed],
            "declared_interface_feature_contacts": [accepted_features[p] for p in sorted(accepted_features)],
            "rejected_interface_feature_contacts": rejected_features,
            "declared_source_patch_complex_contacts": [accepted_patches[p]
                                                       for p in sorted(accepted_patches)],
            "rejected_source_patch_complex_contacts": rejected_patches,
            "missing_source_patch_carrier_face_pairs": [list(pair) for pair in missing_carrier_pairs],
            "source_patch_carrier_solid_interior_checked": carrier_solid_interior_checked,
            "containment": containment,
            "intended_enclosures": [[list(a), list(b)] for a, b in sorted(enclosures)],
            "boundary": "Declared enclosures permit interior placement only; every off-boundary face interior is checked and outside samples reject an intended enclosure. Shared feature contacts require exact GID or authenticated source-point crosswalk, owner, source-lineage, and intersection proof. Crosswalk contacts accept point contacts only. Declared face interfaces do not prove vessel lumen continuity."}


def validate_capture_receipt(receipt, pack_sha256):
    if receipt.get("schema") != "numi.human.accepted-render-geometry.v1":
        raise ValueError("unsupported accepted capture receipt")
    if receipt.get("physical_endpoint") != "accepted" or receipt.get("pack_file_sha256") != pack_sha256:
        raise ValueError("capture is not bound to this accepted pack")
    step, time = receipt.get("accepted_step"), receipt.get("accepted_time_s")
    if type(step) is not int or step < 0 or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
        raise ValueError("invalid accepted physical time")
    count = receipt.get("accepted_body_pose_count")
    poses = receipt.get("accepted_body_poses", [])
    if type(count) is not int or count <= 0 or len(poses) != count:
        raise ValueError("accepted body pose count")
    for label, rows in (("accepted", poses), ("registered", receipt.get("accepted_registered_body_poses", []))):
        ids = [r["body_index"] for r in rows]
        if any(type(v) is not int or not 0 <= v < count for v in ids) or len(set(ids)) != len(ids):
            raise ValueError(label + " body identity")
        if label == "accepted" and sorted(ids) != list(range(count)):
            raise ValueError("accepted body coverage")
        for row in rows:
            p, quat = row["position_m"], row["quaternion_xyzw"]
            if len(p) != 3 or len(quat) != 4 or not all(math.isfinite(v) for v in p + quat):
                raise ValueError(label + " body pose is nonfinite")
            if abs(sum(v * v for v in quat) - 1) > 2e-5:
                raise ValueError(label + " body quaternion is not unit")
    return {"accepted_step": step, "accepted_time_s": time,
            "accepted_body_pose_count": count,
            "registered_body_pose_count": len(receipt.get("accepted_registered_body_poses", []))}


def load_capture(pack, receipt_path):
    pack, receipt_path = Path(pack), Path(receipt_path)
    with pack.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = json.loads(receipt_path.read_text())
    endpoint = validate_capture_receipt(receipt, digest)
    sections = _pack_sections(pack)
    for kind, stride in ((2, 80), (3, 4), (4, 64)):
        raw, count, actual = sections[kind]
        if actual != stride or count * stride != len(raw):
            raise ValueError("native capture section layout")
    vertices = np.frombuffer(sections[2][0], dtype="<f4").reshape(-1, 20)
    indices = np.frombuffer(sections[3][0], dtype="<u4")
    primitives = np.frombuffer(sections[4][0], dtype="<u4").reshape(-1, 16)
    if hashlib.sha256(sections[2][0]).hexdigest() != receipt.get("captured_vertex_buffer_sha256"):
        raise ValueError("captured vertex buffer hash")
    for name, count in (("vertex_count", len(vertices)), ("index_count", len(indices)),
                        ("primitive_count", len(primitives))):
        if receipt.get(name) != count:
            raise ValueError("capture receipt " + name)
    if not np.isfinite(vertices[:, :3]).all() or not np.isfinite(vertices[:, 4:7]).all():
        raise ValueError("nonfinite capture coordinates or normals")
    grouped, owners = {}, {}
    for primitive in primitives:
        first, count = map(int, primitive[:2])
        key = (int(primitive[4]), int(primitive[7]))
        if key[0] not in ANATOMY_SEMANTICS:
            continue
        typed_key(key)
        # MRVisualPrimitiveGPUV2.identity is semantic, instance, link,
        # stable primitive ID. UINT32_MAX is the existing deformable owner.
        owner = int(primitive[6])
        if owner != 0xffffffff and owner >= endpoint["accepted_body_pose_count"]:
            raise ValueError("primitive rigid owner outside accepted pose buffer")
        if key in owners and owners[key] != owner:
            raise ValueError("typed surface has inconsistent rigid ownership")
        owners[key] = owner
        if count == 0 or count % 3 or first + count > len(indices):
            raise ValueError("native primitive triangle range")
        faces = indices[first:first + count].reshape(-1, 3)
        if faces.max() >= len(vertices):
            raise ValueError("native primitive global vertex range")
        grouped.setdefault(key, []).append(faces)
    surfaces = {}
    for key, chunks in grouped.items():
        global_faces = np.concatenate(chunks)
        used, inverse = np.unique(global_faces, return_inverse=True)
        surfaces[key] = (vertices[used, :3].copy(),
                         inverse.reshape(-1, 3).astype(np.uint32),
                         vertices[used, 4:7].copy(), used)
    return surfaces, {"pack_sha256": digest,
                      "capture_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                      **endpoint}


def audit_changed_surfaces(surfaces, changed, *, targets=None, interfaces=None, required_closed=(),
                           intended_enclosures=(), shared_feature_interfaces=None,
                           source_patch_complex_interfaces=None, progress=None):
    changed = [typed_key(key) for key in changed]
    targets = list(surfaces) if targets is None else [typed_key(key) for key in targets]
    interfaces = interfaces or {}
    shared_feature_interfaces = shared_feature_interfaces or {}
    source_patch_complex_interfaces = source_patch_complex_interfaces or {}
    enclosures = _intended_enclosures(intended_enclosures)
    if any(a not in surfaces or b not in surfaces for a, b in enclosures):
        raise ValueError("intended enclosure identity absent from capture")
    required_closed = {typed_key(key) for key in required_closed} | {key for key in changed if key[0] == 51011}
    cache, rows = {}, []
    if progress is not None and not callable(progress):
        raise TypeError("audit progress observer must be callable")
    def record(row):
        rows.append(row)
        if progress is not None:
            progress(row)
    def prepared(key):
        if key not in cache:
            value = surfaces[key]
            cache[key] = value if isinstance(value, PreparedSurface) else prepare_surface(
                key, value[0], value[1], normals=value[2], source_vertex_ids=value[3])
        return cache[key]
    seen = set()
    for key in changed:
        surface = prepared(key)
        if key in required_closed and not surface.closed:
            record({"stage": "preparation", **surface.diagnostics(),
                         "passed": False, "reason": "required closed surface is open"})
            return {"passed": False, "failed_stage": "preparation", "rows": rows}
        result = surface.self_audit()
        record({"stage": "self", **surface.diagnostics(), **result,
                     "passed": result["count"] == 0})
        if result["count"]:
            return {"passed": False, "failed_stage": "self", "rows": rows}
        for target in targets:
            if target == key or frozenset((key, target)) in seen:
                continue
            seen.add(frozenset((key, target)))
            value = surfaces[target]
            positions = value.positions if isinstance(value, PreparedSurface) else value[0]
            if (np.any(surface.positions.min(axis=0) > positions.max(axis=0)) or
                    np.any(positions.min(axis=0) > surface.positions.max(axis=0))):
                declared_enclosures = [pair for pair in enclosures
                                       if set(pair) == {key, target}]
                record({"stage": "nearby_pair", "identities": [list(key), list(target)],
                             "passed": not declared_enclosures, "aabb_disjoint": True,
                             "triangle_pair_count": 0,
                             "declared_enclosure_outside_aabb": bool(declared_enclosures),
                             "intended_enclosures": [[list(a), list(b)]
                                                     for a, b in declared_enclosures]})
                if declared_enclosures:
                    return {"passed": False, "failed_stage": "nearby_pair", "rows": rows}
                continue
            try:
                other = prepared(target)
            except ValueError as exc:
                row = {
                    "stage": "target_preparation",
                    "identities": [list(key), list(target)],
                    "failed_identity": list(target),
                    "passed": False,
                    "reason": "target surface preparation failed before pair predicates",
                    "preparation_error_type": type(exc).__name__,
                    "preparation_error": str(exc),
                }
                diagnostic = getattr(exc, "diagnostic", None)
                if diagnostic is not None:
                    row["target_topology_witness"] = diagnostic
                record(row)
                return {"passed": False, "failed_stage": "target_preparation", "rows": rows}
            pair = audit_surface_pair(surface, other,
                                      intended_face_pairs=interfaces.get((key, target), ()),
                                      intended_shared_feature_pairs=shared_feature_interfaces.get((key, target), ()),
                                      intended_source_patch_complexes=source_patch_complex_interfaces.get(
                                          (key, target), ()),
                                      intended_enclosures=[pair for pair in enclosures
                                                           if set(pair) == {key, target}])
            record({"stage": "nearby_pair", **pair})
            if not pair["passed"]:
                return {"passed": False, "failed_stage": "nearby_pair", "rows": rows}
    return {"passed": True, "rows": rows,
            "scope": "Selected changed surfaces against declared targets; no vessel continuity or full anatomy admission."}


def _sequence_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sequence_prepared_surface(key, surface):
    if not isinstance(surface, PreparedSurface):
        raise TypeError("sequence overrides must be existing PreparedSurface objects")
    if typed_key(surface.key) != key:
        raise ValueError("prepared surface typed identity differs from its sequence key")
    positions = np.asarray(surface.positions)
    faces = np.asarray(surface.faces)
    source_ids = np.asarray(surface.source_vertex_ids)
    if (positions.dtype != np.dtype("float32") or positions.ndim != 2 or positions.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or source_ids.shape != (len(positions),) or source_ids.dtype.kind not in "iu"
            or len(np.unique(source_ids)) != len(source_ids) or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(positions))):
        raise ValueError("prepared surface arrays are not aligned")
    metadata = (surface.source_face_lineage, surface.source_vertex_lineage,
                surface.ordered_core_body_indices, surface.core_weights_by_gid)
    if any(value is not None for value in metadata):
        if any(value is None for value in metadata):
            raise ValueError("prepared source and material lineage is incomplete")
        if len(surface.source_face_lineage) != len(faces):
            raise ValueError("prepared face lineage is not aligned with face order")
        declared = surface.declared_core_body_indices
        if declared is None:
            if len(surface.ordered_core_body_indices) != 21:
                raise ValueError("prepared ordered material owners are not the existing 21-owner format")
        elif (_declared_material_inventory(declared) != declared
              or surface.ordered_core_body_indices != declared):
            raise ValueError("prepared material inventory differs from its declared owner order")
        referenced = {int(source_ids[int(local)]) for face in faces for local in face}
        if referenced - surface.source_vertex_lineage.keys():
            raise ValueError("prepared vertex lineage does not cover referenced source IDs")
        if referenced - surface.core_weights_by_gid.keys():
            raise ValueError("prepared material weights do not cover referenced source IDs")
        for gid in referenced:
            weights = np.asarray(surface.core_weights_by_gid[gid])
            if (weights.dtype != np.dtype("float64")
                    or weights.shape != (len(surface.ordered_core_body_indices),)
                    or not np.isfinite(weights).all()
                    or np.any(weights < 0 if declared is not None else weights <= 0)
                    or abs(float(weights.sum()) - 1.0) > 1e-12):
                raise ValueError("prepared material weights changed or are invalid")
        if any(not _has_source_face_lineage(row) for row in surface.source_face_lineage):
            raise ValueError("prepared source face lineage is incomplete")
    return {"typed_identity": list(key),
            "prepared_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "prepared_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "prepared_vertex_count": len(positions),
            "prepared_face_count": len(faces),
            "source_face_lineage_count": None if surface.source_face_lineage is None
                                      else len(surface.source_face_lineage),
            "material_lineage_present": surface.core_weights_by_gid is not None,
            "declared_material_owner_order": (None if surface.declared_core_body_indices is None
                                               else list(surface.declared_core_body_indices))}


def _bind_prepared_surface_to_capture(key, surface, captured):
    positions, faces, normals, captured_global_vertex_ids = captured
    captured_positions, captured_faces = np.asarray(positions), np.asarray(faces)
    captured_normals = np.asarray(normals)
    captured_global_vertex_ids = np.asarray(captured_global_vertex_ids)
    if (captured_positions.dtype != np.dtype("float32")
            or captured_normals.dtype != np.dtype("float32")
            or captured_positions.shape != captured_normals.shape
            or captured_faces.ndim != 2 or captured_faces.shape[1] != 3
            or captured_global_vertex_ids.shape != (len(captured_positions),)):
        raise ValueError("native captured surface arrays are not aligned")
    candidate_faces = np.asarray(surface.faces, dtype=np.int64)
    if candidate_faces.shape != captured_faces.shape:
        raise ValueError("prepared/captured ordered face counts differ")
    capture_global_faces = captured_global_vertex_ids[captured_faces].astype(np.int64)
    _, face_binding = retain_bound_source_faces(
        key, captured, candidate_faces, 0,
        np.arange(len(candidate_faces), dtype=np.uint32))
    namespace_offset = face_binding["source_capture_vertex_namespace_offset"]
    capture_corners = np.searchsorted(captured_global_vertex_ids,
                                      capture_global_faces.reshape(-1))
    if (np.any(capture_corners >= len(captured_global_vertex_ids))
            or not np.array_equal(captured_global_vertex_ids[capture_corners],
                                  capture_global_faces.reshape(-1))):
        raise ValueError("prepared/captured face references a missing capture vertex")
    prepared_corners = candidate_faces.reshape(-1)
    prepared_positions = surface.positions[prepared_corners]
    native_positions = captured_positions[capture_corners]
    if prepared_positions.tobytes() != native_positions.tobytes():
        raise ValueError("prepared/captured Float32 XYZ differs at referenced face corners")
    if surface.normals is None:
        raise ValueError("prepared sequence surface lacks emitted Float32 normals")
    prepared_normals = np.asarray(surface.normals)
    if prepared_normals.dtype != np.dtype("float32") or prepared_normals.shape != surface.positions.shape:
        raise ValueError("prepared sequence normals are not aligned Float32 XYZ")
    native_normal_corners = captured_normals[capture_corners]
    prepared_normal_corners = prepared_normals[prepared_corners]
    if prepared_normal_corners.tobytes() != native_normal_corners.tobytes():
        raise ValueError("prepared/captured Float32 normals differ at referenced face corners")
    return {"typed_identity": list(key),
            "source_capture_vertex_namespace_offset": namespace_offset,
            "all_ordered_face_corners_bound": True,
            "referenced_corner_count": len(prepared_corners),
            "positions_exact_at_referenced_corners": True,
            "normals_exact_at_referenced_corners": True,
            "prepared_positions_sha256": hashlib.sha256(prepared_positions.tobytes()).hexdigest(),
            "captured_positions_sha256": hashlib.sha256(native_positions.tobytes()).hexdigest(),
            "prepared_normals_sha256": hashlib.sha256(prepared_normal_corners.tobytes()).hexdigest(),
            "captured_normals_sha256": hashlib.sha256(native_normal_corners.tobytes()).hexdigest()}


def audit_capture_sequence(capture_manifest, *, expected_manifest_sha256,
                           expected_source_sha256, changed, prepared_surfaces_by_step=None,
                           targets=None, interfaces_by_step=None, required_closed=(),
                           intended_enclosures_by_step=None,
                           shared_feature_interfaces_by_step=None):
    """Run the existing exact audit in accepted-capture order and stop at first failure.

    Prepared faces remain in the source owner's local vertex namespace; the existing
    complete ordered-face mapper proves the capture namespace offset before XYZ/normal
    bytes are compared at every referenced corner. Source GIDs remain lineage keys.
    """
    manifest_path = Path(capture_manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("accepted capture manifest SHA-256 mismatch")
    if not isinstance(expected_source_sha256, dict):
        raise TypeError("source SHA pins must map existing input paths to hashes")
    if not expected_source_sha256:
        raise ValueError("at least one existing source SHA-256 pin is required")
    input_pins = {}
    for raw_path, expected in sorted(expected_source_sha256.items(), key=lambda item: str(item[0])):
        path = Path(raw_path)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)):
            raise ValueError("invalid source SHA-256 pin")
        actual = _sequence_sha256(path)
        if actual != expected:
            raise ValueError("source input SHA-256 mismatch: " + str(path))
        input_pins[str(path)] = actual
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("accepted capture index must be an object")
    capture_rows = manifest.get("capture_rows")
    if not isinstance(capture_rows, list) or not capture_rows:
        raise ValueError("accepted capture index has no capture_rows")
    steps, previous_time = [], -math.inf
    for row in capture_rows:
        if not isinstance(row, dict):
            raise TypeError("accepted capture row must be an object")
        step, accepted_time = row.get("step"), row.get("accepted_time_s")
        if (type(step) is not int or step < 0 or type(accepted_time) not in (int, float)):
            raise ValueError("accepted capture step/time identity")
        if not math.isfinite(float(accepted_time)) or float(accepted_time) <= previous_time:
            raise ValueError("accepted capture rows must have finite increasing times")
        previous_time = float(accepted_time)
        for path_key, sha_key in (("pack_path", "recorded_pack_sha256"),
                                  ("receipt_path", "receipt_sha256")):
            digest = row.get(sha_key)
            if (not isinstance(row.get(path_key), str) or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError("accepted capture path or SHA-256 pin is invalid: " + path_key)
        steps.append(step)
    if len(set(steps)) != len(steps):
        raise ValueError("accepted capture steps are duplicated")
    if steps != sorted(steps):
        raise ValueError("accepted capture steps must be increasing")
    changed_keys = [typed_key(key) for key in changed]
    if not changed_keys or len(set(changed_keys)) != len(changed_keys):
        raise ValueError("changed typed surface identities are empty or duplicated")
    prepared_by_step = None
    if prepared_surfaces_by_step is not None:
        if (not isinstance(prepared_surfaces_by_step, dict)
                or any(type(step) is not int for step in prepared_surfaces_by_step)
                or set(prepared_surfaces_by_step) != set(steps)):
            raise ValueError("prepared surfaces must cover exactly the manifest capture steps")
        prepared_by_step = {}
        for step in steps:
            entries = prepared_surfaces_by_step[step]
            if not isinstance(entries, dict):
                raise TypeError("prepared surfaces for a capture must be a typed mapping")
            normalized = {}
            for raw_key, surface in entries.items():
                key = typed_key(raw_key)
                if key in normalized:
                    raise ValueError("duplicate prepared typed surface identity")
                normalized[key] = surface
                _validate_sequence_prepared_surface(key, surface)
            if set(normalized) != set(changed_keys):
                raise ValueError("prepared typed surface identities do not match changed set")
            prepared_by_step[step] = normalized
    interfaces_by_step = interfaces_by_step or {}
    targets = None if targets is None else [typed_key(key) for key in targets]
    required_closed = [typed_key(key) for key in required_closed]
    if any(type(step) is not int for step in interfaces_by_step) or set(interfaces_by_step) - set(steps):
        raise ValueError("interface declarations refer to an absent capture step")
    shared_feature_interfaces_by_step = shared_feature_interfaces_by_step or {}
    if (any(type(step) is not int for step in shared_feature_interfaces_by_step)
            or set(shared_feature_interfaces_by_step) - set(steps)):
        raise ValueError("shared feature declarations refer to an absent capture step")
    intended_enclosures_by_step = intended_enclosures_by_step or {}
    if (any(type(step) is not int for step in intended_enclosures_by_step)
            or set(intended_enclosures_by_step) - set(steps)):
        raise ValueError("enclosure declarations refer to an absent capture step")
    for declarations in intended_enclosures_by_step.values():
        for pair in declarations:
            if len(pair) != 2:
                raise ValueError("intended enclosure must contain two typed surface identities")
            typed_key(pair[0])
            typed_key(pair[1])
    completed, capture_pins = [], []
    for capture_index, row in enumerate(capture_rows):
        step = row["step"]
        surfaces, identity = load_capture(row["pack_path"], row["receipt_path"])
        if (identity.get("pack_sha256") != row["recorded_pack_sha256"]
                or identity.get("capture_receipt_sha256") != row["receipt_sha256"]
                or identity.get("accepted_step") != step
                or float(identity.get("accepted_time_s")) != float(row["accepted_time_s"])):
            return {"passed": False, "failed_stage": "capture_pin",
                    "capture_index": capture_index, "capture_step": step,
                    "capture_identity": identity, "manifest_row": row,
                    "completed_captures": completed, "manifest_sha256": manifest_sha,
                    "source_sha256": input_pins}
        bindings = []
        if prepared_by_step is not None:
            surfaces = dict(surfaces)
            for key in changed_keys:
                if key not in surfaces:
                    return {"passed": False, "failed_stage": "capture_identity",
                            "capture_index": capture_index, "capture_step": step,
                            "missing_typed_identity": list(key), "completed_captures": completed}
                try:
                    proof = _bind_prepared_surface_to_capture(
                        key, prepared_by_step[step][key], surfaces[key])
                except ValueError as error:
                    return {"passed": False, "failed_stage": "surface_binding",
                            "capture_index": capture_index, "capture_step": step,
                            "typed_identity": list(key), "error": str(error),
                            "completed_captures": completed}
                surfaces[key] = prepared_by_step[step][key]
                bindings.append(proof)
        result = audit_changed_surfaces(
            surfaces, changed_keys, targets=targets,
            interfaces=interfaces_by_step.get(step),
            shared_feature_interfaces=shared_feature_interfaces_by_step.get(step),
            required_closed=required_closed,
            intended_enclosures=intended_enclosures_by_step.get(step, ()))
        capture_pins.append({"capture_index": capture_index, "capture_step": step,
                             "accepted_time_s": float(row["accepted_time_s"]),
                             "pack_sha256": identity["pack_sha256"],
                             "capture_receipt_sha256": identity["capture_receipt_sha256"]})
        completed.append({"capture_index": capture_index, "capture_step": step,
                          "surface_bindings": bindings, "audit": result})
        if not result.get("passed", False):
            return {"passed": False, "failed_stage": result.get("failed_stage", "geometry"),
                    "capture_index": capture_index, "capture_step": step,
                    "completed_captures": completed, "capture_pins": capture_pins,
                    "manifest_sha256": manifest_sha, "source_sha256": input_pins}
    return {"passed": True, "completed_captures": completed,
            "capture_pins": capture_pins, "manifest_sha256": manifest_sha,
            "source_sha256": input_pins,
            "scope": "Existing accepted captures, exact owner-bound surfaces, and selected changed-surface audits only."}


def validate_rest_emission(emitted, rigid_rest_reconstruction, *, tolerance_m=2e-6):
    """Compare native emission with an existing NHRIGID-owner reconstruction.

    The caller supplies the existing owner's ordered Float32 result; this
    verifier does not implement another pose or material-parent transform.
    """
    first, second = np.asarray(emitted), np.asarray(rigid_rest_reconstruction)
    if first.dtype != np.dtype("float32") or second.dtype != np.dtype("float32"):
        raise ValueError("rest comparison requires exact Float32 emission")
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3 or not len(first):
        raise ValueError("rest comparison shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("rest comparison is nonfinite")
    error = np.linalg.norm(first.astype(float) - second.astype(float), axis=1)
    return {"passed": bool(np.all(error <= tolerance_m)),
            "max_rest_error_m": float(error.max()), "tolerance_m": tolerance_m,
            "outside_tolerance_vertex_ids": np.flatnonzero(error > tolerance_m).tolist(),
            "emitted_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "rigid_rest_reconstruction_sha256": hashlib.sha256(second.tobytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    capture = parser.add_mutually_exclusive_group(required=True)
    capture.add_argument("--pack", type=Path)
    capture.add_argument("--capture-manifest", type=Path,
                         help="existing accepted-capture index with capture_rows")
    parser.add_argument("--capture-receipt", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--source-pin", action="append", default=[],
                        help="existing source PATH=SHA256; required for a capture sequence")
    parser.add_argument("--required-closed", action="append", default=[],
                        help="semantic:stable_id; reject open changed surfaces")
    parser.add_argument("--changed", action="append", required=True, help="semantic:stable_id")
    parser.add_argument("--target", action="append", help="semantic:stable_id; default all anatomy")
    parser.add_argument("--intended-enclosure", action="append", default=[],
                        help="enclosed SEM:ID,container SEM:ID; permits containment, never crossings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key = lambda value: typed_key(tuple(int(v) for v in value.split(":")))
    if args.pack is not None:
        if args.capture_receipt is None:
            parser.error("--pack requires --capture-receipt")
        if args.manifest_sha256 is not None or args.source_pin:
            parser.error("sequence pins require --capture-manifest")
    elif args.capture_receipt is not None or not args.manifest_sha256 or not args.source_pin:
        parser.error("--capture-manifest requires --manifest-sha256 and --source-pin, without --capture-receipt")
    identity = {"pack": str(args.pack), "capture_receipt": str(args.capture_receipt)}
    if args.capture_manifest is not None:
        identity = {"capture_manifest": str(args.capture_manifest)}
    try:
        enclosures = []
        for value in args.intended_enclosure:
            pair = value.split(",")
            if len(pair) != 2:
                raise ValueError("intended enclosure requires two comma-separated typed identities")
            enclosures.append(tuple(key(v) for v in pair))
        changed = [key(v) for v in args.changed]
        targets = None if args.target is None else [key(v) for v in args.target]
        closed = [key(v) for v in args.required_closed]
        if args.capture_manifest is None:
            surfaces, identity = load_capture(args.pack, args.capture_receipt)
            report = audit_changed_surfaces(surfaces, changed, targets=targets,
                                           required_closed=closed,
                                           intended_enclosures=enclosures)
        else:
            pins = {}
            for value in args.source_pin:
                source, separator, digest = value.rpartition("=")
                if not separator or not source or source in pins:
                    raise ValueError("source pins require unique PATH=SHA256 entries")
                pins[source] = digest
            manifest = json.loads(args.capture_manifest.read_bytes())
            declarations = {row["step"]: enclosures for row in manifest["capture_rows"]}
            report = audit_capture_sequence(
                args.capture_manifest, expected_manifest_sha256=args.manifest_sha256,
                expected_source_sha256=pins, changed=changed, targets=targets,
                required_closed=closed, intended_enclosures_by_step=declarations)
    except (TypeError, ValueError, KeyError, OSError, HumanImportError) as error:
        report = {"passed": False, "failed_stage": "preparation", "error": str(error)}
    report.update(identity)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


def retain_bound_source_faces(key, captured_surface, source_faces, source_vertex_offset,
                              retained_source_face_ids):
    """Apply source-facet ancestry only after proving the complete ordered ID mapping.

    Capture and NHA vertex IDs use different namespaces. One constant offset must
    explain every ordered facet corner; proximity and partial matching are rejected.
    Positions, normals and source vertex IDs remain the exact captured arrays.
    """
    key = typed_key(key)
    positions, faces, normals, gids = captured_surface
    sf = np.asarray(source_faces)
    selected = np.asarray(retained_source_face_ids)
    if sf.ndim != 2 or sf.shape[1] != 3 or not np.issubdtype(sf.dtype, np.integer):
        raise ValueError('source facet indices must be integer triangles')
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError('retained source facet ancestry must be integer ordinals')
    if len(selected) and (selected.min() < 0 or selected.max() >= len(sf)
                         or np.any(np.diff(selected.astype(np.int64)) <= 0)):
        raise ValueError('retained source facet ancestry must be ordered and unique')
    cf = np.asarray(faces)
    ids = np.asarray(gids)
    if cf.shape != sf.shape or not len(sf):
        raise ValueError('complete source/capture facet inventories disagree')
    global_capture = ids[cf].astype(np.int64)
    global_source = sf.astype(np.int64) + int(source_vertex_offset)
    offset = int(global_capture[0, 0]) - int(global_source[0, 0])
    if not np.array_equal(global_capture - offset, global_source):
        raise ValueError('complete ordered source/capture facet mapping disagrees')
    projected = (positions, cf[selected].copy(), normals, gids)
    return projected, {
        'typed_identity': list(key), 'source_capture_vertex_namespace_offset': offset,
        'all_ordered_source_facets_match': True,
        'source_face_count': len(sf), 'retained_source_face_count': len(selected),
        'source_face_ancestry': selected.tolist(),
        'captured_positions_normals_and_vertex_ids_unchanged': True,
        'scope': 'retained captured deformation only; native emission confirmation remains required',
    }
