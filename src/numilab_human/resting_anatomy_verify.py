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
            "unused_slot_policy": "Unreferenced slots retain source ancestry; they do not create a surface boundary.",
            "seam_policy": "Exact packed coordinate identity for predicates only; source faces and mesh bytes retained.",
        }


def prepare_surface(key, positions, faces, *, normals=None, source_vertex_ids=None,
                    source_face_lineage=None, source_vertex_lineage=None,
                    ordered_core_body_indices=None, core_weight_gids=None,
                    core_weights_float64=None):
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
        if np.any(np.abs(np.linalg.norm(normals.astype(float), axis=1) - 1) > 2e-3):
            raise ValueError("emitted normals are not unit")
    if source_vertex_ids is None:
        source_vertex_ids = np.arange(len(p), dtype=np.uint64)
    source_vertex_ids = np.asarray(source_vertex_ids)
    if source_vertex_ids.shape != (len(p),) or source_vertex_ids.dtype.kind not in "iu":
        raise ValueError("invalid source vertex ancestry")
    if len(np.unique(source_vertex_ids)) != len(p):
        raise ValueError("duplicate source vertex ancestry")
    metadata = (source_face_lineage, source_vertex_lineage,
                ordered_core_body_indices, core_weight_gids, core_weights_float64)
    has_material_proof = any(value is not None for value in metadata)
    if has_material_proof and any(value is None for value in metadata):
        raise ValueError("shared-interface proof requires complete source and material lineage")
    face_lineage = None
    vertex_lineage = None
    core_body_indices = None
    weights_by_gid = None
    if has_material_proof:
        face_lineage = tuple(source_face_lineage)
        if len(face_lineage) != len(f) or any(not isinstance(row, dict) for row in face_lineage):
            raise ValueError("source face lineage must map every emitted face")
        vertex_lineage = {}
        for gid, row in source_vertex_lineage.items():
            if type(gid) is not int or not isinstance(row, dict):
                raise ValueError("source vertex lineage must use integer GIDs and records")
            vertex_lineage[gid] = row
        core_body_indices = tuple(ordered_core_body_indices)
        if (len(core_body_indices) != 21 or any(type(value) is not int for value in core_body_indices)
                or len(set(core_body_indices)) != 21):
            raise ValueError("material owner order must contain 21 unique integer body indices")
        weight_gids = np.asarray(core_weight_gids)
        weights = np.asarray(core_weights_float64)
        if (weight_gids.ndim != 1 or weight_gids.dtype.kind not in "iu"
                or weights.shape != (len(weight_gids), 21)
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
               if np.any(weights_by_gid[gid] <= 0)
               or abs(float(weights_by_gid[gid].sum()) - 1.0) > 1e-12]
        if bad:
            raise ValueError("referenced material rows must be strictly positive and normalized: " + str(bad[:8]))
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
                           vertex_lineage, core_body_indices, weights_by_gid)


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
    return isinstance(parent, dict) and any(_lineage_value_present(parent.get(key)) for key in (
        "parent_original_source_face_index", "parent_candidate_face_index", "row_local_face_index"))


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
            and len(record.get("core_body_indices", ())) == 21
            and len(record.get("core_weights", ())) == 21)


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
        if (not np.array_equal(wa, wb) or np.any(wa <= 0)
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
                        or lineage_weights.shape != (21,)
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


def audit_surface_pair(first, second, *, intended_face_pairs=(), intended_enclosures=(),
                       intended_shared_feature_pairs=()):
    allowed = {tuple(pair) for pair in intended_face_pairs}
    enclosures = _intended_enclosures(intended_enclosures)
    if any(set(pair) != {first.key, second.key} for pair in enclosures):
        raise ValueError("intended enclosure must identify this surface pair")
    if not _overlap(first, second):
        return {"identities": [list(first.key), list(second.key)], "passed": True,
                "aabb_disjoint": True, "triangle_pair_count": 0}
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
        if type(first_face) is not int or type(second_face) is not int:
            raise ValueError("shared feature face indices must be integers")
        if not (0 <= first_face < len(first.faces) and 0 <= second_face < len(second.faces)):
            raise ValueError("shared feature face index out of range")
        pair = (first_face, second_face)
        if pair in declared:
            raise ValueError("duplicate shared feature face pair")
        declared.add(pair)
        if not isinstance(feature_gids, (list, tuple)):
            raise TypeError("shared feature GIDs must be a list or tuple")
        if pair not in pair_set:
            rejected_features.append({"face_pair": list(pair), "feature_gids": list(feature_gids),
                                      "reason": "declared_pair_has_no_exact_intersection"})
            continue
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
    unexpected = [pair for pair in pairs
                  if tuple(pair) not in allowed and tuple(pair) not in accepted_features]
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
    return {"identities": [list(first.key), list(second.key)],
            "passed": not unexpected and not inside and not outside_declared_enclosure and not unresolved,
            "triangle_pair_count": len(pairs), "unexpected_triangle_pairs": unexpected,
            "declared_interface_triangle_pairs": [p for p in pairs if tuple(p) in allowed],
            "declared_interface_feature_contacts": [accepted_features[p] for p in sorted(accepted_features)],
            "rejected_interface_feature_contacts": rejected_features,
            "containment": containment,
            "intended_enclosures": [[list(a), list(b)] for a, b in sorted(enclosures)],
            "boundary": "Declared enclosures permit interior placement only; every off-boundary face interior is checked and outside samples reject an intended enclosure. Shared feature contacts require exact GID, owner-map, source-lineage, and intersection proof. Declared face interfaces do not prove vessel lumen continuity."}


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
                           intended_enclosures=(), shared_feature_interfaces=None):
    changed = [typed_key(key) for key in changed]
    targets = list(surfaces) if targets is None else [typed_key(key) for key in targets]
    interfaces = interfaces or {}
    shared_feature_interfaces = shared_feature_interfaces or {}
    enclosures = _intended_enclosures(intended_enclosures)
    if any(a not in surfaces or b not in surfaces for a, b in enclosures):
        raise ValueError("intended enclosure identity absent from capture")
    required_closed = {typed_key(key) for key in required_closed} | {key for key in changed if key[0] == 51011}
    cache, rows = {}, []
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
            rows.append({"stage": "preparation", **surface.diagnostics(),
                         "passed": False, "reason": "required closed surface is open"})
            return {"passed": False, "failed_stage": "preparation", "rows": rows}
        result = surface.self_audit()
        rows.append({"stage": "self", **surface.diagnostics(), **result,
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
                rows.append({"stage": "nearby_pair", "identities": [list(key), list(target)],
                             "passed": True, "aabb_disjoint": True, "triangle_pair_count": 0})
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
                rows.append(row)
                return {"passed": False, "failed_stage": "target_preparation", "rows": rows}
            pair = audit_surface_pair(surface, other,
                                      intended_face_pairs=interfaces.get((key, target), ()),
                                      intended_shared_feature_pairs=shared_feature_interfaces.get((key, target), ()),
                                      intended_enclosures=[pair for pair in enclosures
                                                           if set(pair) == {key, target}])
            rows.append({"stage": "nearby_pair", **pair})
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
        if len(surface.ordered_core_body_indices) != 21:
            raise ValueError("prepared ordered material owners are not the existing 21-owner format")
        referenced = {int(source_ids[int(local)]) for face in faces for local in face}
        if referenced - surface.source_vertex_lineage.keys():
            raise ValueError("prepared vertex lineage does not cover referenced source IDs")
        if referenced - surface.core_weights_by_gid.keys():
            raise ValueError("prepared material weights do not cover referenced source IDs")
        if any(not _has_source_face_lineage(row) for row in surface.source_face_lineage):
            raise ValueError("prepared source face lineage is incomplete")
    return {"typed_identity": list(key),
            "prepared_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "prepared_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "prepared_vertex_count": len(positions),
            "prepared_face_count": len(faces),
            "source_face_lineage_count": None if surface.source_face_lineage is None
                                      else len(surface.source_face_lineage),
            "material_lineage_present": surface.core_weights_by_gid is not None}


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
