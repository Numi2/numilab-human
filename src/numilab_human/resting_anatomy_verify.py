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

ANATOMY_SEMANTICS = frozenset((51004, 51005, 51006, 51007, 51010, 51011, 51012,
                               51020, 51021, 51022, 51023, 51024, 51025))


def typed_key(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("surface identity requires (semantic, stable_id)")
    if any(type(v) is not int for v in value):
        raise ValueError("surface identity must contain integers")
    if value[0] not in ANATOMY_SEMANTICS or value[1] <= 0:
        raise ValueError("invalid typed anatomy identity")
    return tuple(value)


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
    _self_result: dict | None = field(default=None, repr=False)
    _meter_index: object | None = field(default=None, repr=False)

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


def prepare_surface(key, positions, faces, *, normals=None, source_vertex_ids=None):
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
    for topology in (raw, quotient):
        if topology["degenerate_face_ids"] or topology["duplicate_face_ids"]:
            raise ValueError("degenerate or duplicate source faces")
    if quotient["nonmanifold_edges"] or quotient["orientation_defect_edges"]:
        raise ValueError("exact seam quotient is nonmanifold or inconsistently oriented")
    records = tuple(ci._records([keys[i] for i in keep], qf.tolist()))
    return PreparedSurface(key, np.array(p, copy=True, order="C"), np.array(f, dtype=np.uint32, copy=True, order="C"),
                           source_vertex_ids.copy(), normals, raw, quotient, qp, qf, mapping,
                           records, ci._prepare_surface_aabb(records))


def _overlap(first, second):
    return bool(np.all(first.positions.min(axis=0) <= second.positions.max(axis=0)) and
                np.all(second.positions.min(axis=0) <= first.positions.max(axis=0)))


def _contained_components(source, target):
    if not target.closed:
        return {"status": "not_checked_open_target", "inside_component_witnesses": []}
    if target.self_audit()["count"]:
        return {"status": "not_checked_nonembedded_target", "inside_component_witnesses": []}
    prepared = target.meter_index()
    witnesses = []
    for component in source.quotient_topology["face_components"]:
        face_id = component[0]
        vertex_id = int(source.quotient_faces[face_id, 0])
        point = tuple(Fraction(float(v)) for v in source.quotient_positions[vertex_id])
        result = ci.point_location_prepared_indexed(point, prepared)
        if result["location"] not in ("inside", "outside", "boundary"):
            return {"status": "unresolved_exact_point_location", "inside_component_witnesses": witnesses}
        if result["location"] == "inside":
            witnesses.append({"source_face_id": int(face_id),
                              "source_quotient_vertex_id": vertex_id,
                              "point_m": [float(v) for v in point]})
    return {"status": "checked_exact_rational_metres", "inside_component_witnesses": witnesses}


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


def audit_surface_pair(first, second, *, intended_face_pairs=(), intended_enclosures=()):
    allowed = {tuple(pair) for pair in intended_face_pairs}
    enclosures = _intended_enclosures(intended_enclosures)
    if any(set(pair) != {first.key, second.key} for pair in enclosures):
        raise ValueError("intended enclosure must identify this surface pair")
    if not _overlap(first, second):
        return {"identities": [list(first.key), list(second.key)], "passed": True,
                "aabb_disjoint": True, "triangle_pair_count": 0}
    result = ci._audit_pair_prepared_first(first.index, second.records, same_surface=False)
    pairs = result["triangle_pairs"]
    unexpected = [pair for pair in pairs if tuple(pair) not in allowed]
    # Complete containment is invisible to triangle-pair counts. With no
    # boundary crossing, location is constant on each connected component.
    containment = []
    if not pairs:
        containment = [_contained_components(first, second), _contained_components(second, first)]
    for row, (source, target) in zip(containment, ((first, second), (second, first))):
        row["source_identity"] = list(source.key)
        row["target_identity"] = list(target.key)
        row["declared_intended_enclosure"] = (source.key, target.key) in enclosures
    inside = any(row["inside_component_witnesses"] and
                 not row["declared_intended_enclosure"] for row in containment)
    unresolved = any(row["status"] in ("not_checked_nonembedded_target",
                                     "unresolved_exact_point_location") for row in containment)
    return {"identities": [list(first.key), list(second.key)],
            "passed": not unexpected and not inside and not unresolved,
            "triangle_pair_count": len(pairs), "unexpected_triangle_pairs": unexpected,
            "declared_interface_triangle_pairs": [p for p in pairs if tuple(p) in allowed],
            "containment": containment,
            "intended_enclosures": [[list(a), list(b)] for a, b in sorted(enclosures)],
            "boundary": "Declared enclosures permit interior placement only; all boundary crossings remain checked. Declared face interfaces do not prove vessel lumen continuity."}


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
                           intended_enclosures=()):
    changed = [typed_key(key) for key in changed]
    targets = list(surfaces) if targets is None else [typed_key(key) for key in targets]
    interfaces = interfaces or {}
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
            other = prepared(target)
            pair = audit_surface_pair(surface, other,
                                      intended_face_pairs=interfaces.get((key, target), ()),
                                      intended_enclosures=[pair for pair in enclosures
                                                           if set(pair) == {key, target}])
            rows.append({"stage": "nearby_pair", **pair})
            if not pair["passed"]:
                return {"passed": False, "failed_stage": "nearby_pair", "rows": rows}
    return {"passed": True, "rows": rows,
            "scope": "Selected changed surfaces against declared targets; no vessel continuity or full anatomy admission."}


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
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--changed", action="append", required=True, help="semantic:stable_id")
    parser.add_argument("--target", action="append", help="semantic:stable_id; default all anatomy")
    parser.add_argument("--intended-enclosure", action="append", default=[],
                        help="enclosed SEM:ID,container SEM:ID; permits containment, never crossings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    key = lambda value: typed_key(tuple(int(v) for v in value.split(":")))
    identity = {"pack": str(args.pack), "capture_receipt": str(args.capture_receipt)}
    try:
        surfaces, identity = load_capture(args.pack, args.capture_receipt)
        enclosures = []
        for value in args.intended_enclosure:
            pair = value.split(",")
            if len(pair) != 2:
                raise ValueError("intended enclosure requires two comma-separated typed identities")
            enclosures.append(tuple(key(v) for v in pair))
        report = audit_changed_surfaces(surfaces, [key(v) for v in args.changed],
                                       targets=None if args.target is None else [key(v) for v in args.target],
                                       intended_enclosures=enclosures)
    except (ValueError, KeyError, OSError, HumanImportError) as error:
        report = {"passed": False, "failed_stage": "preparation", "error": str(error)}
    report.update(identity)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
