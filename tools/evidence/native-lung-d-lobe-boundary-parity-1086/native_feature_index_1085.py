"""1085 native feature adapter: 1084 indexing plus published exact D/lobe rule.

The pinned 1042 exact predicates and 1084 native-world feature indexing remain
unchanged. For D311/lobe events, this adapter only revisits an event when the
1084 classifier leaves it unclassified, then applies the published exact
source-contact helper against the validated native mapped-union boundary.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Iterable, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_FEATURE_1084 = HERE.parent / "native-lung-exact-classifier-index-1084/native_feature_index.py"
DEFAULT_SOURCE_RULE = Path("/Users/n/numi-human-lung-contact-classifier-001/src/numilab_human/lung_contact_classification.py")
FEATURE_1084_SHA256 = "2d9eab88a802a0223cad34f8b9d42066e55214d53dab5b7e0fc0f9b6aac9cd5e"
SOURCE_RULE_SHA256 = "fd95fa6cca92706faa9c7dcf88fbe2243d1a7d865b391d2f632cc0230cf42710"
SOURCE_RULE_REVISION = "46b3c0b18e5cb9c23a054706aafa72068c439dd2"

_SOURCE_TO_NATIVE_LABEL = {
    "exact_reciprocal_mapped_face": "exact_reciprocal_face",
    "mapped_face_shared_edge_adjacency": "mapped_lobe_face_adjacency",
    "mapped_face_shared_vertex_adjacency": "mapped_lobe_face_adjacency",
    "exact_declared_interface_boundary_contact": "exact_fullunion_boundary_contact",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned classifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _integer_lattice_points(points: Sequence[Sequence[object]]):
    """Convert only exact integer lattice coordinates; never round a witness."""
    converted = []
    for point in points:
        row = []
        for value in point:
            if isinstance(value, int):
                row.append(value)
                continue
            numerator = getattr(value, "numerator", None)
            denominator = getattr(value, "denominator", None)
            if numerator is None or denominator != 1:
                return None
            row.append(int(numerator))
        if len(row) != 3:
            return None
        converted.append(tuple(row))
    return tuple(converted) if converted else None


def _triangle_lattice_points(base, pose_row, face_row: int):
    faces = pose_row.get("f")
    vertices = pose_row.get("v")
    if faces is None or vertices is None or not 0 <= int(face_row) < len(faces):
        return None
    return tuple(base.pkey(vertices[int(vertex), :3]) for vertex in faces[int(face_row)])


def install(base, *, feature_1084=None, source_rule=None):
    """Install the 1084 adapter, then strict source-rule D311/lobe fallback.

    `base` is the existing pinned 1042 audit module. Optional module arguments
    support focused tests; default use loads hash-pinned immutable dependencies.
    """
    if feature_1084 is None:
        if _sha256(DEFAULT_FEATURE_1084) != FEATURE_1084_SHA256:
            raise ValueError("pinned native feature index 1084 changed")
        feature_1084 = _load_module(DEFAULT_FEATURE_1084, "native_feature_index_1084_pinned")
    if source_rule is None:
        if _sha256(DEFAULT_SOURCE_RULE) != SOURCE_RULE_SHA256:
            raise ValueError("published exact lung contact helper changed")
        source_rule = _load_module(DEFAULT_SOURCE_RULE, "lung_contact_classification_46b3c0_pinned")

    feature_1084.install(base)
    indexed_classify = base.classify

    def classify(a, b, ra, rb, points, pose, maps, lobe_maps):
        # Keep all lobe/lobe behavior from the already-tested 1084 feature index.
        if 311 not in (a, b):
            return indexed_classify(a, b, ra, rb, points, pose, maps, lobe_maps)

        # Keep every class already admitted by 1084, including reciprocal face
        # mapping and mapped-face adjacency. The source helper is an exact fallback
        # only for 1084's otherwise-unclassified D/lobe witnesses.
        legacy_label = indexed_classify(a, b, ra, rb, points, pose, maps, lobe_maps)
        if legacy_label != "unclassified_cross_intersection":
            return legacy_label

        if a == 311:
            d_record, l_record, lobe = ra, rb, b
        else:
            l_record, d_record, lobe = ra, rb, a
        native_map = maps.get(lobe)
        if not isinstance(native_map, dict) or not native_map.get("valid"):
            return "unclassified_current_map_mismatch"
        boundary_edges = native_map.get("db")
        if boundary_edges is None or native_map.get("db") != native_map.get("lb"):
            return "unclassified_current_map_mismatch"

        lattice_points = _integer_lattice_points(points)
        if lattice_points is None:
            # The published API accepts exact binary32-lattice integer points.
            # Fractional intersection witnesses fail closed; never quantize them.
            return "unclassified_cross_intersection"

        d_face = int(d_record[3])
        lobe_face = int(l_record[3])
        mapped_lobe_face = native_map.get("d2l", {}).get(d_face)
        mapped_triangle = () if mapped_lobe_face is None else _triangle_lattice_points(
            base, pose[lobe], int(mapped_lobe_face))
        candidate_triangle = _triangle_lattice_points(base, pose[lobe], lobe_face)
        if mapped_triangle is None or candidate_triangle is None:
            return "unclassified_current_map_mismatch"

        source_label = source_rule.classify_diaphragm_lobe(
            mapped_triangle,
            candidate_triangle,
            lattice_points,
            boundary_edges,
            is_reciprocal_mapped_face=(mapped_lobe_face == lobe_face),
        )
        return _SOURCE_TO_NATIVE_LABEL.get(source_label, "unclassified_cross_intersection")

    base.classify = classify
    base.native_source_contact_rule_revision = SOURCE_RULE_REVISION
    return base
