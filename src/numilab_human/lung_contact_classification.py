"""Exact feature classification for packed-Float32 lung contact witnesses.

Classes describe exact lattice geometry; they are not physical-intersection
waivers. Callers must retain raw events and count every label outside the
pair-specific allowlist as unclassified.
"""

from __future__ import annotations
from collections import Counter
from typing import Iterable, Sequence

Point = tuple[int, int, int]
Edge = tuple[Point, Point]
UNCLASSIFIED = "unmapped_or_unclassified_diaphragm_contact"
UNCLASSIFIED_LOBE = "unclassified_cross_intersection"

ALLOWED_LABELS = {
    "lobe_lobe": frozenset({
        "shared_exact_vertex", "shared_exact_edge", "coincident_exact_triangle",
    }),
    "diaphragm_lobe": frozenset({
        "exact_reciprocal_mapped_face", "mapped_face_shared_edge_adjacency",
        "mapped_face_shared_vertex_adjacency", "exact_declared_interface_boundary_contact",
    }),
}


def point_on_segment(point: Point, a: Point, b: Point) -> bool:
    """Return exact integer-lattice membership on the closed segment a-b."""
    d = tuple(b[k] - a[k] for k in range(3))
    q = tuple(point[k] - a[k] for k in range(3))
    cross = (
        q[1] * d[2] - q[2] * d[1],
        q[2] * d[0] - q[0] * d[2],
        q[0] * d[1] - q[1] * d[0],
    )
    if cross != (0, 0, 0):
        return False
    dot = sum(q[k] * d[k] for k in range(3))
    length2 = sum(x * x for x in d)
    return 0 <= dot <= length2


def _points_on_boundary(points: Sequence[Point], boundary_edges: Iterable[Edge]) -> bool:
    """Require the whole point or segment witness on one concrete boundary edge."""
    edges = tuple(boundary_edges)
    return bool(points) and any(
        all(point_on_segment(p, a, b) for p in points)
        for a, b in edges
    )


def classify_lobe_lobe(
    tri_a: Sequence[Point], tri_b: Sequence[Point], intersection_points: Sequence[Point]
) -> str:
    """Describe only the exact shared coordinate feature for a lobe pair."""
    if not intersection_points:
        raise ValueError("an exact intersection event must contain a witness point")
    common = set(tri_a) & set(tri_b)
    if len(common) == 1 and all(p in common for p in intersection_points):
        return "shared_exact_vertex"
    if len(common) == 2 and all(point_on_segment(p, *tuple(common)) for p in intersection_points):
        return "shared_exact_edge"
    if len(common) == 3:
        return "coincident_exact_triangle"
    return UNCLASSIFIED_LOBE


def classify_diaphragm_lobe(
    mapped_lobe_triangle: Sequence[Point],
    candidate_lobe_triangle: Sequence[Point],
    intersection_points: Sequence[Point],
    declared_union_boundary_edges: Iterable[Edge],
    *,
    is_reciprocal_mapped_face: bool = False,
) -> str:
    """Classify a D/lobe witness against map support or exact declared boundary.

    `mapped_lobe_triangle` is the lobe face reciprocally mapped to the D face;
    `candidate_lobe_triangle` is the lobe face participating in this witness.
    Boundary contact is accepted only when each exact witness point lies on a
    concrete edge in the mapped-union boundary. Empty boundary input can never
    authorize a contact.
    """
    if not intersection_points:
        raise ValueError("an exact intersection event must contain a witness point")
    if is_reciprocal_mapped_face:
        return "exact_reciprocal_mapped_face"
    support = set(mapped_lobe_triangle) & set(candidate_lobe_triangle)
    if len(support) == 1 and all(p in support for p in intersection_points):
        return "mapped_face_shared_vertex_adjacency"
    if len(support) == 2 and all(point_on_segment(p, *tuple(support)) for p in intersection_points):
        return "mapped_face_shared_edge_adjacency"
    if _points_on_boundary(intersection_points, declared_union_boundary_edges):
        return "exact_declared_interface_boundary_contact"
    return UNCLASSIFIED


def count_unclassified(pair_kind: str, labels: Iterable[str]) -> int:
    """Count every event whose label is not explicitly allowed for this pair."""
    allowed = ALLOWED_LABELS.get(pair_kind, frozenset())
    return sum(label not in allowed for label in labels)


def classification_counts(pair_kind: str, labels: Iterable[str]) -> tuple[dict[str, int], int]:
    """Return event histogram and fail-closed unclassified count."""
    counts = Counter(labels)
    unclassified = sum(count for label, count in counts.items()
                       if label not in ALLOWED_LABELS.get(pair_kind, frozenset()))
    return dict(counts), unclassified
