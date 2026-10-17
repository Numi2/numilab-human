"""Strict local edge-collapse construction for emitted forearm surfaces.

This owner performs only a topology change. It never moves or merges source
points, canonical identities, or full21 material terms.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np


def referenced_surface_topology_gate(prepared_surface) -> dict:
    """Report the shared owner's referenced-surface topology result.

    PreparedSurface.closed excludes unreferenced allocated source slots,
    whose ancestry must remain intact. Do not use the aggregate topology flag:
    it marks those retained slots as unused even when the rendered surface is
    closed and oriented.
    """
    topology = prepared_surface.quotient_topology
    closed = bool(prepared_surface.closed)
    return {
        "passed_referenced_surface_topology": closed,
        "referenced_surface_closed": closed,
        "face_component_count": int(topology.get("face_component_count", 0)),
        "unused_allocated_source_slot_count": len(topology.get("unused_vertex_ids", ())),
        "boundary_edge_count": len(topology.get("boundary_edges", ())),
        "nonmanifold_edge_count": len(topology.get("nonmanifold_edges", ())),
        "orientation_defect_edge_count": len(topology.get("orientation_defect_edges", ())),
        "degenerate_face_count": len(topology.get("degenerate_face_ids", ())),
        "repeated_vertex_face_count": len(topology.get("repeated_vertex_face_ids", ())),
        "duplicate_face_count": len(topology.get("duplicate_face_ids", ())),
        "vertex_link_defect_count": len(topology.get("vertex_manifold_defect_ids", ())),
    }


def collapse_exact_duplicate_edge(
    faces_local_u32: np.ndarray,
    positions_world_f32: np.ndarray,
    canonical_material_gids_u64: np.ndarray,
    keep_vertex: int,
    retire_vertex: int,
    *,
    expected_incident_faces: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Collapse a byte-identical emitted edge after checking the simplicial link.

    The kept vertex's complete material owner remains authoritative. The retired
    vertex and its source/parent records remain allocated and unchanged; only
    face indices are rewritten, and the two incident degenerate faces are
    removed. This is a local inferred topology repair, not an anatomical gate.
    """
    faces = np.asarray(faces_local_u32)
    points = np.asarray(positions_world_f32)
    gids = np.asarray(canonical_material_gids_u64)
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be ordered UInt32 triangles")
    if points.dtype != np.dtype("<f4") or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("emitted points must be Float32 metre triples")
    if gids.dtype != np.dtype("<u8") or gids.shape != (len(points),):
        raise ValueError("canonical material GIDs must align with emitted points")
    keep, retire = int(keep_vertex), int(retire_vertex)
    if keep == retire or min(keep, retire) < 0 or max(keep, retire) >= len(points):
        raise ValueError("edge endpoints must be distinct valid vertices")
    if len(np.unique(gids)) != len(gids) or gids[keep] == gids[retire]:
        raise ValueError("edge endpoints require distinct unique canonical GIDs")
    if points[keep].tobytes() != points[retire].tobytes():
        raise ValueError("edge endpoints are not byte-identical emitted points")
    if int(faces.max(initial=0)) >= len(points):
        raise ValueError("face index exceeds emitted vertex count")
    if np.any((faces[:, 0] == faces[:, 1]) |
              (faces[:, 1] == faces[:, 2]) |
              (faces[:, 2] == faces[:, 0])):
        raise ValueError("input contains a topological degenerate face")

    keep_faces = np.any(faces == keep, axis=1)
    retire_faces = np.any(faces == retire, axis=1)
    edge_faces = np.flatnonzero(keep_faces & retire_faces)
    if len(edge_faces) != 2:
        raise ValueError(f"edge must have exactly two incident faces, got {len(edge_faces)}")
    if expected_incident_faces is not None and tuple(map(int, edge_faces)) != tuple(
        sorted(map(int, expected_incident_faces))
    ):
        raise ValueError("incident face indices differ from pinned local repair witness")

    keep_neighbors = set(map(int, faces[keep_faces].ravel())) - {keep}
    retire_neighbors = set(map(int, faces[retire_faces].ravel())) - {retire}
    common_link = keep_neighbors & retire_neighbors
    opposites = []
    for face_index in edge_faces:
        opposite = [int(v) for v in faces[int(face_index)] if int(v) not in (keep, retire)]
        if len(opposite) != 1:
            raise ValueError("edge incident triangle has no unique opposite vertex")
        opposites.append(opposite[0])
    if len(set(opposites)) != 2 or common_link != set(opposites):
        raise ValueError("edge fails the strict simplicial link condition")

    rewritten = faces.copy()
    rewritten[rewritten == retire] = np.uint32(keep)
    degenerate = np.flatnonzero(
        (rewritten[:, 0] == rewritten[:, 1]) |
        (rewritten[:, 1] == rewritten[:, 2]) |
        (rewritten[:, 2] == rewritten[:, 0])
    )
    if tuple(map(int, degenerate)) != tuple(map(int, edge_faces)):
        raise ValueError("collapse would remove faces beyond the two edge incident triangles")
    repaired = np.delete(rewritten, degenerate, axis=0)

    canonical_faces = [tuple(sorted(map(int, face))) for face in repaired]
    if len(set(canonical_faces)) != len(canonical_faces):
        raise ValueError("collapse would create duplicate unordered faces")
    edge_counts = Counter()
    for face in repaired:
        a, b, c = map(int, face)
        edge_counts[tuple(sorted((a, b)))] += 1
        edge_counts[tuple(sorted((b, c)))] += 1
        edge_counts[tuple(sorted((c, a)))] += 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    non_two_incidence = [edge for edge, count in edge_counts.items() if count != 2]
    if boundary or non_two_incidence:
        raise ValueError(
            f"collapse output fails closed edge incidence: boundary={len(boundary)}, "
            f"non_two_incidence={len(non_two_incidence)}"
        )
    return repaired.astype("<u4", copy=False), {
        "method": "strict_link_condition_collapse_of_byte_identical_emitted_edge",
        "kept_vertex_local_index": keep,
        "retired_vertex_local_index_still_allocated": retire,
        "kept_canonical_material_gid": int(gids[keep]),
        "retired_canonical_material_gid": int(gids[retire]),
        "endpoint_world_positions_byte_identical": True,
        "incident_face_indices_removed": list(map(int, edge_faces)),
        "opposite_vertices": sorted(opposites),
        "neighbor_link_intersection": sorted(common_link),
        "strict_link_condition_passed": True,
        "remaining_face_count": int(len(repaired)),
        "closed_edge_incidence_two_after_collapse": True,
        "vertex_link_orientation_and_full_manifold_gate_required": True,
        "no_vertex_or_material_arrays_are_modified_by_this_face_only_operation": True,
        "anatomical_admission": False,
    }



def collapse_short_edge_in_two_frames(
    faces_local_u32,
    positions_source_f32,
    positions_world_f32,
    keep_vertex: int,
    retire_vertex: int,
    *,
    modifiable_vertex_mask,
    maximum_edge_length_m: float,
    require_zero_face_reduction: bool = True,
) -> tuple[np.ndarray, dict]:
    """Infer a short-edge topology repair without changing any allocated source point.

    This construction requires a strict simplicial link and closed oriented
    output links. By default it requires a strict reduction in the combined
    source/world zero-face count; collision-driven callers may disable only
    that strict reduction, while retaining both per-frame nonincrease checks.
    Every previously nonzero surviving face must remain nonzero and retain its
    orientation in both frames. Only callers with authenticated source ancestry
    and authorization for local remeshing may mark endpoints modifiable. Retained
    source slots may participate: the inference concerns face topology, while
    every allocated source coordinate and identity stays intact. This never
    relabels a source point as a generated point. The complete input star is ancestry
    for rewritten faces, including the two retired incident faces.

    This does not check exact self intersections, neighboring anatomy, material
    ownership, or breathing poses, and cannot admit an anatomical candidate.
    """
    if type(require_zero_face_reduction) is not bool:
        raise ValueError("zero-face reduction policy must be boolean")
    faces = np.asarray(faces_local_u32)
    source = np.asarray(positions_source_f32)
    world = np.asarray(positions_world_f32)
    mask = np.asarray(modifiable_vertex_mask)
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be ordered UInt32 triangles")
    if (source.dtype != np.dtype("<f4") or source.ndim != 2 or source.shape[1] != 3
            or world.dtype != np.dtype("<f4") or world.shape != source.shape
            or not np.isfinite(source).all() or not np.isfinite(world).all()):
        raise ValueError("source and world positions must be aligned finite Float32 triples")
    if mask.dtype != np.dtype(np.bool_) or mask.shape != (len(source),):
        raise ValueError("modifiable mask must contain one boolean per allocated vertex")
    if type(keep_vertex) is not int or type(retire_vertex) is not int:
        raise ValueError("edge endpoints must be integer source slots")
    keep, retire = keep_vertex, retire_vertex
    if keep == retire or min(keep, retire) < 0 or max(keep, retire) >= len(source):
        raise ValueError("edge endpoints must be distinct valid source slots")
    if not mask[keep] or not mask[retire]:
        raise ValueError("both endpoints must be explicitly modifiable source slots")
    if (isinstance(maximum_edge_length_m, (bool, np.bool_))
            or not np.isfinite(maximum_edge_length_m) or maximum_edge_length_m <= 0):
        raise ValueError("maximum edge length must be finite and positive in metres")
    if not len(faces) or int(faces.max()) >= len(source):
        raise ValueError("empty faces or face index exceeds source allocation")
    # This also rejects duplicate/repeated faces and malformed input links. It
    # uses index topology because a construction input may have geometric zeros.
    before_topology = check_closed_oriented_vertex_links(faces, len(source))
    if not before_topology["closed_oriented_indexed_manifold"]:
        raise ValueError("input must have closed oriented indexed manifold links")

    lengths = tuple(float(np.linalg.norm(points[keep].astype(np.float64)
                                         - points[retire].astype(np.float64)))
                    for points in (source, world))
    if max(lengths) > maximum_edge_length_m:
        raise ValueError("edge exceeds declared source/world metre bound")
    keep_use = np.any(faces == keep, axis=1)
    retire_use = np.any(faces == retire, axis=1)
    incident = np.flatnonzero(keep_use & retire_use)
    if len(incident) != 2:
        raise ValueError("edge must have exactly two incident faces")
    left = set(map(int, faces[keep_use].ravel())) - {keep, retire}
    right = set(map(int, faces[retire_use].ravel())) - {keep, retire}
    opposites = set(map(int, faces[incident].ravel())) - {keep, retire}
    if len(opposites) != 2 or left & right != opposites:
        raise ValueError("edge fails the strict simplicial link condition")
    star = np.flatnonzero(keep_use | retire_use)
    rewritten = faces.copy()
    rewritten[rewritten == retire] = np.uint32(keep)
    live = ((rewritten[:, 0] != rewritten[:, 1])
            & (rewritten[:, 1] != rewritten[:, 2])
            & (rewritten[:, 2] != rewritten[:, 0]))
    if not np.array_equal(np.flatnonzero(~live), incident):
        raise ValueError("collapse retires faces beyond the incident pair")
    changed = np.flatnonzero(live & retire_use)
    candidate = rewritten[live]

    def vectors(points, triangles):
        xyz = points[triangles].astype(np.float64)
        return np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])

    before_zeros, after_zeros = [], []
    for label, points in (("source", source), ("world", world)):
        before = vectors(points, faces)
        after = vectors(points, candidate)
        old_star = before[star[live[star]]]
        new_star = vectors(points, rewritten[star[live[star]]])
        nonzero = np.any(old_star != 0, axis=1)
        if np.any(nonzero & np.all(new_star == 0, axis=1)):
            raise ValueError(label + " collapse creates a new surviving zero face")
        if np.any(nonzero & (np.einsum("ij,ij->i", old_star, new_star) <= 0)):
            raise ValueError(label + " collapse reverses a surviving face")
        before_zeros.append(int(np.count_nonzero(np.all(before == 0, axis=1))))
        after_zeros.append(int(np.count_nonzero(np.all(after == 0, axis=1))))
    if (any(new > old for old, new in zip(before_zeros, after_zeros))
            or (require_zero_face_reduction and sum(after_zeros) >= sum(before_zeros))):
        raise ValueError("combined zero-face inventory must strictly improve without frame regression")
    topology = check_closed_oriented_vertex_links(candidate, len(source))
    if not topology["closed_oriented_indexed_manifold"]:
        raise ValueError("collapse output fails closed oriented indexed manifold links")
    return np.ascontiguousarray(candidate, dtype="<u4"), {
        "method": "strict_link_bounded_inferred_edge_retirement_in_two_frames",
        "kept_vertex_local_index": keep,
        "retired_vertex_local_index_still_allocated": retire,
        "source_world_edge_lengths_m": list(lengths),
        "maximum_edge_length_m": float(maximum_edge_length_m),
        "incident_face_indices_removed": incident.astype(int).tolist(),
        "input_star_face_ids": star.astype(int).tolist(),
        "rewritten_input_face_ids": changed.astype(int).tolist(),
        "source_world_zero_face_counts_before": before_zeros,
        "source_world_zero_face_counts_after": after_zeros,
        "closed_oriented_indexed_topology": topology,
        "source_and_world_position_arrays_unchanged": True,
        "complete_input_star_required_as_rewritten_face_ancestry": True,
        "strict_zero_face_reduction_required": require_zero_face_reduction,
        "exact_self_and_neighbor_checks_required": True,
        "anatomical_admission": False,
        "native_admission": False,
    }

def _cyclic_equal(a, b):
    a = tuple(map(int, a))
    b = tuple(map(int, b))
    return b in (a, a[1:] + a[:1], a[2:] + a[:2])


def _opposite_cyclic(a, b):
    a = tuple(map(int, a))
    return _cyclic_equal((a[0], a[2], a[1]), b)


def check_closed_oriented_vertex_links(faces_local_u32, vertex_count: int) -> dict:
    """Check indexed edge incidence, directed winding, and one-cycle vertex links."""
    faces = np.asarray(faces_local_u32)
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be ordered UInt32 triangles")
    if type(vertex_count) is not int or vertex_count < 3:
        raise ValueError("vertex_count must be an integer >= 3")
    if not len(faces) or int(faces.max(initial=0)) >= vertex_count:
        raise ValueError("empty faces or out-of-range local index")
    edge_uses = defaultdict(list)
    vertex_links = defaultdict(list)
    face_keys = set()
    referenced = set()
    for fi, face in enumerate(faces):
        a, b, c = map(int, face)
        if len({a, b, c}) != 3:
            raise ValueError(f"degenerate indexed face {fi}")
        key = tuple(sorted((a, b, c)))
        if key in face_keys:
            raise ValueError(f"duplicate indexed face {fi}")
        face_keys.add(key)
        referenced.update((a, b, c))
        for x, y in ((a, b), (b, c), (c, a)):
            edge_uses[tuple(sorted((x, y)))].append((x, y, fi))
        vertex_links[a].append((b, c))
        vertex_links[b].append((c, a))
        vertex_links[c].append((a, b))

    boundary_edges = [edge for edge, uses in edge_uses.items() if len(uses) == 1]
    non_two_edges = [edge for edge, uses in edge_uses.items() if len(uses) != 2]
    orientation_defects = [
        edge for edge, uses in edge_uses.items()
        if len(uses) == 2 and (uses[0][0] != uses[1][1] or uses[0][1] != uses[1][0])
    ]
    link_defects = []
    for vertex, opposite_edges in vertex_links.items():
        adjacency = defaultdict(set)
        link_edge_counts = defaultdict(int)
        for a, b in opposite_edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
            link_edge_counts[tuple(sorted((a, b)))] += 1
        if (not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values())
                or any(count != 1 for count in link_edge_counts.values())):
            link_defects.append({"vertex": vertex, "reason": "link degree/repeated edge"})
            continue
        reached = set()
        pending = [next(iter(adjacency))]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(adjacency[current] - reached)
        if len(reached) != len(adjacency):
            link_defects.append({"vertex": vertex, "reason": "multiple link cycles"})

    report = {
        "face_count": int(len(faces)),
        "edge_count": int(len(edge_uses)),
        "boundary_edge_count": int(len(boundary_edges)),
        "non_two_incidence_edge_count": int(len(non_two_edges)),
        "orientation_defect_edge_count": int(len(orientation_defects)),
        "vertex_link_defect_count": int(len(link_defects)),
        "unused_vertex_count": int(vertex_count - len(referenced)),
        "edge_incidence_two": not non_two_edges,
        "consistent_directed_edge_orientation": not orientation_defects,
        "single_cycle_vertex_links": not link_defects,
        "closed_oriented_indexed_manifold": (
            not non_two_edges and not orientation_defects and not link_defects
        ),
        "witnesses": {
            "boundary_edges": [list(map(int, x)) for x in boundary_edges[:12]],
            "non_two_incidence_edges": [list(map(int, x)) for x in non_two_edges[:12]],
            "orientation_defect_edges": [list(map(int, x)) for x in orientation_defects[:12]],
            "vertex_link_defects": link_defects[:12],
        },
    }
    if not report["closed_oriented_indexed_manifold"]:
        raise ValueError("rewritten indexed surface is not a closed oriented 2-manifold")
    return report


def _diagnose_indexed_topology(faces_local_u32, vertex_count: int) -> dict:
    faces = np.asarray(faces_local_u32)
    edge_uses = defaultdict(list)
    links = defaultdict(list)
    keys = set()
    duplicate_count = 0
    degenerate_count = 0
    referenced = set()
    for fi, face in enumerate(faces):
        a, b, c = map(int, face)
        if len({a, b, c}) != 3:
            degenerate_count += 1
            continue
        key = tuple(sorted((a, b, c)))
        if key in keys:
            duplicate_count += 1
        keys.add(key)
        referenced.update((a, b, c))
        for x, y in ((a, b), (b, c), (c, a)):
            edge_uses[tuple(sorted((x, y)))].append((x, y, fi))
        links[a].append((b, c)); links[b].append((c, a)); links[c].append((a, b))
    non_two = [uses for uses in edge_uses.values() if len(uses) != 2]
    orient = [uses for uses in edge_uses.values() if len(uses) == 2
              and (uses[0][0] != uses[1][1] or uses[0][1] != uses[1][0])]
    link_count = 0
    for opposite in links.values():
        adjacency = defaultdict(set); counts = defaultdict(int)
        for a, b in opposite:
            adjacency[a].add(b); adjacency[b].add(a); counts[tuple(sorted((a, b)))] += 1
        if (not adjacency or any(len(v) != 2 for v in adjacency.values())
                or any(v != 1 for v in counts.values())):
            link_count += 1
            continue
        reached = set(); pending = [next(iter(adjacency))]
        while pending:
            x = pending.pop()
            if x in reached: continue
            reached.add(x); pending.extend(adjacency[x] - reached)
        if len(reached) != len(adjacency): link_count += 1
    return {
        "face_count": int(len(faces)), "edge_count": int(len(edge_uses)),
        "non_two_incidence_edge_count": int(len(non_two)),
        "orientation_defect_edge_count": int(len(orient)),
        "vertex_link_defect_count": int(link_count),
        "duplicate_face_count": int(duplicate_count),
        "degenerate_face_count": int(degenerate_count),
        "unused_vertex_count": int(vertex_count-len(referenced)),
        "closed_oriented_indexed_manifold": False,
        "witnesses": {"non_two_incidence_edge_count": int(len(non_two)),
                      "orientation_defect_edge_count": int(len(orient)),
                      "vertex_link_defect_count": int(link_count)},
    }


def repair_duplicate_interface_birth_faces(
    faces_local_u32: np.ndarray,
    positions_world_f32: np.ndarray,
    canonical_material_gids_u64: np.ndarray,
    duplicate_pairs: list[dict],
    *,
    require_closed_oriented: bool = True,
) -> tuple[np.ndarray, dict]:
    """Rewire exact duplicate birth vertices and cancel their opposite face pairs.

    Each pair mapping has only keep_gid, retire_gid and face_indices. Both source
    vertices stay allocated. No position, GID, source slot, or full21 material
    array is accepted as mutable input.
    """
    faces = np.asarray(faces_local_u32)
    points = np.asarray(positions_world_f32)
    gids = np.asarray(canonical_material_gids_u64)
    if faces.dtype != np.dtype("<u4") or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be ordered UInt32 triangles")
    if points.dtype != np.dtype("<f4") or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must be emitted Float32 metre triples")
    if gids.dtype != np.dtype("<u8") or gids.shape != (len(points),):
        raise ValueError("canonical material GIDs must align with positions")
    if len(np.unique(gids)) != len(gids) or not np.isfinite(points).all():
        raise ValueError("material GIDs must be unique and positions finite")
    if not len(faces) or int(faces.max(initial=0)) >= len(points):
        raise ValueError("empty faces or out-of-range local index")
    if not isinstance(duplicate_pairs, list) or not duplicate_pairs:
        raise ValueError("at least one exact duplicate interface pair is required")

    gid_to_local = {int(gid): i for i, gid in enumerate(gids)}
    substitutions = {}
    specs = []
    removed = set()
    for spec in duplicate_pairs:
        if not isinstance(spec, dict) or set(spec) != {
                "keep_gid", "retire_gid", "face_indices"}:
            raise ValueError("pair requires keep_gid, retire_gid, and face_indices")
        keep_gid, retire_gid = int(spec["keep_gid"]), int(spec["retire_gid"])
        keep = gid_to_local.get(keep_gid)
        retire = gid_to_local.get(retire_gid)
        pair_faces = tuple(map(int, spec["face_indices"]))
        if keep is None or retire is None or keep == retire:
            raise ValueError("representative and retired GIDs must resolve to distinct local vertices")
        if keep in substitutions or retire in substitutions or keep_gid == retire_gid:
            raise ValueError("a source vertex may belong to only one substitution")
        if points[keep].tobytes() != points[retire].tobytes():
            raise ValueError("duplicate birth positions are not byte-identical Float32")
        if len(pair_faces) != 2 or pair_faces[0] == pair_faces[1]:
            raise ValueError("each duplicate interface requires two distinct face ordinals")
        if min(pair_faces) < 0 or max(pair_faces) >= len(faces):
            raise ValueError("duplicate interface face ordinal is out of range")
        if removed.intersection(pair_faces):
            raise ValueError("duplicate interface face pairs overlap")
        removed.update(pair_faces)
        substitutions[retire] = keep
        specs.append({
            "keep_gid": keep_gid, "retire_gid": retire_gid,
            "keep_local": keep, "retire_local": retire,
            "face_indices": pair_faces,
        })

    rewritten = faces.copy()
    for retire, keep in substitutions.items():
        rewritten[rewritten == np.uint32(retire)] = np.uint32(keep)

    for spec in specs:
        first, second = spec["face_indices"]
        a, b = rewritten[first], rewritten[second]
        if len(set(map(int, a))) != 3 or len(set(map(int, b))) != 3:
            raise ValueError("pinned duplicate face pair became degenerate before cancellation")
        if tuple(sorted(map(int, a))) != tuple(sorted(map(int, b))):
            raise ValueError("pinned faces do not become the same triangle after representative substitution")
        if not _opposite_cyclic(a, b):
            raise ValueError("pinned duplicate faces are not oppositely wound")

    degenerate = np.flatnonzero(
        (rewritten[:, 0] == rewritten[:, 1]) |
        (rewritten[:, 1] == rewritten[:, 2]) |
        (rewritten[:, 2] == rewritten[:, 0])
    )
    if len(degenerate):
        raise ValueError("representative substitution creates an unpinned degenerate face")

    keep_mask = np.ones(len(faces), dtype=bool)
    keep_mask[np.asarray(sorted(removed), dtype=np.int64)] = False
    result_faces = rewritten[keep_mask].astype("<u4", copy=False)
    try:
        topology = check_closed_oriented_vertex_links(result_faces, len(points))
        topology_error = None
    except ValueError as exc:
        if require_closed_oriented:
            raise
        # Keep a failed construction for diagnosis; it is never eligible for gates.
        topology = _diagnose_indexed_topology(result_faces, len(points))
        topology_error = str(exc)

    old_to_new = np.full(len(faces), -1, dtype="<i8")
    old_to_new[np.flatnonzero(keep_mask)] = np.arange(len(result_faces), dtype="<i8")
    changed = np.flatnonzero(np.any(rewritten != faces, axis=1))
    return result_faces, {
        "method": "authenticated_representative_rewire_and_opposite_interface_face_cancellation",
        "representative_substitutions": [
            {"keep_gid": s["keep_gid"], "retire_gid": s["retire_gid"],
             "keep_local": s["keep_local"], "retire_local_still_allocated": s["retire_local"],
             "duplicate_face_indices_removed": list(s["face_indices"])}
            for s in specs
        ],
        "all_source_vertex_arrays_unchanged": True,
        "both_lineage_records_retained": True,
        "rewritten_face_indices_before_removal": changed.astype("<u4").tolist(),
        "removed_face_indices": sorted(removed),
        "old_to_new_face_ordinal_map": old_to_new,
        "rewritten_faces_before_removal": rewritten,
        "old_face_count": int(len(faces)),
        "new_face_count": int(len(result_faces)),
        "indexed_topology": topology,
        "indexed_topology_error": topology_error,
        "anatomical_admission": False,
        "native_admission": False,
    }
