"""Conforming local refinement of the existing NHSKIN visual shell.

This is offline inferred geometry preparation. Original source slots, ownership
and source hashes remain intact; appended points carry edge ancestry. No mass,
contact point, muscle or per-step state is created.
"""

from __future__ import annotations

import hashlib
import struct
from fractions import Fraction

import numpy as np

from . import model as human
from .skin_source_payload_preflight import decode_payload


def refine_skin_patch(raw: bytes, face_ids, *, global_source_matrix):
    """Split selected face edges and conform every incident neighbor.

    Red/green subdivision adds only midpoint births. The returned proof records
    parent face ordinals and exact dyadic barycentrics, including unmodified
    faces. Full dense weights use the same convex interpolation; four diagnostic
    fields never replace those weights. Clearance/native admission remain false.
    """
    from .common_atlas_skin_geometry_registration import (
        compose_disjoint_skin_position_corrections,
    )

    base = decode_payload(raw)
    faces = base["indices"].reshape(-1, 3).astype(np.int64)
    selected = np.asarray(face_ids)
    if (
        selected.ndim != 1
        or not len(selected)
        or selected.dtype.kind not in "iu"
        or selected.min() < 0
        or selected.max() >= len(faces)
        or len(np.unique(selected)) != len(selected)
    ):
        raise human.ImportError("skin refinement requires unique valid face ordinals")
    xyz = base["vertices_f"][:, :3]
    dense = base["full_weights"]
    used = np.unique(faces)
    if (
        not np.isfinite(base["vertices_f"][used, :6]).all()
        or not np.isfinite(dense).all()
        or np.any(dense < 0)
        or np.max(np.abs(dense.sum(axis=1, dtype=np.float64) - 1)) > 4e-7
    ):
        raise human.ImportError("skin refinement input geometry/weights invalid")
    edges = sorted(
        {
            tuple(sorted((int(a), int(b))))
            for face in faces[selected]
            for a, b in zip(face, np.roll(face, -1))
        }
    )
    nv = int(base["vertex_count"])
    edge_id = {edge: nv + i for i, edge in enumerate(edges)}
    records = np.zeros((len(edges), 14), dtype="<u4")
    floats = records.view("<f4")
    weights = np.empty((len(edges), int(base["binding_count"])), dtype="<f4")
    births = []
    for i, (a, b) in enumerate(edges):
        floats[i, :3] = ((xyz[a].astype(float) + xyz[b].astype(float)) * 0.5).astype(
            "<f4"
        )
        weights[i] = ((dense[a].astype(float) + dense[b].astype(float)) * 0.5).astype(
            "<f4"
        )
        if not np.array_equal(weights[i] > 0, (dense[a] > 0) | (dense[b] > 0)):
            raise human.ImportError("skin midpoint loses positive owner support")
        order = np.argsort(-weights[i].astype(float), kind="stable")[:4]
        records[i, 6:10] = np.pad(order, (0, 4 - len(order)))
        diagnostic = weights[i, order].astype(float)
        diagnostic /= diagnostic.sum()
        floats[i, 10:14] = np.pad(diagnostic, (0, 4 - len(order)))
        births.append(
            {
                "vertex_id": nv + i,
                "parent_edge_vertex_ids": [a, b],
                "parent_edge_weights": ["1/2", "1/2"],
            }
        )
    out_faces, parents, barycentrics, coverage = [], [], [], []
    for parent, face in enumerate(faces):
        a, b, c = map(int, face)
        mids = [edge_id.get(tuple(sorted(edge))) for edge in [(a, b), (b, c), (c, a)]]
        count = sum(x is not None for x in mids)
        if count == 0:
            children = [(a, b, c)]
        elif count == 1:
            i = next(i for i, x in enumerate(mids) if x is not None)
            u, v, z = [a, b, c][i:] + [a, b, c][:i]
            m = mids[i]
            children = [(u, m, z), (m, v, z)]
        elif count == 2:
            # Rotate so both split edges are u-v and v-z.
            i = next(
                i
                for i in range(3)
                if mids[i] is not None and mids[(i + 1) % 3] is not None
            )
            u, v, z = [a, b, c][i:] + [a, b, c][:i]
            m, n = mids[i], mids[(i + 1) % 3]
            children = [(u, m, z), (m, n, z), (m, v, n)]
        else:
            m, n, p = mids
            children = [(a, m, p), (m, b, n), (p, n, c), (m, n, p)]
        coords = {
            a: (Fraction(1), Fraction(0), Fraction(0)),
            b: (Fraction(0), Fraction(1), Fraction(0)),
            c: (Fraction(0), Fraction(0), Fraction(1)),
        }
        for edge, m in zip([(a, b), (b, c), (c, a)], mids):
            if m is not None:
                coords[m] = tuple(
                    (x + y) / 2 for x, y in zip(coords[edge[0]], coords[edge[1]])
                )
        total = Fraction(0)
        for child in children:
            bc = [coords[v] for v in child]
            u = [bc[1][j] - bc[0][j] for j in [1, 2]]
            v = [bc[2][j] - bc[0][j] for j in [1, 2]]
            area = u[0] * v[1] - u[1] * v[0]
            if area <= 0:
                raise human.ImportError(
                    "skin refinement reversed/collapsed parent patch"
                )
            total += area
            out_faces.append(child)
            parents.append(parent)
            barycentrics.append([[str(x) for x in row] for row in bc])
        if total != 1:
            raise human.ImportError("skin refinement does not cover its parent face")
        coverage.append(str(total))
    new_faces = np.asarray(out_faces, dtype="<u4")
    binding_end = 60 + 36 * int(base["binding_count"])
    header = bytearray(raw[:60])
    struct.pack_into("<II", header, 16, nv + len(edges), int(new_faces.size))
    vertices = np.concatenate([base["vertices_u"], records])
    full = np.concatenate([dense, weights]).astype("<f4")
    intermediate = (
        bytes(header)
        + raw[60:binding_end]
        + vertices.tobytes()
        + new_faces.tobytes()
        + full.tobytes()
    )
    # Reuse the existing rest-world normal owner; this never adds a force owner.
    normal_raw, normal_proof = compose_disjoint_skin_position_corrections(
        intermediate, [intermediate], global_source_matrix=global_source_matrix
    )
    final = bytearray(normal_raw)
    # Keep every original record exact outside the affected normal star.
    affected = np.unique(faces[selected])
    for _ in range(3):
        affected = np.unique(faces[np.any(np.isin(faces, affected), axis=1)])
    restore = np.setdiff1d(np.arange(nv), affected)
    for i in restore:
        offset = binding_end + int(i) * 56
        final[offset : offset + 56] = raw[offset : offset + 56]
    after = decode_payload(bytes(final))
    if (
        not np.array_equal(after["vertices_u"][:nv, :3], base["vertices_u"][:, :3])
        or not np.array_equal(after["vertices_u"][:nv, 6:], base["vertices_u"][:, 6:])
        or not np.array_equal(after["full_weights"][:nv], dense)
        or bytes(final)[60:binding_end] != raw[60:binding_end]
    ):
        raise human.ImportError("skin refinement changed original positions/owners")
    proof = {
        "base_payload_sha256": hashlib.sha256(raw).hexdigest(),
        "output_payload_sha256": hashlib.sha256(final).hexdigest(),
        "method": "conforming red/green midpoint refinement of selected source faces",
        "selected_source_face_ids": selected.astype(int).tolist(),
        "appended_vertex_births": births,
        "output_face_parent_ids": parents,
        "output_face_parent_barycentrics": barycentrics,
        "exact_parent_oriented_area_ratios": coverage,
        "original_source_vertex_count": nv,
        "original_source_positions_and_all_weight_rows_exact": True,
        "source_archive_identity_and_bindings_exact": True,
        "rest_world_normal_owner": normal_proof,
        "inferred_vertices_are_source_measurements": False,
        "physical_mass_contact_and_physiology_owners_unchanged": True,
        "anatomical_admission": False,
        "native_admission": False,
    }
    return bytes(final), proof
