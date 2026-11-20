"""Finite-triangle guidance for offline repair against closed skull banks."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack

from audit_source import owner, records

MM_PER_M = 1000.
CLEARANCE_M = 0.035e-3
ROLE_WEIGHT = {"outer": 1.5, "interface": 2., "deep": 5., "side": 2.}


def unit(vectors):
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, length, out=np.zeros_like(vectors), where=length > 0.)


def finite_outward_planes(a, b, motion, bound_m):
    """Use triangle SAT axes to select an outward, attainable fixed plane."""
    a_edges, b_edges = (unit(np.roll(x, -1, axis=1) - x) for x in (a, b))
    a_normal, b_normal = (unit(np.cross(edges[:, 0], -edges[:, 2]))
                          for edges in (a_edges, b_edges))
    edge_cross = np.cross(a_edges[:, :, None], b_edges[:, None]).reshape(-1, 9, 3)
    axes = np.concatenate((a_normal[:, None], b_normal[:, None], unit(edge_cross),
                           unit(np.cross(a_normal[:, None], a_edges)),
                           unit(np.cross(b_normal[:, None], b_edges))), axis=1)
    alignment = np.einsum("pnd,pd->pn", axes, b_normal)
    axes *= np.where(alignment >= 0., 1., -1.)[:, :, None]
    center = .5 * (a[:, 0] + b[:, 0])
    a_dot = np.einsum("pnd,pvd->pnv", axes, a - center[:, None])
    b_dot = np.einsum("pnd,pvd->pnv", axes, b - center[:, None])
    gaps = a_dot - b_dot.max(axis=-1)[:, :, None]
    motion_dot = np.einsum("pnd,pvd->pnv", axes, motion)
    attainable = np.min(gaps + bound_m * np.abs(motion_dot), axis=-1)
    score = attainable + .05 * np.min(gaps, axis=-1)
    score[np.abs(alignment) < .15] = -np.inf
    assert np.isfinite(score.max(axis=1)).all()
    choice = score.argmax(axis=1)
    pair = np.arange(len(choice))
    return gaps[pair, choice], motion_dot[pair, choice], axes[pair, choice]


@dataclass
class Snapshot:
    face_rows: np.ndarray
    triangle_rows: np.ndarray
    counts: dict


class SkullContact:
    def __init__(self, faces, parent_faces, roles, triangle_rows):
        self.faces = faces
        self.parent_faces = np.asarray(parent_faces)
        self.roles = np.asarray(roles)
        self.bone_ids = np.stack((triangle_rows["bone"], triangle_rows["triangle"]), axis=1)
        assert len(np.unique(self.bone_ids, axis=0)) == len(triangle_rows)
        self.bone = triangle_rows["vertices"].astype(np.float64)
        # Materialize the bank for repeated line-search queries.
        self.bone_surface = tuple(records(triangle_rows["vertices"]))
        self.role_weight = np.array([ROLE_WEIGHT[role] for role in roles])

    def measure(self, points):
        surface, _ = owner.float32_surface_records(points, self.faces)
        tree = owner._prepare_surface_aabb(surface)
        result = owner._audit_pair_prepared_first(tree, self.bone_surface)
        index = np.asarray(result["triangle_pairs"], dtype=np.intp).reshape(-1, 2)
        faces, bones = index.T
        keys = np.stack((self.parent_faces[faces], bones), axis=1)
        _, distinct = np.unique(keys, axis=0, return_index=True)
        roles = self.roles[faces[distinct]]
        counts = {role: int(np.count_nonzero(roles == role)) for role in ROLE_WEIGHT}
        return Snapshot(faces, bones, counts)

    def inequalities(self, points, snapshot, column_for_node, direction, radius, count):
        indices = self.faces[snapshot.face_rows]
        columns = column_for_node[indices]
        participates = np.any(columns >= 0, axis=1)
        frozen = int(np.count_nonzero(~participates))
        indices = indices[participates]
        columns = columns[participates]
        faces = snapshot.face_rows[participates]
        bone = snapshot.triangle_rows[participates]
        gaps, gradients, _ = finite_outward_planes(
            points[indices], self.bone[bone], direction[indices], radius)
        active = columns >= 0
        pair_count = len(columns)
        rows = np.arange(3 * pair_count).reshape(-1, 3)
        active_system = coo_matrix((-gradients[active], (rows[active], columns[active])),
                                   shape=(3 * pair_count, count)).tocsr()
        slack_rows = coo_matrix((-np.ones(3 * pair_count),
                                 (rows.ravel(), np.repeat(np.arange(pair_count), 3))),
                                shape=(3 * pair_count, pair_count)).tocsr()
        zeros = csr_matrix(active_system.shape)
        system = hstack((active_system, zeros, slack_rows), format="csr")
        # Normalize the imbalance created by local triangle tessellation.
        frequency = np.bincount(faces, minlength=len(self.faces))
        cost = self.role_weight[faces] / np.sqrt(frequency[faces])
        bound = MM_PER_M * (gaps.ravel() - CLEARANCE_M)
        return system, bound, cost, {"active_pairs": pair_count, "frozen_pairs": frozen}


def admissible(previous, new):
    """Forbid deep regressions and require improvement in total pairs."""
    p, n = previous.counts, new.counts
    return n["deep"] <= p["deep"] and sum(n.values()) < sum(p.values())
