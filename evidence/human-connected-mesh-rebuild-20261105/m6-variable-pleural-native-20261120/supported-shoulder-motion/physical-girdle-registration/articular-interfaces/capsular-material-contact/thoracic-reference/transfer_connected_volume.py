"""Screen a connected shoulder registration on the actual skin and fat tetrahedra."""

import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32
from numilab_human.capsular_material_registration import (
    emitted_body_vertices,
    material_bank,
    read_material_banks,
)
from register_connected_arm import descendants
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from screen_material_transport import inputs as layer_inputs
from shoulder_inputs import digest


def geometry(x, tets):
    volume = np.empty(len(tets))
    quality = np.empty(len(tets))
    for i in range(0, len(tets), 48000):
        y = x[tets[i : i + 48000]].astype(np.float64)
        a, b, c = np.moveaxis(y[:, 1:] - y[:, :1], 1, 0)
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        max_edge_sq = np.maximum.reduce(
            [
                np.einsum("ij,ij->i", y[:, j] - y[:, k], y[:, j] - y[:, k])
                for j, k in combinations(range(4), 2)
            ]
        )
        volume[i : i + len(y)] = det / 6
        quality[i : i + len(y)] = det / max_edge_sq**1.5
    return volume, quality


def blend_connected_owners(x, tets, labels, band_m):
    """Blend only across shared tetrahedral edges of the physical material."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    # A unique undirected edge has one geometric length irrespective of how
    # many tetrahedra meet at it. Spatially adjacent limbs never add edges.
    pairs = np.empty((6, len(tets)), dtype=np.uint64)
    for i, (j, k) in enumerate(combinations(range(4), 2)):
        a, b = tets[:, j], tets[:, k]
        pairs[i] = (np.minimum(a, b).astype(np.uint64) << 32) | np.maximum(a, b).astype(
            np.uint64
        )
    keys = np.unique(pairs)
    del pairs
    a = (keys >> 32).astype(np.int32)
    b = (keys & np.uint64(0xFFFFFFFF)).astype(np.int32)
    length = np.linalg.norm(x[a] - x[b], axis=1)
    graph = coo_matrix(
        (np.r_[length, length], (np.r_[a, b], np.r_[b, a])), shape=(len(x), len(x))
    ).tocsr()
    weights = []
    for group in range(5):
        own = labels == group
        border = own[a] != own[b]
        seeds = np.unique(np.r_[a[border], b[border]])
        if not len(seeds):
            weights.append(own.astype(float))
            continue
        dist = dijkstra(
            graph, directed=True, indices=seeds, min_only=True, limit=band_m
        )
        signed = np.minimum(dist, band_m) * np.where(own, 1.0, -1.0)
        u = 0.5 + 0.5 * signed / band_m
        weights.append(u * u * u * (10 + u * (-15 + 6 * u)))
    weights = np.asarray(weights)
    return weights / weights.sum(0)


def run(scene, loader, receipt, bones_path, head, arms, output):
    _state, banks, _, points, pins = layer_inputs(scene, loader, receipt, bones_path)
    cfg = json.loads((scene / "launch.json").read_text())
    env = cfg["environment"]
    base = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    tet_file = Path(env[base + "TETGEN_ELEMENTS"])
    node_map = Path(env[base + "COMBINED_NODE_MAP_CSV"])
    if digest(node_map) != env[base + "COMBINED_NODE_MAP_SHA256"]:
        raise ValueError("Volume donor map changed")
    tets = np.loadtxt(tet_file, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
    tags = tets[:, 4].copy()
    tets = tets[:, :4] - 1
    x = points.astype(float)
    with node_map.open() as f:
        header = next(csv.reader(f))
    cols = [
        header.index(k)
        for k in (
            "joint_node_id",
            "current_x_m",
            "current_y_m",
            "current_z_m",
            "donor_body",
        )
    ]
    donors = np.loadtxt(node_map, skiprows=1, usecols=cols, delimiter=",")
    # The paired partitions repeat their shared interface nodes. One physical
    # node carries one binding and must contribute volume only once.
    ids = donors[:, 0].astype(np.int64) - 1
    if (
        np.any(ids < 0)
        or np.any(ids >= len(x))
        or not np.array_equal(donors[:, 1:4].astype("f4"), points[ids])
    ):
        raise ValueError("Partition and physical node coordinates differ")
    unique, first = np.unique(ids, return_index=True)
    if not np.array_equal(unique, np.arange(len(x))) or np.any(
        donors[:, 4] != donors[first, 4][ids]
    ):
        raise ValueError("Shared skin/fat nodes have contradictory physical owners")
    donors = donors[first]
    if not np.array_equal(donors[:, 0], np.arange(1, len(x) + 1)) or not np.array_equal(
        donors[:, 1:4].astype("f4"), points
    ):
        raise ValueError(
            "The current node identities or coordinates differ from the retained physical owner"
        )
    labels = np.zeros(len(x), dtype="u1")
    data = read_material_banks(bones_path)
    records = data[2]
    arm = json.loads(arms.read_text())
    for i, g in enumerate(arm["moved"], 1):
        side, group = g["group"].split("_")
        sc, hu = (11, 13) if side == "l" else (10, 12)
        rigid = Path(cfg["argv"][1])
        bodies = (
            descendants(rigid, records[hu][0]) if group == "arm" else {records[sc][0]}
        )
        labels[np.isin(donors[:, 4], list(bodies))] = i
    del donors
    original_v, original_q = geometry(points, tets)
    if np.any(original_v <= 1e-18) or np.any(original_q <= 1e-5):
        raise ValueError("The referenced volume fails its existing admission limits")
    hrows = json.loads(head.read_text())["rows"]
    sigma = np.mean([r["variants"][0]["radius_m"] for r in hrows])
    # Two articular radii span the shoulder diameter measured in this adult.
    band = 2 * sigma
    weight = blend_connected_owners(x, tets, labels, band)
    delta = np.zeros_like(x)
    body_changes = {}
    for i, g in enumerate(arm["moved"], 1):
        r = Rotation.from_quat(g["rotation_xyzw"])
        center = np.array(g["pivot_before_m"])
        target = np.array(g["pivot_after_m"])
        delta += weight[i, :, None] * (target + r.apply(x - center) - x)
        for sid in g["banks"]:
            body_changes[sid] = (r, center, target)
    new = (x + delta).astype("f4")
    v, q = geometry(new, tets)
    tinfo = []
    for tag in sorted(set(tags)):
        b = tags == tag
        failed = b & ((v <= 1e-18) | (q <= 1e-5))
        ids = np.flatnonzero(failed)
        # The reported span and quality together reconstruct the maximum
        # edge length of each tetrahedron without a second geometry traversal.
        span = np.cbrt(6.0 * original_v[b] / original_q[b])
        flipped = v[b] <= 0
        failed_tets = tets[b][flipped]
        owners_per_cell = labels[failed_tets]
        pure_source_owner = np.all(owners_per_cell == owners_per_cell[:, :1], axis=1)
        failure_span = span[flipped]
        tinfo.append(
            {
                "material": int(tag),
                "cells": int(b.sum()),
                "inversions": int((v[b] <= 0).sum()),
                "low_quality": int((q[b] <= 1e-5).sum()),
                "low_positive_volume": int(((v[b] > 0) & (v[b] <= 1e-18)).sum()),
                "minimum_signed_volume_m3": float(v[b].min()),
                "source_volume_m3": float(original_v[b].sum()),
                "candidate_volume_m3": float(v[b].sum()),
                "source_cells_over_100mm": int((span > 0.1).sum()),
                "maximum_source_edge_m": float(span.max()),
                "inversion_span_quantiles_m": np.quantile(
                    failure_span, [0.0, 0.5, 0.9, 1.0]
                ).tolist()
                if len(failure_span)
                else [],
                "inversions_on_cells_with_one_donor_group": int(
                    pure_source_owner.sum()
                ),
                "inversions_on_cells_over_100mm": int(
                    np.count_nonzero(failure_span > 0.1)
                ),
                "first_failed_rows": ids[:12].tolist(),
            }
        )
    tree = cKDTree(x)
    surfaces = {}
    for layer in (1, 0xFFFFFD11, 0xFFFFFD12):
        p, f = banks[51007, layer]
        dist, idx = tree.query(p)
        if dist.max() != 0:
            raise ValueError("A registered layer differs from its mechanical nodes")
        surfaces[layer] = (new[idx], f)
    poses = {
        r["body_index"]: np.array(r["position_m"] + r["quaternion_xyzw"])
        for r in json.loads(receipt.read_text())["accepted_body_poses"]
    }
    install_for_float32()
    crossings = []
    for sid, (r, center, target) in sorted(body_changes.items()):
        row = records[sid]
        _, _, attrs, f = material_bank(data, sid, row[0])
        p = poses[row[0]]
        verts = p[:3] + Rotation.from_quat(p[3:]).apply(
            emitted_body_vertices(attrs, row)
        )
        moved = target + r.apply(verts - center)
        for layer, (verts, faces) in surfaces.items():
            pairs = intersections(moved, f, verts, faces)
            if pairs:
                crossings.append([sid, layer, len(pairs), pairs[0]])
    output.write_text(
        json.dumps(
            {
                "inputs": pins,
                "donors_sha256": digest(node_map),
                "registered_arm_sha256": digest(arms),
                "nodes": len(x),
                "donor_group_node_counts": [int((labels == v).sum()) for v in range(5)],
                "geodesic_transition_halfwidth_m": float(band),
                "candidate_float32_nodes_sha256": hashlib.sha256(
                    new.tobytes()
                ).hexdigest(),
                "largest_displacement_m": float(np.linalg.norm(new - x, axis=1).max()),
                "materials": tinfo,
                "tested_layer_crossings": crossings,
                "entire_skin_min_z_m": float(surfaces[1][0][:, 2].min()),
                "maximum_partition_unity_error": float(
                    np.max(np.abs(weight.sum(0) - 1))
                ),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*map(Path, sys.argv[1:]))
