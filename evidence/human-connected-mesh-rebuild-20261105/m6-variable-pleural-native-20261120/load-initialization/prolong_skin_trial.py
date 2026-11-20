"""Prolong a same-source coarse equilibrium trial into the fine skin/fat initial geometry.

The output is a TetGen node file for a trial initialization. It leaves material
reference coordinates, bed geometry, nodal force ownership, and clocks intact.
"""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np

from run_recruitment import CONTROL, digest

TAG = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
REGISTRATION_FIELDS = ("joint_node_id", "current_x_m", "current_y_m", "current_z_m",
                       "parent_tet_row", "parent_bary0", "parent_bary1", "parent_bary2",
                       "parent_bary3", "parent_current_residual_x_m",
                       "parent_current_residual_y_m", "parent_current_residual_z_m")


def columns(path, names):
    with path.open() as f:
        header = f.readline().strip().split(",")
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=[header.index(k) for k in names])


def parent_nodes(path):
    data = columns(path, ("parent_node_index", "compiled_global_node", "current_x_m",
                         "current_y_m", "current_z_m", "compiled_lumped_mass_kg"))
    assert np.array_equal(data[:, 0], np.arange(len(data)))
    assert len(np.unique(data[:, 1])) == len(data)
    assert np.isfinite(data).all() and (data[:, -1] > 0.).all()
    return data


def require_digest(p, expected):
    assert digest(p) == expected, p
    return p


def determinant(points, tets, displacement=None):
    result = np.empty(len(tets))
    for start in range(0, len(tets), 32768):
        part = tets[start:start+32768]
        tet = points[part].astype(np.float64)
        edges = tet[:, 1:] - tet[:, :1]
        if displacement is not None:
            movement = displacement[part]
            edges += movement[:, 1:] - movement[:, :1]
        a, b, c = np.moveaxis(edges, 1, 0)
        result[start:start+len(part)] = np.einsum("ij,ij->i", a, np.cross(b, c))
    return result


def run(current_parent, current_sha, output, publish_nodes, convect_offsets=False):
    launch = json.loads((CONTROL / "launch.json").read_text())
    env = launch["environment"]
    tests = {
        "parent": ("NUMI_H706_FRACTIONAL_SOURCE330_NODES",
                   "NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256"),
        "parent_tets": ("NUMI_H706_FRACTIONAL_SOURCE330_TETS",
                        "NUMI_H706_FRACTIONAL_SOURCE330_TETS_SHA256"),
        "child": (TAG+"TETGEN_NODE", TAG+"TETGEN_NODE_SHA256"),
        "child_tets": (TAG+"TETGEN_ELEMENTS", TAG+"TETGEN_ELEMENTS_SHA256"),
        "mapping": (TAG+"COMBINED_NODE_MAP_CSV", TAG+"COMBINED_NODE_MAP_SHA256"),
        "outer": (TAG+"SURFACE_O_NODES_CSV", TAG+"SURFACE_O_NODES_SHA256"),
    }
    asset = {name: require_digest(Path(env[path]), env[sha])
             for name, (path, sha) in tests.items()}
    measured = parent_nodes(require_digest(current_parent, current_sha))
    parent = parent_nodes(asset["parent"])
    assert np.array_equal(parent[:, [0, 1, 5]], measured[:, [0, 1, 5]])
    coarse_cells = columns(asset["parent_tets"], ("parent_tet_row",
                           "parent_node0", "parent_node1", "parent_node2", "parent_node3"))
    assert np.array_equal(coarse_cells[:, 0], np.arange(len(coarse_cells)))
    coarse_tet_nodes = coarse_cells[:, 1:].astype(np.int32)
    assert np.all(coarse_tet_nodes >= 0) and np.all(coarse_tet_nodes < len(parent))

    reference = np.loadtxt(asset["child"], skiprows=1)
    reference_xyz = reference[:, 1:].astype("<f4")
    registration = columns(asset["mapping"], REGISTRATION_FIELDS)
    assert np.array_equal(reference[:, 0], np.arange(1, len(reference)+1))
    child_rows = registration[:, 0].astype(np.intp) - 1
    assert np.all(child_rows >= 0) and np.all(child_rows < len(reference))
    counts = np.bincount(child_rows, minlength=len(reference))
    assert np.all((counts == 1) | (counts == 2))
    assert np.array_equal(registration[:, 1:4].astype("<f4"), reference_xyz[child_rows])
    rows = registration[:, 4].astype(np.int32)
    barycentric = registration[:, 5:9]
    assert np.all(rows >= 0) and np.all(rows < len(coarse_tet_nodes))
    assert np.isfinite(barycentric).all() and np.max(np.abs(barycentric.sum(1) - 1.)) <= 1.e-9
    assert barycentric.min() >= -1.e-8
    corners = coarse_tet_nodes[rows]
    parent_x = parent[:, 2:5]
    delta = measured[:, 2:5] - parent_x
    reconstructed = np.einsum("ij,ijk->ik", barycentric, parent_x[corners]) + registration[:, 9:]
    reconstruction_error = float(np.linalg.norm(reconstructed-reference_xyz[child_rows], axis=1).max())
    assert reconstruction_error <= 2.e-6, reconstruction_error
    mapped = np.einsum("ij,ijk->ik", barycentric, delta[corners])
    if convect_offsets:
        # A projected source point and its registered offset follow the same
        # local affine displacement. This preserves affine motion while
        # leaving reference geometry and the constitutive law intact.
        parent_tet_x = parent_x[coarse_tet_nodes]
        parent_tet_u = delta[coarse_tet_nodes]
        gradient = np.linalg.solve(parent_tet_x[:, 1:] - parent_tet_x[:, :1],
                                   parent_tet_u[:, 1:] - parent_tet_u[:, :1])
        mapped += np.einsum("ij,ijk->ik", registration[:, 9:], gradient[rows])
    u = np.zeros((len(reference), 3))
    np.add.at(u, child_rows, mapped)
    u /= counts[:, None]
    shared = counts[child_rows] == 2
    shared_mismatch = float(2. * np.linalg.norm(mapped[shared] - u[child_rows[shared]], axis=1).max())
    trial = (reference_xyz.astype(np.float64) + u).astype("<f4")
    assert np.isfinite(trial).all()

    tets = np.loadtxt(asset["child_tets"], skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int32)-1
    assert tets.min() >= 0 and tets.max() < len(trial)
    reference_det = determinant(reference_xyz, tets)
    trial_det = determinant(trial, tets)
    assert np.all(reference_det > 0.)
    ratio = trial_det / reference_det
    displacement_ratio = determinant(reference_xyz, tets, u) / reference_det
    worst_cell = int(np.argmin(ratio))

    # Use the registered exterior role, even if a surface row ID is not a
    # consecutive volume node ID. Values below this unchanged bed are reported.
    with asset["mapping"].open() as stream:
        outer = []
        inner = set()
        for row in csv.DictReader(stream):
            layer = row["registered_surface_layer"]
            node = int(row["joint_node_id"]) - 1
            if layer == "O":
                outer.append((int(row["registered_surface_vertex_row"]), node))
            elif layer == "I":
                inner.add(node)
        assert inner == set(np.flatnonzero(counts == 2))
    outer = np.asarray(sorted(outer), dtype=np.int32)
    outer_source = columns(asset["outer"], ("vertex", "wx", "wy", "wz"))
    assert np.array_equal(outer[:, 0], outer_source[:, 0])
    assert np.array_equal(reference_xyz[outer[:, 1]], outer_source[:, 1:].astype("<f4"))
    z = trial[outer[:, 1], 2]
    stats = {
        "coarse_state_sha256": current_sha,
        "canonical_source_pins": {name: env[pair[1]] for name, pair in tests.items()},
        "fine_nodes": len(trial), "fine_cells": len(ratio),
        "registered_offsets_convected": convect_offsets,
        "shared_inner_nodes": len(inner),
        "maximum_shared_interface_disagreement_m": shared_mismatch,
        "maximum_base_reconstruction_error_m": reconstruction_error,
        "initial_max_displacement_m": float(np.linalg.norm(trial-reference_xyz, axis=1).max()),
        "minimum_initial_current_det_ratio": float(ratio[worst_cell]),
        "minimum_reference_relative_det_ratio": float(displacement_ratio.min()),
        "lowest_float32_ratio_source_cell_row": worst_cell,
        "lowest_float32_ratio_source_node_ids": (tets[worst_cell] + 1).tolist(),
        "lowest_float32_ratio_reference_det_m3": float(reference_det[worst_cell]),
        "reference_relative_ratio_at_lowest_float32_cell": float(displacement_ratio[worst_cell]),
        "inverted_cells": int(np.count_nonzero(ratio <= 0.)),
        "cells_below_positive_motion_floor": int(np.count_nonzero(ratio <= 0.35)),
        "minimum_outer_z_against_fixed_bed_m": float(z.min()),
        "outer_nodes_within_10um_of_bed": int(np.count_nonzero(z <= 1.e-5)),
        "bed_penetrating_outer_nodes": int(np.count_nonzero(z < 0.)),
        "untouched_reference_source": str(asset["child"]),
        "physical_equilibrium_qualification": "pending_native_equations",
    }
    if (publish_nodes and stats["cells_below_positive_motion_floor"] == 0
            and stats["minimum_reference_relative_det_ratio"] > 0.35
            and z.min() >= 0. and shared_mismatch <= 5.e-5):
        with output.with_suffix(".node").open("x") as stream:
            stream.write(f"{len(trial)} 3 0 0\n")
            np.savetxt(stream, np.column_stack([reference[:, 0], trial]),
                       fmt=["%d", "%.17g", "%.17g", "%.17g"])
        stats["trial_node_file_sha256"] = digest(output.with_suffix(".node"))
    with output.open("x") as out:
        out.write(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("coarse_parent_node_table", type=Path)
    p.add_argument("coarse_parent_node_table_sha256")
    p.add_argument("report", type=Path)
    p.add_argument("--write-trial", action="store_true")
    p.add_argument("--convect-offsets", action="store_true")
    args = p.parse_args()
    run(args.coarse_parent_node_table, args.coarse_parent_node_table_sha256,
        args.report, args.write_trial, args.convect_offsets)
