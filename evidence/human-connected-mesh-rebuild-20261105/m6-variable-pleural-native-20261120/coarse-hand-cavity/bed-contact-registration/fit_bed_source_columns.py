"""Fit physical bed-facing columns using registered outer skin witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np

from build_hand_trial import digest
from fit_bank_group_partitions import ROLE_COUNT, fit_group, groups, refinement_edges
from fit_hand_cavity import PARENT, table
from fit_hand_interface import coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_fit", type=Path)
    parser.add_argument("native_replay", type=Path)
    parser.add_argument("stencils", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-ring", action="store_true")
    parser.add_argument("--preserve-deep-anchors", action="store_true",
                        help="Keep the deep fat interface coordinates fixed to the registered reference")
    parser.add_argument("--fit-bed-normal", action="store_true",
                        help="Fit only the plane-normal gap at existing nodes near the source witnesses")
    args = parser.parse_args()
    node_path = PARENT.with_name(PARENT.name+".nodes.csv")
    cell_path = PARENT.with_name(PARENT.name+".tets.csv")
    reference = coordinates(table(node_path), "rest")
    cells = table(cell_path)
    topology = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells])
    materials = np.array([int(row["material_index"]) for row in cells])
    prior = json.loads(args.source_fit.read_text())
    assert prior["registered_parent_nodes_sha256"] == digest(node_path)
    assert prior["registered_parent_cells_sha256"] == digest(cell_path)
    candidate = reference.copy()
    for row in prior["candidate_nodes"]:
        candidate[row["local_node"]] += row["world_delta_m"]
    node_dump = args.native_replay / "initial-skin-fem.csv"
    natives = table(node_dump)
    natives.sort(key=lambda r: int(r["global_node"]))
    emitted = np.array([[float(row[f"{axis}_m"]) for axis in "xyz"] for row in natives])
    assert len(emitted) == len(candidate)
    assert np.max(np.abs(candidate-emitted)) < 3.e-7
    stencils = json.loads(args.stencils.read_text())
    assert [r["detailed_vertex_id"] for r in stencils] == [17764, 21161, 34320]
    # Share the existing native first-touch clearance and fixed flat bed.
    floor_gap = float(np.min(emitted[:ROLE_COUNT, 2]))
    assert 0. < floor_gap < 1.e-6
    anchors = {}
    reports = []
    for stencil in stencils:
        row = stencil["nearest_physical_rows"][0]["row"]
        detail = np.array(stencil["detailed_point_m"])
        assert abs(detail[2]) < 1.e-7
        detail[2] = floor_gap
        if args.fit_bed_normal:
            # Retain each node's in-plane registration. The witnesses select the
            # two physical patches, while the existing nodes carry bed contact.
            detail[:2] = emitted[row, :2]
        delta = detail - emitted[row]
        if np.linalg.norm(delta) < 1.e-6:
            continue
        anchors[row] = (stencil["detailed_vertex_id"], delta)
    edges = refinement_edges()
    for row, (source_id, delta) in anchors.items():
        graph_seed = (row, np.zeros(3), 0., (0, 0))
        [(columns, _, _)] = groups(topology, [graph_seed], edges, args.extra_ring)
        fixed = set()
        if args.preserve_deep_anchors:
            fixed = {2 * ROLE_COUNT + int(column) for column in columns}
        settled, report = fit_group(candidate, topology, materials, [], columns,
                                    edges, fixed, polish=True, sparse=True,
                                    point_targets={row: delta})
        reports.append({"outer_row": row, "detailed_vertex": source_id,
                        "requested_delta_m": delta.tolist(),
                        "accepted_fit": settled is not None, **report})
        if settled is not None:
            if fixed:
                deep = sorted(fixed)
                assert np.array_equal(settled[deep], candidate[deep])
            candidate = settled
    changed = np.flatnonzero(np.any(candidate != reference, axis=1))
    result = {**prior, "scope": "Registered detail to physical bed-facing skin column reconciliation.",
              "fit_parent_sha256": digest(args.source_fit),
              "native_before_sha256": digest(node_dump),
              "support_stencils_sha256": digest(args.stencils),
              "preserve_deep_anchors": args.preserve_deep_anchors,
              "fit_bed_normal": args.fit_bed_normal,
              "bed_geometry_groups": reports,
              "tangent_plane_fit_pass": all(r["accepted_fit"] for r in reports)
              and float(candidate[:ROLE_COUNT, 2].min()) >= 0.,
              "candidate_nodes": [{"local_node": int(node),
                                    "source_world_m": reference[node].tolist(),
                                    "world_delta_m": (candidate[node]-reference[node]).tolist()}
                                   for node in changed]}
    assert len(reports) == 2
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{k:r[k] for k in ("outer_row", "detailed_vertex", "accepted_fit")}
                      for r in reports]), flush=True)


if __name__ == "__main__":
    main()
