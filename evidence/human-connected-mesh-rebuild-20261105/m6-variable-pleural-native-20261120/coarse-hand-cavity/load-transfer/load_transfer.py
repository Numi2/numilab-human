"""Recover elastic retaining forces from the exact native shell and rigid pose."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit import digest, rows
from audit_bed_layer_surfaces import role_surfaces
from audit_bed_motion_shell import terminal_positions


def values(line):
    return {key: value for token in line.split() if "=" in token
            for key, value in (token.split("=", 1),)}


def read_motions(path, steps):
    by_step = {step: {} for step in steps}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            step = int(row["accepted_step"])
            if step not in by_step:
                continue
            body = int(row["body_index"])
            assert body not in by_step[step]
            position = np.array([float(row[f"root_{a}_m"]) +
                                 float(row[f"com_relative_root_{a}_m"]) for a in "xyz"])
            orientation = np.array([float(row[f"orientation_{a}"]) for a in "xyzw"])
            by_step[step][body] = (position, Rotation.from_quat(orientation))
    for step in steps:
        assert by_step[step]
    return by_step


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    log = args.run / "stdout.log"
    cells_file = args.run / "shell-mass.csv.tets.csv"
    nodes_file = args.run / "shell-mass.csv.nodes.csv"
    motions = args.run / "native/resting-body-motion-partition.csv"
    nodes = rows(nodes_file)
    start = np.array([[float(r[f"current_{a}_m"]) for a in "xyz"] for r in nodes], np.float32)
    target = terminal_positions(log, start)
    columns = len(start) // 3
    geometry = start.astype(float)
    assert columns * 3 == len(start)
    deep_faces = np.array(role_surfaces(rows(cells_file))["deep"])
    deep = geometry[deep_faces]
    area = np.linalg.norm(np.cross(deep[:, 1] - deep[:, 0], deep[:, 2] - deep[:, 0]), axis=1) * .5
    tributary = np.bincount(deep_faces.ravel(), weights=np.repeat(area / 3., 3),
                            minlength=len(geometry))
    support_rows, donors, initial, assembly, bed, terminal = [], None, None, None, None, None
    with log.open() as stream:
        for line in stream:
            if line.startswith("whole_body_skin_attachment "):
                support_rows.append(values(line))
            if line.startswith("whole_body_skin_anchor_quality "):
                donors = values(line)
            if line.startswith("resting_initial_anatomical_registration="):
                initial = json.loads(line.partition("=")[2])
            if line.startswith("resting_tissue_shared_runtime="):
                assembly = values(line)
            if line.startswith("resting_fem_skin_bed_accepted_root_reaction "):
                bed = values(line)
            if line.startswith("resting_fascia_fem_terminal_audit="):
                terminal = values(line)
    assert None not in (donors, initial, assembly, bed, terminal)
    assert int(donors["selected"]) == int(donors["elastic_attachments"]) == len(support_rows)
    assert int(donors["hard_ties"]) == 0
    assert int(bed["articulated_support_rows"]) == 0
    complete = json.loads((args.run / "verified-cells.json").read_text())
    assert complete["native_mesh_sha256"] == digest(nodes_file)
    assert complete["connectivity_sha256"] == digest(cells_file)
    assert complete["native_log_sha256"] == digest(log)
    final_step = int(terminal["accepted_steps"])
    assert final_step > 1
    poses = read_motions(motions, (1, final_step))
    before = {r["body_index"]: (np.asarray(r["position_m"]),
                               Rotation.from_quat(r["quaternion_xyzw"]))
              for r in initial["body_poses"]}

    k_total = 0.
    contributions = defaultdict(lambda: {"rows": 0, "stiffness_n_per_m": 0.,
                                         "terminal_force_on_body_n": np.zeros(3)})
    data = []
    for row in support_rows:
        body = int(row["donor_body"])
        col = int(row["local_vertex"])
        tag = int(row["source_vertex"])
        local = 2 * columns + col
        source = nodes[local]
        assert int(source["parent_node_index"]) == local
        assert int(source["node_frame_owner_body"]) == body
        local_point = np.array([float(source[f"donor_local_{a}_m"]) for a in "xyz"], np.float32)
        thickness = np.linalg.norm(geometry[local] - geometry[col + columns])
        reference_area = tributary[local]
        assert min(reference_area, thickness) > 0.
        start_p, start_r = before[body]
        final_p, final_r = poses[final_step][body]
        source_delta = start_p + start_r.apply(local_point) - geometry[local]
        assert np.linalg.norm(source_delta) < 4.e-7, (body, tag, source_delta)
        rel = target[local] - (final_p + final_r.apply(local_point))
        stiffness = float(np.float32(float(donors["shear_prior_pa"]) / thickness) *
                          np.float32(reference_area))
        load = stiffness * rel
        k_total += stiffness
        entry = contributions[body]
        entry["rows"] += 1
        entry["stiffness_n_per_m"] += stiffness
        entry["terminal_force_on_body_n"] += load
        data.append(dict(body=body, source_vertex=tag, shell_node=local,
                         thickness_m=float(thickness), area_m2=float(reference_area),
                         stiffness_n_per_m=stiffness, startup_offset_m=source_delta.tolist(),
                         terminal_displacement_m=rel.tolist(), terminal_body_force_n=load.tolist()))
    for entry in contributions.values():
        entry["terminal_force_on_body_n"] = entry["terminal_force_on_body_n"].tolist()
    body_weight_n = float(assembly["FEM_mass_kg"]) * 9.81
    with motions.open() as stream:
        body_weight_n += sum(float(row["mass_kg"]) * 9.81
                             for row in csv.DictReader(stream) if int(row["accepted_step"]) == 1)
    result = dict(
        scope=("Reconstruction of the direct elastic SAMPLE force path only. Separate "
               "registered triangle and other tissue contacts retain their native ownership."),
        input_sha256={p.name: digest(p) for p in (log, cells_file, nodes_file, motions)},
        terminal_step=final_step, assembly_weight_n=body_weight_n,
        bed_reaction_n=float(bed["mean_normal_force_N"]),
        tributary_area_m2=sum(r["area_m2"] for r in data),
        direct_ties_by_body=contributions,
        total_direct_stiffness_n_per_m=k_total,
        terminal_direct_force_on_body_n=np.sum([row["terminal_body_force_n"] for row in data], axis=0).tolist(),
        synchronized_vertical_displacement_for_weight_m=body_weight_n / k_total,
        maximum_terminal_spring_deflection_m=max(np.linalg.norm(r["terminal_displacement_m"]) for r in data),
        source_samples=data,
    )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print({key: v for key, v in result.items() if key not in ("source_samples", "input_sha256")})


if __name__ == "__main__":
    main()
