"""Check the native endpoint against the declared two passive energy laws."""
from pathlib import Path
import json
import re
import sys

import numpy as np

from fit_hand_cavity import PARENT, ROWS, table, digest

AXES = "xyz"
MATERIALS = (
    "human_skin_dermis_frequency_specific_effective.nmatter",
    "human_subcutaneous_adipose_low_strain_effective.nmatter",
)


def parameters(path):
    text = path.read_text()
    return tuple(1.e6 * float(re.search(
        rf"parameter {name} : MPa = ([.0-9e+-]+)", text)[1])
        for name in ("shear", "bulk"))


def columns(row, key):
    return np.array([float(row[key + axis]) for axis in AXES])


def main():
    case = Path(sys.argv[1])
    source_dir = Path.home() / "Work/skin-fat-native-integration-209/matter/materials"
    parameter_paths = [source_dir / name for name in MATERIALS]
    mu_bulk = np.array([parameters(p) for p in parameter_paths])
    nodes = {}
    jacobians = {}
    completed = None
    log = case / "stdout.log"
    with log.open() as stream:
        for line in stream:
            if line.startswith("resting_integrated_body=completed "):
                completed = dict(pair.split("=",1) for pair in line.split()[1:])
            if not line.startswith(("resting_FEM_node,","resting_FEM_tet,")):
                continue
            kind, *items = line.rstrip().split(",")
            row = dict(s.split("=", 1) for s in items)
            if row["object"] != "BodyParts3D whole-body skin-fat bonded shell":
                continue
            key = int(row["local"])
            if kind == "resting_FEM_node":
                assert key not in nodes and int(row["reference_mode"]) == 1
                nodes[key] = row
            else:
                jacobians[key] = float(row["J"])
    assert completed and completed["physiology_body_clock"] == "matched"
    assert completed["root_assistance"] == "false"
    cells = table(PARENT.with_name(PARENT.name + ".tets.csv"))
    assert len(nodes) == 7233 and len(jacobians) == len(cells) == 28632
    ref = np.array([columns(nodes[j],"initial_rest_").astype(np.float32) for j in range(len(nodes))],dtype=np.float64)
    disp = np.array([columns(nodes[j],"reference_u_hi_") + columns(nodes[j],"reference_u_lo_")
                     for j in range(len(nodes))])
    tet = np.array([[int(row[f"parent_node{j}"]) for j in range(4)] for row in cells])
    mat = np.array([int(row["material_index"]) - 15 for row in cells])
    assert set(mat) == {0,1}
    material = mu_bulk[mat]
    reference_edges = np.swapaxes(ref[tet[:,1:]] - ref[tet[:,:1]],1,2)
    reference_det = np.linalg.det(reference_edges)
    assert np.all(reference_det > 0.)
    inverse = np.linalg.inv(reference_edges)
    u = np.swapaxes(disp[tet[:,1:]] - disp[tet[:,:1]],1,2)
    f = np.eye(3)[None] + u @ inverse
    j = np.linalg.det(f)
    assert np.all(j > 0)
    inverse_f_t = np.swapaxes(np.linalg.inv(f),1,2)
    p = material[:,0,None,None] * (f - inverse_f_t)
    p += (material[:,1] * np.log(j))[:,None,None] * inverse_f_t
    columns_123 = np.swapaxes(-(reference_det / 6)[:,None,None] * p @ np.swapaxes(inverse,1,2),1,2)
    vertex_forces = np.concatenate((-columns_123.sum(axis=1)[:,None,:], columns_123),axis=1)
    native_j = np.array([jacobians[row] for row in range(len(cells))])
    report = {
        "native_stdout_sha256": digest(log),
        "native_launch_sha256": digest(case/"launch.json"),
        "native_completed": completed,
        "material_sha256": {path.name: digest(path) for path in parameter_paths},
        "maximum_j_error": float(np.max(np.abs(j-native_j))),
        "endpoint_j_min_max": [float(j.min()),float(j.max())],
        "external_balance_scope": "This comparison checks tag-0 element force only, without contact or constrained nodes.",
        "witness_cells": [{"row":k, "J_oracle":float(j[k]),"J_native":float(native_j[k])} for k in ROWS],
        "interface_node_force": {},
    }
    assert report["maximum_j_error"] < 1.e-8
    for k in (2261, 4656, 4672, 4673, 4693):
        indices, corners = np.nonzero(tet==k)
        individual = vertex_forces[indices, corners]
        oracle = individual.sum(axis=0)
        assert nodes[k]["constraint_tag"] == "0"
        native = np.array([
            float(nodes[k][f"free_internal_force_{v}_N"]) for v in AXES])
        assert np.linalg.norm(oracle-native) < 1.e-4
        row_material = mat[indices]
        largest = np.argsort(-np.linalg.norm(individual,axis=1))[:4]
        report["interface_node_force"][str(k)] = {
            "native_N": native.tolist(),
            "oracle_N": oracle.tolist(),
            "error_norm_N": float(np.linalg.norm(oracle-native)),
            "oracle_skin_N": individual[row_material==0].sum(axis=0).tolist(),
            "oracle_fat_N": individual[row_material==1].sum(axis=0).tolist(),
            "accepted_displacement_m": disp[k].tolist(),
            "node_velocity_m_s": [float(nodes[k]["v"+axis]) for axis in AXES],
            "compiled_lumped_mass_kg": float(nodes[k]["mass_kg"]),
            "strongest_oracle_tets": [{"row":int(indices[i]),"force_N":individual[i].tolist()} for i in largest],
        }
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
