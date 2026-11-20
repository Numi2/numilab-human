"""Fit source fascia columns jointly while preserving their thickness and source volume."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import struct

import numpy as np
from scipy.optimize import minimize

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weight-anchors", type=float, default=16.)
args = parser.parse_args()

source_id = json.loads((root.parent / "resting-clearance-and-atrium-016/fascia-source-cell-witness.json").read_text())
raw = Path(source_id["asset"]).read_bytes()
_, nr, nodes, cells = struct.unpack_from("<4I", raw, 8)
regions = np.frombuffer(raw, offset=104, count=nr * 8, dtype="<u4").reshape(-1, 8)
base = 104 + nr * 32
core = np.ndarray((nodes, 3), dtype="<f4", buffer=raw, offset=base, strides=(32, 4)).astype(float)
source_vertex = np.ndarray(nodes, dtype="<u4", buffer=raw, offset=base+20, strides=(32,))
tets = np.frombuffer(raw, offset=base+nodes*32, count=cells*5, dtype="<u4").reshape(-1, 5)
native = np.loadtxt(root / "paired-offset-fascia-audit-run/fascia-rest.csv", delimiter=",", skiprows=1)
assert np.array_equal(native[:, 0], np.arange(nodes))
result = native[:, 1:4].astype("<f4")
edges = np.array(list(combinations(range(4), 2)))
quality_floor = 1.e-5
reports = []

for region in range(6):
    first, length = map(int, regions[region, 3:5])
    rows = slice(first, first+length)
    half = length//2
    assert np.array_equal(source_vertex[rows][:half], source_vertex[rows][half:])
    local = tets[tets[:,4] == region, :4] - first
    ids = local % half
    source = 1000. * core[rows]
    y = 1000. * native[rows, 1:4]
    anchor = native[rows, 4] != 0
    weight = np.where(anchor, args.weight_anchors, 1.)
    source_relative, y_relative = source-source.mean(axis=0), y-y.mean(axis=0)
    u, _, v = np.linalg.svd(source_relative.T @ y_relative)
    u[:, -1] *= np.linalg.det(u @ v)
    rigid = source_relative @ u @ v + y.mean(axis=0)
    a = source[local]
    determinant = np.linalg.det(a[:, 1:] - a[:, :1])
    reference_sign = np.sign(determinant)
    source_measure = abs(determinant).sum()
    take_row = np.repeat(np.arange(len(local)), 12)
    take_col = (3*ids[:, :, None] + np.arange(3)).ravel()

    def measure(q, x=None):
        if x is None:
            delta = q.reshape(-1,3)
            x = rigid + np.concatenate((delta,delta))
        p = x[local]
        e = p[:, 1:] - p[:, :1]
        pair_delta = p[:, edges[:, 0]] - p[:, edges[:, 1]]
        squared = np.sum(pair_delta**2, axis=-1)
        longest = np.argmax(squared,axis=1)
        line = np.arange(len(local))
        l = squared[line, longest]
        scale = l**1.5
        cofactors = np.cross(e[:, [1,2,0]], e[:, [2,0,1]])
        signed = reference_sign * np.einsum("ij,ij->i", e[:, 0], cofactors[:, 0])
        gradient = reference_sign[:, None, None] * np.concatenate(
            (-cofactors.sum(axis=1)[:, None], cofactors),axis=1)
        quality = signed/scale
        sl = np.zeros_like(gradient)
        scale_gradient = 3*pair_delta[line, longest] / l[:, None]
        sl[line, edges[longest, 0]] = scale_gradient
        sl[line, edges[longest, 1]] = -scale_gradient
        slope = gradient/scale[:,None,None] - quality[:,None,None]*sl
        jacobian = np.zeros((len(local),3*half))
        np.add.at(jacobian,(take_row,take_col),slope.ravel())
        mass_jacobian = np.zeros(3*half)
        np.add.at(mass_jacobian,take_col,gradient.ravel()/source_measure)
        return 1e3*quality,1e3*jacobian,float(signed.sum()/source_measure-1.),mass_jacobian

    def objective(q):
        delta = q.reshape(-1, 3)
        mismatch = rigid + np.concatenate((delta,delta)) - y
        weighted = weight[:,None] * mismatch
        return .5*np.sum(weighted*mismatch), (weighted[:half]+weighted[half:]).ravel()
    initial = np.zeros(3*half)
    setup_min = measure(initial)[0].min()
    assert setup_min >= quality_floor*1000
    model = minimize(objective,initial,jac=True,method="SLSQP",constraints=[
        {"type":"ineq","fun":lambda q:measure(q)[0]-1.1e3*quality_floor,
         "jac":lambda q:measure(q)[1]},
        {"type":"eq","fun":lambda q:measure(q)[2], "jac":lambda q:measure(q)[3]},
    ],options={"maxiter":200,"ftol":1e-9})
    delta = model.x.reshape(-1,3)
    candidate = ((rigid+np.vstack((delta,delta)))*.001).astype("<f4")
    q, _, volume_error, _ = measure(None, x=candidate.astype(float)*1000)
    error = np.linalg.norm(candidate-y*.001,axis=1)
    displacement = np.linalg.norm(candidate-rigid*.001,axis=1)
    record = {"region":region,"iterations":int(model.nit),"solver_success":bool(model.success),
              "solver_result":str(model.message),"minimum_normalized_volume":float(q.min()*1e-3),
              "relative_source_volume_error":abs(volume_error),
              "maximum_fit_change_m":float(error.max()),"rms_fit_change_m":float(np.linalg.norm(error)/np.sqrt(len(error))),
              "maximum_fixed_node_fit_change_m":float(error[anchor].max()),
              "maximum_rigid_delta_m":float(displacement.max())}
    print(json.dumps(record),flush=True)
    reports.append(record)
    result[rows] = candidate

geometry_pass = all(r["minimum_normalized_volume"] >= quality_floor and
                    r["relative_source_volume_error"] <= 2.e-4 for r in reports)
(npz := root/"pectoral-material-sheet-f32.npz").exists() and exit("candidate already exists")
np.savez_compressed(npz,positions_world_m=result)
(root/"pectoral-material-sheet.json").write_text(json.dumps({
    "regions":reports, "orientation_and_volume_screen_pass":geometry_pass,
    "physical_admission":False,
    "scope":"Paired source layer positions, target orientation, source signed volume and point proximity. No tissue crossing, native admission or dynamic claim.",
},indent=2)+"\n")
assert geometry_pass
