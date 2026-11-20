import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path.cwd()
trial = root / "pectoral-energy-trial"
samples = defaultdict(lambda: ({}, {}))
with (trial / "energy.csv").open() as stream:
    for row in csv.reader(stream):
        kind, outer, tick, stage, status, index = row[:6]
        key = tuple(map(int, (outer,tick,stage,status)))
        nodes, cells = samples[key]
        table = nodes if kind == "n" else cells
        assert kind in {"n","e"} and int(index) not in table
        table[int(index)] = np.asarray(row[6:], dtype=float)

history = []
for (outer,tick,stage,status),(nodes,cells) in sorted(samples.items()):
    assert sorted(nodes) == list(range(38)), (outer,tick,stage,status)
    assert sorted(cells) == list(range(54)), (outer,tick,stage,status)
    node = np.stack([nodes[i] for i in range(len(nodes))])
    cell = np.stack([cells[i] for i in range(len(cells))])
    assert node.shape == (38,27)
    assert cell.shape == (54,37)
    assert np.isfinite(node).all() and np.isfinite(cell).all()
    dt = float(node[0,0])/256
    mass = node[:,1]
    fixed = node[:,2] != 0
    old_u, old_v = node[:,6:9], node[:,9:12]
    new_u, new_v = node[:,12:15], node[:,15:18]
    ext, reaction, rhs = node[:,18:21], node[:,21:24], node[:,24:27]
    tets = cell[:,:4].astype(int)-194
    assert tets.min() == 0 and tets.max() == 37
    inv_rest = cell[:,5:17].reshape(-1,3,4)[:,:,:3]
    du = new_u[tets[:,1:]]-new_u[tets[:,[0]]]
    f = np.eye(3)+np.swapaxes(du,1,2) @ inv_rest
    j = np.linalg.det(f)
    j_safe = np.where(j>0,j,np.nan)
    shear,bulk = 1840,500000
    log_j = np.log(j_safe)
    volume = cell[:,4]
    matrix_energy = np.sum(volume*(0.5*shear*(np.sum(f*f,axis=(1,2))-3-2*log_j)
                                   +0.5*bulk*log_j**2))
    per_cell_force = cell[:,21:].reshape(-1,4,4)[:,:,:3]
    internal = np.zeros((38,3))
    np.add.at(internal,tets.ravel(),per_cell_force.reshape(-1,3))
    p_iso = shear*f+(bulk*log_j-shear)[:,None,None]*np.swapaxes(np.linalg.inv(f),1,2)
    iso_cols = -volume[:,None,None]*(p_iso @ np.swapaxes(inv_rest,1,2))
    iso_forces = np.zeros_like(per_cell_force)
    iso_forces[:,1:] = np.swapaxes(iso_cols,1,2)
    iso_forces[:,0] = -iso_forces[:,1:].sum(axis=1)
    delta_u = new_u-old_u
    # Kinetic energy and work here are for this one fascia object only.
    body_force_on_anchors = -reaction[fixed]
    unbalanced = ((internal+ext)*dt-mass[:,None]*(new_v-old_v))
    result = {
        "root":outer,"microtick":tick,"stage": "certificate" if stage else "assembled",
        "status":status,"timestep_s":dt,
        "j_min":float(np.min(j)), "j_max":float(np.max(j)), "j_tet274":float(j[4]),
        "old_kinetic_j":float(np.sum(0.5*mass*np.sum(old_v**2,axis=1))),
        "new_kinetic_j":float(np.sum(0.5*mass*np.sum(new_v**2,axis=1))),
        "isotropic_strain_energy_j":float(matrix_energy),
        "max_speed_m_s":float(np.max(np.linalg.norm(new_v,axis=1))),
        "max_speed_node":194+int(np.argmax(np.linalg.norm(new_v,axis=1))),
        "peak_external_force_n":float(np.max(np.linalg.norm(ext,axis=1))),
        "peak_external_force_node":194+int(np.argmax(np.linalg.norm(ext,axis=1))),
        "external_force_nodes":[194+int(i) for i in np.flatnonzero(np.any(ext!=0,axis=1))],
        "external_incremental_work_j":float(np.sum(ext*delta_u)),
        "boundary_incremental_work_j":float(np.sum(body_force_on_anchors*delta_u[fixed])),
        "internal_incremental_work_j":float(np.sum(internal*delta_u)),
        "free_momentum_balance_without_gravity_ns":float(np.linalg.norm(unbalanced[~fixed]-rhs[~fixed])),
        "free_motion_update_error_m":float(np.max(np.abs((delta_u-dt*new_v)[~fixed]))),
        "witness_corners_internal_n":per_cell_force[4].tolist(),
        "witness_corners_isotropic_n":iso_forces[4].tolist(),
        "witness_corners_net_internal_n":internal[[20,1,2,37]].tolist(),
    }
    history.append(result)

for row in history:
    if row["stage"] != "certificate" or row["status"] != 0:
        continue
    if row["microtick"] not in (0,16,48,60,61,255):
        continue
    print("{root}:{microtick:<3} J={j_tet274:<8.5f} Jrange={j_min:.5f}..{j_max:.5f} "
          "maxv={max_speed_m_s:8.4f} Fext={peak_external_force_n:8.4f} node={peak_external_force_node} "
          "Ek={new_kinetic_j:11.5g} U={isotropic_strain_energy_j:11.5g} "
          "Wext={external_incremental_work_j:11.5g} Wanchor={boundary_incremental_work_j:11.5g} "
          "motion_error={free_motion_update_error_m:.3g}".format(**row))
(root/"pectoral-energy-witness.json").write_text(json.dumps(history, indent=2)+"\n")
