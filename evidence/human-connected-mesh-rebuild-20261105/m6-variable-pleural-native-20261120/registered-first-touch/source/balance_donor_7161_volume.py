"""Make the sole deficient source domain density-feasible with a small inner-layer displacement."""
from pathlib import Path
import json
import subprocess

import numpy as np

from candidate import load

root = Path.cwd()
base, x, tets, mat, origin, _, roles, _, _ = load()
lineage_row = 401330
rows = np.flatnonzero(origin == lineage_row)
assert len(rows) == 1 and mat[rows[0]] == 15
target_row = int(rows[0])
nodes = tets[target_row]
assert np.all(roles[nodes] == 'I')
point = x[nodes].astype(float)
edge = point[1:] - point[0]
cross = np.cross(np.roll(edge, -1, axis=0), np.roll(edge, 1, axis=0))
grad = np.vstack((-cross.sum(axis=0), cross)) / 6.
volume = float(np.dot(edge[0], cross[0]) / 6.)
shortage = 1.1347015774352492e-6
target_volume_decrease = (shortage + 2e-8) / 900.
assert 0.02 < target_volume_decrease / volume < 0.10
affected = np.flatnonzero(np.isin(tets, nodes).any(axis=1))
assert len(affected) < 200
tet_edge_i = np.array([0, 0, 0, 1, 1, 2])
tet_edge_j = np.array([1, 2, 3, 2, 3, 3])

def evaluate(position):
    points = position[tets[affected]].astype(float)
    sides = points[:, tet_edge_j] - points[:, tet_edge_i]
    a, b, c = np.moveaxis(points[:, 1:] - points[:, :1], 1, 0)
    det = np.einsum('ni,ni->n',a,np.cross(b,c))
    length = np.einsum('nki,nki->nk', sides, sides).max(axis=1)
    return det / 6., det / length**1.5

before, old_quality = evaluate(x)
assert np.all(old_quality >= 1e-5)
which = int(np.flatnonzero(affected == target_row)[0])
deltas = {}
for count in (1, 2, 4):
    strongest = np.argsort(-np.linalg.norm(grad, axis=1))[:count]
    delta = np.zeros_like(grad)
    delta[strongest] = - target_volume_decrease * grad[strongest] / np.square(grad[strongest]).sum()
    deltas[count] = delta
ordered = []
for count, delta in deltas.items():
    for multiplier in (1.001, 1.02, 1.06):
        proposal = x.copy()
        proposal[nodes] = point + multiplier * delta
        actual, quality = evaluate(proposal)
        displacement = float(np.linalg.norm(proposal[nodes].astype(float)-point, axis=1).max())
        if np.min(quality) < 1e-5 or volume-actual[which] < (shortage+1e-8)/900. or displacement > .00025:
            continue
        ordered.append((displacement, count, proposal, actual, quality))
assert ordered, 'no determinant-safe candidate within 0.25 mm'

packed = (root / 'fine-boundary-f32.bin').read_bytes()
for displacement, count, proposal, volumes, quality in sorted(ordered,key=lambda candidate:candidate[0]):
    trial = root / 'fine-mass-local-candidate.bin'
    with trial.open('wb') as out:
        out.write(packed[:8+12*len(x)])
        proposal.astype('<f4',copy=False).tofile(out)
        out.write(packed[8+24*len(x):])
    log = subprocess.check_output(['./audit_fine_boundary',str(trial)],text=True)
    counts = [int(line.split()[-1]) for line in log.splitlines() if line.startswith('pair ')]
    assert len(counts) == 20
    print('inner patch', count, 'movement_m',displacement,'normalized_determinant_min',quality.min(),
          'crossings',sum(counts),flush=True)
    if any(counts):
        continue
    (root / 'fine-mass-local-boundary-audit.log').write_text(log)
    q = np.unique(nodes[np.any(proposal[nodes] != x[nodes],axis=1)])
    np.savez_compressed('fine-mass-local-position-delta.npz',changed_rows=q,positions_f32=proposal[q],
                       affected_rows=affected, volume_m3=volumes)
    (root / 'fine-mass-local-fit.json').write_text(json.dumps({
        'source_parent_row':7161, 'source_child':lineage_row,
        'changed_inner_nodes':q.tolist(),'max_displacement_m':displacement,
        'incident_cells':len(affected),'original_target_volume_m3':volume,
        'repaired_target_volume_m3':float(volumes[which]),
        'minimum_incident_normalized_determinant':float(quality.min()),
        'crossings':sum(counts),
    },indent=2)+'\n')
    break
else:
    raise RuntimeError('all density-volume patches produced a boundary crossing')
