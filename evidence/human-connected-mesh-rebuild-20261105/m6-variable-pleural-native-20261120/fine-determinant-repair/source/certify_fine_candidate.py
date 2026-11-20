"""Census the full emitted fine volume and its material mass envelope."""
import hashlib
import json
from pathlib import Path

import numpy as np

from candidate import load

original, x, cells, material, _, _, roles, _, _ = load()
with np.load('fine-source-carried-mass.npz', allow_pickle=False) as mass:
    m=mass['element_mass_kg']
assert len(cells)==len(m) and np.isfinite(x).all()
minimum=np.full(2,np.inf)
inverted=np.zeros(2,dtype=int)
low_quality=np.zeros(2,dtype=int)
volume=np.zeros(len(cells))
finite_faces=True
edge_i=np.array([0,0,0,1,1,2])
edge_j=np.array([1,2,3,2,3,3])
for offset in range(0,len(cells),32768):
    part=slice(offset,offset+32768)
    p=x[cells[part]].astype(float)
    edges=p[:,edge_j]-p[:,edge_i]
    l2=np.einsum('nki,nki->nk',edges,edges).max(axis=1)
    a,b,c=np.moveaxis(p[:,1:]-p[:,:1],1,0)
    det=np.einsum('ni,ni->n',a,np.cross(b,c))
    volume[part]=det/6
    q=det/l2**1.5
    assert np.all(np.isfinite(q))
    for lane,label in enumerate((15,16)):
        t=q[material[part]==label]
        if len(t):
            minimum[lane]=min(minimum[lane],t.min())
            inverted[lane]+=np.count_nonzero(t<=0)
            low_quality[lane]+=np.count_nonzero((t>0)&(t<1e-5))
audit=Path('fine-boundary-corrected-audit.log')
counts=[int(row.split()[-1]) for row in audit.read_text().splitlines() if row.startswith('pair ')]
assert len(counts)==20 and not any(counts)
packed=Path('fine-boundary-f32.bin').read_bytes()
assert np.array_equal(np.frombuffer(packed,dtype='<f4',count=3*len(x),offset=8+12*len(x)).reshape(-1,3),x)
rho=m/volume
result=dict(
    cell_count=len(cells),
    node_count=len(x),
    cell_inversions_by_material=dict(zip(('15','16'),map(int,inverted))),
    cell_below_quality_floor_by_material=dict(zip(('15','16'),map(int,low_quality))),
    normalized_determinant_min_by_material=dict(zip(('15','16'),map(float,minimum))),
    oriented_fine_boundary_pair_crossings=sum(counts[10:]),
    boundary_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
    position_sha256=hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest(),
    cell_connectivity_sha256=hashlib.sha256(cells.astype('<i4',copy=False).tobytes()).hexdigest(),
    mass_distribution_by_material={},
    subject='offline_registered_skin_and_fat',
    native_motion_verified=False)
for label,lb,ub in ((15,900,1100),(16,850,950)):
    t=material==label
    density=rho[t]
    v=volume[t].sum()
    result['mass_distribution_by_material'][str(label)]=dict(
        bounds_kg_m3=[lb,ub],
        cell_volume_m3=float(v),mass_kg=float(m[t].sum()),
        average_kg_m3=float(m[t].sum()/v),
        density_below_gate=int(np.count_nonzero(density<lb*(1-1e-8))),
        density_above_gate=int(np.count_nonzero(density>ub*(1+1e-8))),
        extra_mass_to_raise_under_minimum_cells_kg=float(np.maximum(0,lb*volume[t]-m[t]).sum()),
        excess_mass_above_maximum_cells_kg=float(np.maximum(0,m[t]-ub*volume[t]).sum()))
assert not np.any(inverted) and not np.any(low_quality)
Path('fine-final-certificate.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed('fine-reference-measures.npz',volume_m3=volume)
print(json.dumps(result),flush=True)
