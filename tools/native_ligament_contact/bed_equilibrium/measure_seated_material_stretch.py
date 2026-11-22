"""Measure principal stretches of each material at installation, with original cells and positions."""
import json,sys
from pathlib import Path
import numpy as np
cfg,current,out=map(Path,sys.argv[1:])
env=json.loads(cfg.read_text())['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
rows=np.loadtxt(env[prefix+'TETGEN_NODE'],skiprows=1);coords=np.loadtxt(current,skiprows=1)
assert np.array_equal(rows[:,0],coords[:,0])
ref=rows[:,1:].astype('f4').astype(float);q=coords[:,1:];assert np.array_equal(q,q.astype('f4'))
ele=np.loadtxt(env[prefix+'TETGEN_ELEMENTS'],skiprows=1,dtype=np.int32);t=ele[:,1:5]-1;tag=ele[:,5]
u=q-ref;stretches=np.empty((len(t),3));vol=np.empty(len(t))
for start in range(0,len(t),24576):
    sl=slice(start,min(len(t),start+24576));ids=t[sl]
    dm=(ref[ids[:,1:]]-ref[ids[:,:1]]).transpose(0,2,1)
    du=(u[ids[:,1:]]-u[ids[:,:1]]).transpose(0,2,1)
    F=np.eye(3)+du@np.linalg.inv(dm).astype('f4').astype(float)
    stretches[sl]=np.linalg.svd(F,compute_uv=False);vol[sl]=np.linalg.det(dm)/6.
record={}
for mat in np.unique(tag):
    ids=np.flatnonzero(tag==mat);s=stretches[ids];worst=int(ids[s[:,0].argmax()])
    record[str(mat)]={'cells':len(ids),'min_stretch':float(s[:,2].min()),'max_stretch':float(s[:,0].max()),
        'cells_max_stretch_over_1_5':int((s[:,0]>1.5).sum()),'cells_min_stretch_under_0_5':int((s[:,2]<.5).sum()),
        'reference_volume_fraction_max_stretch_over_1_5':float(vol[ids[s[:,0]>1.5]].sum()/vol[ids].sum()),
        'worst_stretch_cell':worst,'worst_cell_xyz_m':q[t[worst]].mean(0).tolist()}
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
