"""Apply a localized, fixed-connectivity repair on its pinned connected material reference."""
import sys,json
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
mesh,parent,child,out=map(Path,sys.argv[1:])
record=json.loads((child.parent/'result.json').read_text())
if sha(mesh/'mesh.node')!=record['physical_source_sha256'] or sha(child)!=record['proposal_sha256'] or record['failed_cells_frozen_endpoint']:
    raise ValueError('Unmatched or failing physical reference patch')
with np.load(parent) as a,np.load(child) as b:
    points={int(i):x for i,x in zip(a['nodes'],a['x'])}
    points.update({int(i):x for i,x in zip(b['nodes'],b['x'])})
    ids=np.array(sorted(points),dtype='i4')
    rows=np.union1d(a['cell'],b['cell'])
with (mesh/'mesh.ele').open() as f:
    n=int(next(f).split()[0]);el=np.loadtxt(f,max_rows=n,usecols=(1,2,3,4,5),dtype='i4')[rows]
# Preserve every row of each touched cell for the P1 and density replay.
nodes=np.union1d(ids,np.unique(el[:,:4]-1))
with (mesh/'mesh.node').open() as f:
    n=int(next(f).split()[0]);x=np.loadtxt(f,max_rows=n,usecols=(1,2,3),dtype='f4')[nodes]
x[np.searchsorted(nodes,ids)]=np.array([points[int(i)] for i in ids])
np.savez_compressed(out,nodes=nodes,cell=rows,x=x,t=np.searchsorted(nodes,el[:,:4]-1).astype('i4'),role=el[:,4])
