"""Map a GPU witness to its material star once a compiled element offset is known."""
from pathlib import Path
import json,sys
import numpy as np
def run(out,offset,witness):
    v=out/'relative/volume';reader=lambda f:np.loadtxt(f,usecols=(1,2,3),skiprows=1).astype('f4').astype(float)
    x,y=reader(v/'mesh.node'),reader(v/'current.node');e=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=e[:,:4]-1;c=witness-offset;nodes=t[c];a=x[nodes[1:]]-x[nodes[0]];b=y[nodes[1:]]-y[nodes[0]];F=np.linalg.solve(a,b)
    ids=np.flatnonzero(np.isin(t,nodes).any(1));z=np.load(out.parent/'registered-boundary/registered-boundary.npz')
    layers={k:np.intersect1d(nodes,z[k+'_node_joint']).tolist() for k in 'OID'}
    role=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')[c];court=np.load(out/'binding/contact/material-contact.npz')['samples']
    result=dict(global_witness=witness,compiled_offset=offset,material_cell=c,source_label=role.tolist(),nodes=nodes.tolist(),reference=x[nodes].tolist(),starting=y[nodes].tolist(),displacement_mm=((y[nodes]-x[nodes])*1000).tolist(),singular_values=np.linalg.svd(F,compute_uv=False).tolist(),initial_J=float(np.linalg.det(F)),shape_condition=float(np.linalg.cond(a)),incident_cells=len(ids),layers=layers,cortical_nodes=np.intersect1d(nodes,court[:,0]).tolist())
    print(json.dumps(result,indent=2))
if __name__=='__main__':run(Path(sys.argv[1]).resolve(),int(sys.argv[2]),int(sys.argv[3]))
