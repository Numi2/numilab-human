"""Determine which severe deformation cells can be corrected by interior registration."""
import sys,json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
def run(job,out):
    mat=job/'interval-certified-scene/material';a,b=[np.loadtxt(mat/k,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float) for k in ('mesh.node','current.node')]
    rows=np.loadtxt(mat/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');g=rows[:,:4]-1
    z=np.load(job/'registered-boundary/registered-boundary.npz');ca=np.load(job/'deep-fat-cortex/registered-cortical-facets.npz');pins=np.zeros(len(a),bool)
    for role in ('O','I','D'):pins[z[role+'_node_joint']]=True
    pins[ca['cell_side_material_bank_nodes'][:,-3:].ravel()]=True
    anc=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    pins[anc['first_contact_new']+[t[2] for t in anc['anchors']]]=True
    st=np.empty((len(g),3))
    for i in range(0,len(g),20000):
        v=g[i:i+20000];st[i:i+len(v)]=np.linalg.svd(np.linalg.solve(a[v[:,1:]]-a[v[:,:1]],b[v[:,1:]]-b[v[:,:1]]),compute_uv=False)
    bad=(st[:,0]>2)|(st[:,-1]<.5)
    out.mkdir();result={}
    for m in [15,16]:
        ids=np.flatnonzero(bad&(rows[:,4]==m))
        w=np.maximum(st[ids,0],1./st[ids,2]);order=np.argsort(-w)
        result[m]=dict(severe_cells=len(ids),isolated_interior_dofs=np.bincount((~pins[g[ids]]).sum(1),minlength=5).tolist(),largest=[dict(local_cell=int(ids[k]),stretch=st[ids[k]].tolist(),center_m=b[g[ids[k]]].mean(0).tolist(),free_nodes=int((~pins[g[ids[k]]]).sum())) for k in order[:6]])
    free=np.unique(g[bad]);free=free[~pins[free]]
    neighborhood=np.flatnonzero(np.isin(g,free).any(1))
    np.savez(out/'interior-patch.npz',free=free,tetrahedra=neighborhood,severe=np.flatnonzero(bad))
    (out/'map.json').write_text(json.dumps(dict(layers=result,free_nodes=len(free),patch_cells=len(neighborhood),source=digest(mat/'current.node')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
