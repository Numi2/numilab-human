"""Find which mesh coordinates can condition the skin/fat failures without changing anatomical walls."""
import json,sys
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from build_bed_audit import sha

def inspect(v,r,out):
    bad=json.loads(r.read_bytes())['metrics'];roles=sorted(bad);selected=np.array([q['row'] for k in roles for q in bad[k]['cells'] if q['quality'][2]<1e-5])
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,dtype=np.int32,usecols=(1,2,3,4,5));t=raw[:,:4]-1
    walls=np.load(v/'material-walls.npz');X=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3))
    ct=t[selected];nodes,inv=np.unique(ct,return_inverse=True)
    inc=coo_matrix((np.ones(ct.size),(np.repeat(np.arange(len(ct)),4),inv.ravel())),shape=(len(ct),len(nodes))).tocsr()
    n,group=connected_components(inc@inc.T,directed=False);surfaces={k:np.unique(walls[k]) for k in 'OIDC'}
    comp=[]
    for j in range(n):
        ids=selected[group==j];nn=np.unique(t[ids]);mask={k:np.isin(nn,rows) for k,rows in surfaces.items()};inter=nn[~np.logical_or.reduce(list(mask.values()))]
        comp.append(dict(rows=ids.tolist(),material_labels=np.unique(raw[ids,4]).tolist(),wall_membership={k:int(a.sum()) for k,a in mask.items()},interior_nodes=inter.tolist(),centroid_m=X[nn].mean(0).tolist()))
    out.write_text(json.dumps(dict(mesh=sha(v/'mesh.ele'),native_quality=sha(r),regions=comp),indent=2)+'\n')
if __name__=='__main__':inspect(*(Path(x)for x in sys.argv[1:]))
