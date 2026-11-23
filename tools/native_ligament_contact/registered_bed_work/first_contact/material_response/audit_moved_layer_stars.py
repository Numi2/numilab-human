"""Collect all exact current intersections touching a moved triangle and all known original witnesses."""
import sys,json,ctypes
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compile_connected_joint import digest
def test(x,groups,nodes,library):
    x=np.ascontiguousarray(x,'f8');f=np.ascontiguousarray(np.vstack(groups),'u4');start=np.r_[0,np.cumsum(list(map(len,groups)))]
    changed=np.flatnonzero(np.isin(f,nodes).any(1));t=x[f];lo=t.min(1);hi=t.max(1);cent=(lo+hi)*.5;rad=np.max((hi-lo)*.5,axis=1);del t
    bins=np.frexp(rad)[1];candidates=[]
    for k in np.unique(bins):
        ids=np.flatnonzero(bins==k);tr=cKDTree(cent[ids]);hood=tr.query_ball_point(cent[changed],(rad[changed]+rad[ids].max())*(1+5e-14)+1e-15,p=np.inf)
        for i,neighbors in zip(changed,hood):
            near=ids[neighbors];near=near[(near!=i)&np.all(hi[near]>=lo[i],axis=1)&np.all(lo[near]<=hi[i],axis=1)]
            if len(near):candidates.append(np.sort(np.column_stack((np.full(len(near),i),near)),axis=1))
    pairs=np.ascontiguousarray(np.unique(np.vstack(candidates),axis=0),'u4');arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(library)).batch;fn.argtypes=[arg,arg,arg,ctypes.c_uint,arg];fn.restype=None
    hit=np.zeros(len(pairs),'u1');fn(x,f,pairs,len(pairs),hit);witnesses={}
    for a,b in pairs[hit>0]:
        ai,bi=np.searchsorted(start,[a,b],side='right')-1;witnesses.setdefault('OID'[ai]+'OID'[bi],[]).append([int(a-start[ai]),int(b-start[bi])])
    return dict(changed_faces=len(changed),pairs=len(pairs),witnesses=witnesses)
def run(job,h,lib,field,out):
    s=np.load(field/'candidate.npz');x=np.loadtxt(h/'closed-native/native/mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    z=np.load(job/'registered-boundary/registered-boundary.npz');groups=[z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'OID'];old=np.array(json.loads((job/'connected-layer-crossings/reference.json').read_bytes())['witnesses']['2-2'])
    n=np.unique(np.r_[s['nodes'],groups[2][old.ravel()].ravel()]);x[s['nodes']]=s['positions'];r=test(x,groups,n,lib);r['candidate']=digest(field/'candidate.npz');out.write_text(json.dumps(r,indent=2)+'\n');print({k:len(v) for k,v in r['witnesses'].items()})
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
