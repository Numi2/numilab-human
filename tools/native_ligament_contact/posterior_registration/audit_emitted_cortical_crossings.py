"""Test exact triangle crossings on the Float32 vertices supplied to the native contact queries."""
from pathlib import Path
import csv,json,sys
import numpy as np
from compare_hand_interfaces import query
from count_accepted_cortical_crossings import dyadic,plane,straddles
from material_context import sha

def load_vertices(path,counts):
    state={};step=None
    with path.open() as f:
        for r in csv.DictReader(f):
            label=r['field'];i=int(r['index']);current=int(r['accepted_step'])
            if step is None:step=current
            if current!=step or label not in counts:raise ValueError('Mixed step or unknown field')
            values=state.setdefault(label,{})
            if i in values:raise ValueError('Duplicate native vertex')
            # Nine significant decimal digits round-trip to the exact GPU Float32.
            values[i]=np.array([r[k] for k in ('x','y','z')],dtype='f4')
    if step is None:raise ValueError('Missing accepted surface')
    ans={}
    for name,n in counts.items():
        if len(state[name])!=n:raise ValueError(('Uncaptured surface vertices',name,n))
        ans[name]=np.stack([state[name][i] for i in range(n)]).astype('f8')
        if not np.isfinite(ans[name]).all():raise ValueError('Nonfinite contact vertex')
    return step,ans

def tests(assets,capture,predicate,out):
    spec=json.loads((assets/'manifest.json').read_text());g=spec['closed_reverse'];sides=g['sides']
    step,vert=load_vertices(capture,{'material_vertex':sum(r['vertices'] for r in sides),'cortical_vertex':g['unique_vertices']})
    mf=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)[:,0,:3]
    t=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[g['first']:]
    domain=pool[:,3].copy().view('u4');bank=pool[:,4].copy().view('u4');result={}
    for side,(surface,donor) in enumerate(zip(sides,(131,145))):
        i=surface['first_vertex'];n=surface['vertices'];f0=surface['first_face']
        V=vert['material_vertex'][i:i+n];F=mf[f0:f0+surface['faces']].astype(int)-i
        assert F.min()>=0 and F.max()<n
        for body in (1,donor,128):
            ix=np.flatnonzero((domain[:,1]==donor)&(domain[:,2]==body))
            banks=np.unique(bank[ix,:2],axis=0)
            if len(banks)!=1:raise ValueError('Missing unique cortical bank')
            a,num=banks[0];c0=np.ascontiguousarray(pool[ix,2,:3])
            local={tuple(q):j for j,q in enumerate(c0)}
            if len(local)!=len(ix):raise ValueError('Duplicate cortical source vertex')
            C=np.array([[local[tuple(q)] for q in tri] for tri in t[a:a+num]],dtype=int)
            B=vert['cortical_vertex'][ix];vertices=np.vstack((V,B));faces=np.vstack((F,C+n))
            changed=np.arange(len(faces))<len(F)
            pairs=sorted((j,k-len(F)) for j,k in query(predicate,vertices,faces,changed) if j<len(F)<=k)
            xyz=dyadic(vertices);trans=[];touch=0
            for j,k in pairs:
                A=[xyz[z] for z in F[j]];D=[xyz[n+z] for z in C[k]]
                if straddles(D,A[0],plane(*A)) and straddles(A,D[0],plane(*D)):trans.append([j+f0,k])
                else:touch+=1
            result[f'{29+side}/{body}']={'intersecting_pairs':len(pairs),'transverse_pairs':len(trans),'boundary_only_pairs':touch,'native_face_witnesses':trans[:8]}
    out.write_text(json.dumps({'accepted_step':step,'native_surface_sha256':sha(capture),'assets_sha256':sha(assets/'manifest.json'),'predicate_sha256':sha(predicate),'regional':result,'total_transverse':sum(r['transverse_pairs'] for r in result.values()),'total_boundary_only':sum(r['boundary_only_pairs'] for r in result.values())},indent=2)+'\n')
if __name__=='__main__':tests(*map(Path,sys.argv[1:]))
