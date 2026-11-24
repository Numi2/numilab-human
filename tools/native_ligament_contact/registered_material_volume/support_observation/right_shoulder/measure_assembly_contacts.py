"""Compare occupied shoulder volumes in a matched loaded body, including registered skin and fat."""
import itertools,json,sys
from pathlib import Path
import numpy as np
from prepare_material_interface import solid
from compile_connected_joint import digest
from prepare_shoulder import read,MOVING
from prepare_winding_material import cgal
import igl


def surface(path):
    with path.open() as s:
        n=int(s.readline().split()[0])
        v=np.array([[float(k)for k in s.readline().split()[1:4]]for _ in range(n)])
        m=int(s.readline().split()[0]);faces=[]
        for i in range(m):
            s.readline();faces.append([int(k)-1 for k in s.readline().split()[1:4]])
    return v,np.asarray(faces)


def canonical(x,f):
    # Released native surface banks can be topologically closed yet geometrically folded.
    # Normalize their positive winding before using a solid Boolean for measurement.
    u,inv=np.unique(x,axis=0,return_inverse=True);f=inv[f]
    g=np.sort(f,axis=1)
    e,n=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    if np.any(n!=2):return None,{'open_or_nonmanifold_edges':int((n!=2).sum())}
    sign=np.einsum('ij,ij->',u[f[:,0]],np.cross(u[f[:,1]],u[f[:,2]]))/6
    if sign<0:f=np.ascontiguousarray(f[:,::-1])
    issues=cgal.remesh_self_intersections(u,f,detect_only=True)[2]
    if len(issues):u,f,_=cgal.mesh_boolean(u,f,type_str='union')
    s,_,meta=solid(u,f,'material')
    if s.is_empty() or str(s.status())!='Error.NoError':return None,{'invalid_solid':str(s.status())}
    return s,dict(self_pairs=len(issues),normalized=bool(len(issues)),minxyz=u.min(0).tolist(),maxxyz=u.max(0).tolist())


def cross(reg,fix,out):
    org=np.array(read(reg/'registration.json')['origin'])
    candidates={};meta={};pairs=[];unresolved=[]
    for i in MOVING:
        before=surface(reg/f'mesh-{i}/source.poly')
        p=np.load(fix/f'bank-{i}.npz');after=((p['native']-org)*1000,p['f'])
        box=np.vstack([before[0],after[0]])
        objects=[]
        for geom in (before,after):
            s,r=canonical(*geom)
            if s is None:raise ValueError(f'Unknown subject solid {i} {r}')
            objects.append(s);meta.setdefault(i,[]).append(r)
        candidates[i]=(objects,box.min(0),box.max(0))
    def measure(i,n,bank,s,other,limits):
        A,lo,hi=candidates[i]
        if np.any(hi<limits[0]) or np.any(lo>limits[1]):return
        vols=[]
        for a,b in zip(A,s):
            result=a^b
            if str(result.status())!='Error.NoError':raise ValueError('Invalid material overlap')
            vols.append(abs(float(result.volume())))
        pairs.append(dict(muscle=i,kind=n,bank=bank,before_mm3=vols[0],after_mm3=vols[1]))
    for i,b in itertools.combinations(MOVING,2):
        other,lo,hi=candidates[b];measure(i,51005,b,other,'dynamic',(lo,hi))
    neigh=np.load(reg/'neighbors.npz')
    for name in neigh.files:
        if not name.endswith('-v'):continue
        k,b,_=name.split('-');x=neigh[name];f=neigh[name[:-1]+'f'];lo,hi=x.min(0),x.max(0)
        ids=[i for i,(q,l,h) in candidates.items() if not(np.any(h<lo)or np.any(l>hi))]
        if not ids:continue
        s,r=canonical(x,f)
        if s is None:unresolved.append(dict(kind=int(k),bank=int(b),affected=list(ids),detail=r));continue
        for i in ids:measure(i,int(k),int(b),[s,s],'fixed',(lo,hi))
    res=dict(reference=digest(reg/'registration.json'),candidate=digest(fix/'results.json'),muscles=meta,neighbors=digest(reg/'neighbors.npz'),indeterminate=unresolved,pairs=pairs)
    out.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    for i in MOVING:
        q=sorted((r for r in pairs if r['muscle']==i),key=lambda r:-r['after_mm3'])
        print(i,len(q),[(r['kind'],r['bank'],round(r['before_mm3']),round(r['after_mm3']))for r in q[:5]])
    print('indeterminate',len(unresolved),'scanned pairs',len(pairs),flush=True)
if __name__=='__main__':cross(*(Path(a).resolve() for a in sys.argv[1:]))
