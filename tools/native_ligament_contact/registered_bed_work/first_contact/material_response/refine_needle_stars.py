"""Insert one conforming ring around each long-pointed skin/fat cavity, with explicit parent identities."""
from pathlib import Path
import sys,json
import numpy as np
from verify_carpal_mesh import classify
from fit_carpal_clearance import geom
from compile_connected_joint import digest


def run(mesh,output,fraction):
    z=np.load(mesh);x=z['x'].astype(float);t=z['t'].copy();r=z['role'].copy()
    keep=z['markers']!=1000;f=z['facets'][keep].copy();mk=z['markers'][keep].copy();locked=z['locked'].copy()
    v,q=geom(x[t]);bad=np.flatnonzero((v<=1e-18)|(q<1e-5));parent=np.arange(len(t))
    apices={}
    for row in t[bad]:
        # A needle has one distant vertex and a three-vertex cluster.
        d=np.linalg.norm(x[row,None,:]-x[row][None,:,:],axis=2)
        p=int(row[np.argmax(d.sum(1))]);rest=[n for n in row if n!=p]
        length=np.linalg.norm(x[rest]-x[p],axis=1).min()
        diameter=np.linalg.norm(x[rest,None,:]-x[rest][None,:,:],axis=2).max()
        if length<5*diameter:continue
        apices.setdefault(p,set()).update(rest)
    ops=[]
    for p,cluster in apices.items():
        incident=(t==p).any(1)&np.isin(t,list(cluster)).any(1)
        sites=np.unique(t[incident]);sites=sites[sites!=p]
        for j in sorted(sites,key=lambda j:(-float(np.linalg.norm(x[j]-x[p])),int(j))):
            cells=np.flatnonzero((t==p).any(1)&(t==j).any(1));face=np.flatnonzero((f==p).any(1)&(f==j).any(1))
            if not len(cells) or np.any(mk[face]!=1516):continue
            pos=(x[p]+fraction*(x[j]-x[p])).astype('f4').astype(float);k=len(x)
            a=t[cells].copy();b=a.copy();a[a==p]=k;b[b==j]=k
            trial=np.vstack((x,pos));vv,_=geom(trial[np.vstack((a,b))])
            if vv.min()<=1e-18:continue
            t[cells]=a;t=np.vstack((t,b));r=np.r_[r,r[cells]];parent=np.r_[parent,parent[cells]]
            if len(face):
                a=f[face].copy();b=a.copy();a[a==p]=k;b[b==j]=k
                f[face]=a;f=np.vstack((f,b));mk=np.r_[mk,mk[face]]
            x=trial;locked=np.r_[locked,False];ops.append((int(p),int(j),k))
    rr,outer=classify(x,t,f,mk)
    if not np.array_equal(r,rr):raise ValueError('Regional material partition changed')
    v,q=geom(x[t]);out=output;out.mkdir()
    np.savez_compressed(out/'domain.npz',x=x.astype('f4'),t=t,role=r,locked=locked,facets=f,markers=mk,outer=outer,fence=z['fence'])
    np.savez_compressed(out/'subdivision.npz',parent=parent,nodes=np.asarray(ops))
    print(len(ops),len(bad),int((q<1e-5).sum()),v.min(),q.min())
    (out/'result.json').write_text(json.dumps(dict(source=digest(mesh),mesh=digest(out/'domain.npz'),parents=digest(out/'subdivision.npz'),fraction=fraction,apices=list(apices),original_vertices=len(z['x']),after=int((q<1e-5).sum())),indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),float(sys.argv[3]))
