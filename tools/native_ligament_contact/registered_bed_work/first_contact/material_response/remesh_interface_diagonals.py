"""Explore coupled interface diagonal flips only where a tetrahedron separates two triangulations of one near-planar quadrilateral."""
import sys,json
from pathlib import Path
from itertools import combinations
import numpy as np
from build_carpal_cavity import FACES
from fit_carpal_clearance import geom
from verify_carpal_mesh import classify
from audit_registered_skeleton import digest

def run(source,out):
    z=np.load(source);x=z['x'].astype(float);t=z['t'];r=z['role'].copy();keep=z['markers']!=1000;f=z['facets'][keep];mk=z['markers'][keep]
    v,q=geom(x[t]);bad=(q<1e-5)|(v<=1e-18)
    maps={tuple(sorted(row)):j for j,row in enumerate(f)}
    links={}
    for i,row in enumerate(t):
        for face in row[FACES]:links.setdefault(tuple(sorted(face)),[]).append(i)
    ops=[];selected=set()
    for i in np.flatnonzero(bad)[np.argsort(q[bad])]:
        faces=t[i][FACES];keys=[tuple(sorted(row)) for row in faces];labels=[mk[maps[k]] if k in maps else 0 for k in keys]
        before=np.flatnonzero(np.array(labels)==1516);after=np.flatnonzero(np.array(labels)==0)
        if len(before)!=2 or len(after)!=2 or any(len(links[k])!=2 for k in keys) or any(k in selected for k in keys):continue
        corners=np.unique(faces[before])
        if len(corners)!=4 or np.any(z['locked'][corners]):continue
        a,b=faces[before];j,k=list(set(a)&set(b));c,d=list(set(corners)-{j,k})
        # Maximum of the piecewise-linear patch difference occurs between the two diagonals.
        A=np.column_stack((x[k]-x[j],x[c]-x[d]));st=np.linalg.lstsq(A,x[c]-x[j],rcond=None)[0]
        if np.min(st)<0 or np.max(st)>1:continue
        departure=float(np.linalg.norm(x[j]+(x[k]-x[j])*st[0]-(x[c]+(x[d]-x[c])*st[1])))
        if departure>85e-6:continue
        old=int(r[i]);r[i]=31-old;selected.update(keys)
        ops.append(dict(cell=int(i),original=old,deviation_um=departure*1e6,material_transfer_mm3=float(v[i]*1e9),removed=[maps[keys[k]] for k in before],added=faces[after].tolist()))
    if not ops:raise ValueError('No source-compatible shared interface diagonal')
    dead=np.array([k for op in ops for k in op['removed']]);ff=np.vstack((np.delete(f,dead,axis=0),*[op['added'] for op in ops]));mm=np.r_[np.delete(mk,dead),np.full(2*len(ops),1516)]
    classified,ext=classify(x,t,ff,mm)
    if not np.array_equal(classified,r):raise ValueError('Interface surgery changed a disconnected region')
    out.mkdir();res={k:z[k] for k in z.files if k not in ('facets','markers','role','outer')};np.savez_compressed(out/'domain.npz',**res,facets=ff,markers=mm,role=r,outer=ext)
    (out/'result.json').write_text(json.dumps(dict(source=digest(source),operations=ops),indent=2)+'\n');print(ops)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
