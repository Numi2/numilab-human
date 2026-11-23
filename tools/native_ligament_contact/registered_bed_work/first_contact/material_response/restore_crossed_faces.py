"""Restore only witnessed facets to their previous cortical-clear position, preserving tetrahedral correspondence."""
import sys,json
from pathlib import Path
import numpy as np
import manifold3d as md
from build_carpal_cavity import boundary,parts,shell
from audit_registered_skeleton import digest
from fit_carpal_clearance import geom

def run(source,clear,bones,out):
    z=np.load(source);x=z['x'].astype(float);old=np.load(clear)['x'].astype(float);t=z['t'];roles=z['role'];w=np.load(bones)
    c=x.mean(0);body={}
    for sid in (8,9,10,11,12,13):
        y,inv=np.unique(w['cortex'+str(sid)],axis=0,return_inverse=True);f=inv[w['faces'+str(sid)]];pts=(y.astype(float)-c)*1000
        if np.einsum('ij,ij->',pts[f[:,0]],np.cross(pts[f[:,1]],pts[f[:,2]]))<0:f=f[:,[0,2,1]]
        body[sid]=shell(pts,f,md.Manifold.reserve_ids(1))
    fixed=set();log=[]
    for it in range(12):
        add=set();counts={}
        for mat in (15,16):
            faces=boundary(t[roles==mat]);mid=md.Manifold.reserve_ids(1);m=shell((x-c)*1000,faces,mid)
            for sid,b in body.items():
                inter=m^b;vol=inter.volume()
                if vol<=0:continue
                pts,tri,owner,face=parts(inter);rows=np.unique(face[owner==mid]);nodes=np.unique(faces[rows])
                add.update(int(i) for i in nodes if np.any(old[i]!=x[i]));counts[str((mat,sid))]=[float(vol),len(rows)]
        log.append(counts)
        if not counts:break
        if not add:raise ValueError('Crossed unchanged material surface')
        ids=sorted(add);x[ids]=old[ids];fixed.update(add)
    v,q=geom(x[t]);res={key:z[key] for key in z.files};res['x']=x.astype('f4')
    out.mkdir();np.savez_compressed(out/'domain.npz',**res);np.save(out/'fixed.npy',sorted(fixed))
    (out/'result.json').write_text(json.dumps(dict(parent=digest(source),clear=digest(clear),fixed=sorted(fixed),passes=log,low=int(((q<1e-5)|(v<=1e-18)).sum()),inverted=int((v<=0).sum())),indent=2)+'\n')
    print('restored',len(fixed),'low',int(((q<1e-5)|(v<=1e-18)).sum()),'inverted',int((v<=0).sum()))
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
