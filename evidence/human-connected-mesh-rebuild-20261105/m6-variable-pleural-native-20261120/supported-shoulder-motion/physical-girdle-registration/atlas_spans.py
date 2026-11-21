"""Test whether the corresponding straight chords are even available in the original common source atlas."""
import json,sys
from pathlib import Path
import numpy as np
from compare_registration import prepare
from shoulder_inputs import old,unit
def run(launch,ref,motion,archive,sites,out):
    *_,data=prepare(launch,ref,motion,archive)
    result=[]
    for rec in json.loads(sites.read_text())['regions']:
        side=rec['side']; n=rec['name']
        a,b=((10,8 if n.startswith('coracoclavicular') else 12) if side else (11,9 if n.startswith('coracoclavicular') else 13))
        pts=[];ns=[];ids=[];ds=[];indices=[]
        for k,center,r in zip((a,b),rec['centers_atlas_m'],rec['bounds_m']):
            v,f=data[k][2:4];t=v[f];p=t.mean(1);d=np.linalg.norm(p-center,axis=1);ix=np.flatnonzero(d<=r)
            pts.append(p[ix]);ns.append(unit(np.cross(t[ix,1]-t[ix,0],t[ix,2]-t[ix,0])));ds.append(d[ix]);indices.append(ix)
        dv=pts[1][None]-pts[0][:,None];q=(np.einsum('ijk,ik->ij',dv,ns[0])>.00005)&(np.einsum('ijk,jk->ij',-dv,ns[1])>.00005)
        ii,jj=np.where(q);rank=np.argsort(ds[0][ii]**2+ds[1][jj]**2)
        best=None
        for trial in rank[:500]:
            i=ii[trial];j=jj[trial];x,y=pts[0][i],pts[1][j]
            if not any(old.audit.lp.vector_hits(x,y,(data[k][2],data[k][3])) for k in (a,b)):
                best=dict(faces=[int(indices[0][i]),int(indices[1][j])],length_m=float(np.linalg.norm(y-x)));break
        result.append(dict(name=n,side=side,pairs=q.size,outward_pairs=len(rank),best=best))
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
