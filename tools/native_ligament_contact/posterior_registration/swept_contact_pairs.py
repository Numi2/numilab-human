"""Spatially enumerate all cortical/material face pairs entering an observed pose envelope."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from audit_finite_cortical_crossings import load_vertices
from material_context import sha

def gather(base,captures,extra_m=0.):
    spec=json.loads((base/'manifest.json').read_text())
    sides=spec['closed_reverse']['sides']
    mf=np.fromfile(base/'closed-material-triangles.bin','u4').reshape(-1,3,4)[:,0,:3]
    cf=np.fromfile(base/'cortical-certificate-faces.bin','u4').reshape(-1,4)
    counts={'material_vertex':sum(r['vertices'] for r in sides),'cortical_vertex':spec['closed_reverse']['unique_vertices']}
    boxes={}
    for c in captures:
        _,x=load_vertices(c,counts)
        for name,f in [('material_vertex',mf),('cortical_vertex',cf[:,:3])]:
            bounds=x[name][f]
            lo=bounds.min(axis=1);hi=bounds.max(axis=1)
            if name in boxes:
                lo=np.minimum(lo,boxes[name][0]);hi=np.maximum(hi,boxes[name][1])
            boxes[name]=(lo,hi)
    old=np.fromfile(base/'shared-finite-faces.bin','u4').reshape(-1,4)
    known={(int(v[3]),tuple(sorted(map(int,v[:3])))) for v in old}
    output=[];by=[]
    for side,d in enumerate(sides):
        ml,mh=boxes['material_vertex'];s=d['first_face'];e=s+d['faces'];ml,mh=ml[s:e],mh[s:e]
        bl,bh=boxes['cortical_vertex']
        for bank in (v for v in spec['cortical_certificate']['banks'] if v['muscle']==29+side):
            b=bank['first'];be=b+bank['count'];lo,hi=bl[b:be],bh[b:be];center=.5*(lo+hi)
            radius=np.linalg.norm(.5*(hi-lo),axis=1)
            near=cKDTree(center).query_ball_point(.5*(ml+mh),np.linalg.norm(.5*(mh-ml),axis=1)+radius.max()+np.sqrt(3)*extra_m)
            start=len(output)
            for j,idx in enumerate(near):
                ids=np.asarray(idx,dtype=int)
                where=np.all((hi[ids]+extra_m>=ml[j])&(mh[j]+extra_m>=lo[ids]),axis=1)
                row=s+j;k=tuple(sorted(map(int,mf[row])))
                output.extend((int(g),row) for g in ids[where]+b if (int(g),k) not in known)
            by.append(dict(muscle=29+side,cortex=bank['cortical_body'],added=len(output)-start))
    return np.asarray(sorted(output),dtype='u4').reshape(-1,2),{'capture_sha256':[sha(x) for x in captures],'margin_m':extra_m,'banks':by}

if __name__=='__main__':
    base,output=map(Path,sys.argv[1:3]);rows,meta=gather(base,list(map(Path,sys.argv[3:])))
    np.save(output,rows);output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(len(rows),meta['banks'])
