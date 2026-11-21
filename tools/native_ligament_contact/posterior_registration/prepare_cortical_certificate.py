"""Map every cortical triangle to the actual native reverse-query vertices, per muscle bank."""
from pathlib import Path
import json,sys
import numpy as np
from material_context import sha
def prepare(base,out):
    out.mkdir(exist_ok=False)
    spec=json.loads((base/'manifest.json').read_text());blocks=[]
    g=spec['closed_reverse'];first=g['first'];count=g['unique_vertices']
    points=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4)[first:first+count]
    triangles=np.fromfile(base/'cortical-triangles.bin','f4').reshape(-1,10,4)
    ids=points[:,3].copy().view('u4');banks=points[:,4].copy().view('u4');reports=[]
    for side,donor in enumerate((131,145)):
        for body in (1,donor,128):
            ix=np.flatnonzero((ids[:,1]==donor)&(ids[:,2]==body))
            if len(ix)==0:raise ValueError(('empty registered bank',donor,body))
            coords=points[ix,2,:3];lookup={tuple(v):int(i) for v,i in zip(coords,ix)}
            ranges=np.unique(banks[ix,:2],axis=0)
            if len(ranges)!=1 or len(lookup)!=len(ix):raise ValueError(('ambiguous cortical mapping',donor,body))
            start,n=map(int,ranges[0]);t=triangles[start:start+n,:3,:3]
            data=np.zeros((n,4),'u4');data[:,3]=side
            data[:,:3]=[[lookup[tuple(v)] for v in tri] for tri in t]
            if not np.array_equal(points[data[:,:3],2,:3],t):raise ValueError('source face identity lost')
            reports.append(dict(muscle=29+side,cortical_body=body,first=sum(len(x) for x in blocks),count=n))
            blocks.append(data)
    for p in base.iterdir():
        if p.name!='manifest.json':(out/p.name).hardlink_to(p)
    dest=out/'cortical-certificate-faces.bin';np.concatenate(blocks).tofile(dest)
    spec['files'][dest.name]=sha(dest);spec['cortical_certificate']={'banks':reports,'parent':sha(base/'manifest.json')}
    (out/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
