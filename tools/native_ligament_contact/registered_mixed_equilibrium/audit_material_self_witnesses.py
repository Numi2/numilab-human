"""Record exact nonadjacent material faces and connected components at a native endpoint."""
import json,sys
from pathlib import Path
import numpy as np
from audit_finite_cortical_crossings import load_vertices
from count_accepted_cortical_crossings import dyadic,plane,straddles
from pelvic_closed_solid import prepare
from compare_hand_interfaces import query
from material_context import sha
assets,capture,predicate,out=map(Path,sys.argv[1:])
g=json.loads((assets/'manifest.json').read_text())['closed_reverse']
step,v=load_vertices(capture,{'material_vertex':sum(s['vertices'] for s in g['sides']),'cortical_vertex':g['unique_vertices']})
faces=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)[:,0,:3]
by_side={}
for side,s in enumerate(g['sides']):
    i,n,j,m=[s[k] for k in ('first_vertex','vertices','first_face','faces')]
    points,tri,labels,status=prepare(v['material_vertex'][i:i+n],faces[j:j+m].astype(int)-i,predicate)
    xyz=dyadic(points);hits=[]
    for a,b in sorted(query(predicate,points,tri,np.ones(m,dtype=bool))):
        if set(tri[a]).intersection(tri[b]):continue
        A=[xyz[z] for z in tri[a]];B=[xyz[z] for z in tri[b]]
        hits.append(dict(faces=[j+a,j+b],components=[int(labels[tri[a,0]]),int(labels[tri[b,0]])],transverse=bool(straddles(B,A[0],plane(*A)) and straddles(A,B[0],plane(*B)))))
    assert len(hits)==status['nonadjacent_self_intersections']
    by_side[str(29+side)]=dict(topology=status,witnesses=hits)
out.write_text(json.dumps(dict(step=step,source=sha(capture),predicate=sha(predicate),material=by_side),indent=2)+'\n')
