"""Identify the detailed geometry and registered source coordinates around the shared slivers."""
from pathlib import Path
from itertools import combinations
import json,sys
import numpy as np
from compile_connected_joint import digest

def run(job,volume,physical,out):
    d=json.loads(physical.read_bytes());deep=[s for s in d['cells'] if len(s['layer_nodes']['D'])==4]
    common=set.intersection(*(set(s['nodes'])for s in deep));ids=np.unique([n for s in deep for n in s['nodes']])
    z=job/'registered-boundary/registered-boundary.npz';b=np.load(z);mesh=b['D_node_joint'][b['D_face_nodes']]
    fi=np.flatnonzero(np.isin(mesh,ids).any(1));faces=mesh[fi]
    ref=np.loadtxt(volume/'mesh.node',usecols=(1,2,3),skiprows=1).astype('f4').astype(float)
    sp=ref[faces];edges=np.linalg.norm(np.roll(sp,1,axis=1)-sp,axis=2);length=edges.max(1)
    height=np.linalg.norm(np.cross(sp[:,1]-sp[:,0],sp[:,2]-sp[:,0]),axis=1)/length
    if np.any(height<=0):raise ValueError('Degenerate deep boundary triangle')
    native_d=b['D_node_joint'];retained=len(b['D_retained_source_node']);source=[]
    for v in ids:
        ix=np.flatnonzero(native_d==v)
        if len(ix)!=1:raise ValueError('Ambiguous layer coordinate')
        idx=int(ix[0]);n=idx-retained
        if n<0:
            source.append(dict(node=int(v),retained_source_node=int(b['D_retained_source_node'][idx])))
        else:
            source.append(dict(node=int(v),parent_face=int(b['D_new_node_parent_face'][n]),
                barycentric=b['D_new_node_barycentric'][n].tolist(),residual_m=b['D_new_node_residual_m'][n].tolist()))
    result=dict(registered_boundary_sha256=digest(z),native_witnesses_sha256=digest(physical),shared_nodes=sorted(common),
                nodes=source,boundary_faces=len(faces),min_face_altitude_m=float(height.min()),
                max_edge_over_altitude=float(np.max(length/height)),worst_source_face_row=int(fi[np.argmax(length/height)]),
                local_cell_altitude_range_m=[min(q['smallest_altitude_m'] for q in deep),max(q['smallest_altitude_m'] for q in deep)],
                geometric_source_records=sorted({(q['material'],q['body'],q['region'],q['source_face'])for q in deep}))
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
