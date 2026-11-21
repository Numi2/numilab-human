"""Check every material boundary sample left outside the reference broad phase at an accepted native pose."""
import json,sys,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from material_context import sha
from fit_contact_domain import points
from cortical_distance import CorticalSurface
from qualify_material_motion import local,frame

def audit(part,prev,assets,snapshot,lib,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    def items(key):
        return {int(r['index']):np.array([float(r[k]) for k in 'xyzw']) for r in rows if r['field']==key}
    pose=items('relative_position');rot={b:Rotation.from_quat(q) for b,q in items('orientation').items()}
    mat=items('material_node');cfg=json.loads((assets/'manifest.json').read_text());width=cfg['contact_pool']['reference_band_m']
    faces=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    fixed=np.fromfile(assets/'cortical-samples.bin','f4').reshape(2,256,5,4)
    ref={r['body_index']:frame(r) for r in json.loads((prev/'native-release-motion/step-8.receipt.json').read_text())['initial_anatomical_registration']['body_poses']}
    counts={};start=0
    for side,sid in enumerate((29,30)):
        nodes,w,*_=points(part,prev,sid);p=np.load(part/'projected'/f'{sid}.npz');x=np.einsum('ni,nij->nj',w,p['vertices'][nodes])
        curr=np.stack([mat[start+i][:3] for i in range(len(p['vertices']))]);start+=len(curr);now=np.einsum('ni,nij->nj',w,curr[nodes])
        banks={}
        for row in fixed[side,:cfg['sides'][side]['rows']]:
            b=int(row[3].copy().view('u4')[2]);lo,n=map(int,row[4].copy().view('u4')[:2]);banks[(b,lo,n)]=True
        for b,lo,n in banks:
            mesh=faces[lo:lo+n].reshape(-1,3);surf=CorticalSurface(lib,mesh,np.arange(len(mesh)).reshape(-1,3))
            initial=local(x,ref[b]);end=rot[b].inv().apply(rot[128].apply(now)-pose[b][:3])
            source=surf.query(initial)[0];omit=source>width;distance=surf.query(end[omit])[0];lower=source[omit]-np.linalg.norm(end[omit]-initial[omit],axis=1)
            counts[f'{sid}/{b}']=dict(omitted=len(distance),minimum_signed_distance_m=float(distance.min()),minimum_displacement_bound_m=float(lower.min()),penetrating=int((distance<0).sum()))
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),snapshot_sha256=sha(snapshot),pairs=counts),indent=2)+'\n')
if __name__=='__main__':audit(*map(Path,sys.argv[1:]))
