"""Compile the existing material surface in the common pelvic rest frame."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_material_motion import frame
from material_context import sha
def prepare(part,base,receipt,out):
    out.mkdir(exist_ok=False)
    d=json.loads((base/'manifest.json').read_text())
    for source in base.iterdir():
        if source.name!='manifest.json':(out/source.name).hardlink_to(source)
    pose=frame(next(r for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses'] if r['body_index']==128))
    if sha(receipt)!=d['common_material_reference']['pose']:raise ValueError('P1 reference pose mismatch')
    rp=Rotation.from_quat(pose[3:]).inv()
    b=np.fromfile(base/'closed-material-vertices.bin','f4').reshape(-1,2,4);refs=[];sides=[]
    nr=np.fromfile(base/'pelvic-reference-nodes.bin','f4').reshape(-1,3,4)
    z=nr[:,0,:3].astype(float)+nr[:,1,:3].astype(float)
    for m,g in zip((29,30),d['closed_reverse']['sides']):
        src=part/'conforming-boundary'/f'{m}.npz';v=np.load(src)
        xyz=rp.apply(v['vertices'].astype(float)-pose[:3])
        q=b[g['first_vertex']:g['first_vertex']+g['vertices']]
        idx=q[:,1].copy().view('u4');w=q[:,0]
        parent=np.load(part/'projected'/f'{m}.npz')
        side=0 if m==29 else 1;first=d['contact_flexibility'][side]['first_node']
        if not np.array_equal(v['node']+first,idx) or not np.array_equal(v['vertex_shape'].astype('f4'),w):raise ValueError('Native P1 binding mismatch')
        parents=parent['vertices'][v['node']].astype(float)
        exact=rp.apply(np.einsum('ik,ikj->ij',v['vertex_shape'],parents)-pose[:3])
        err=np.linalg.norm(exact-xyz,axis=1).max()
        if err>1.e-10 or not np.isfinite(xyz).all():raise ValueError(('invalid source interpolation',m,err))
        existing=np.einsum('ik,ikj->ij',w.astype(float),z[idx])
        f=np.zeros((len(q),2,4),'f4');f[:,0,:3]=xyz;f[:,1,:3]=xyz-f[:,0,:3].astype(float)
        refs.append(f)
        sides.append(dict(muscle=m,source=sha(src),vertices=len(q),maximum_P1_rest_error_m=float(err),maximum_packed_absolute_weight_error_m=float(np.linalg.norm(existing-xyz,axis=1).max())))
    target=out/'closed-material-reference.bin';np.concatenate(refs).tofile(target)
    d['files'][target.name]=sha(target);d['surface_material_reference']={'source_manifest':sha(base/'manifest.json'),'pose':sha(receipt),'sides':sides}
    (out/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
