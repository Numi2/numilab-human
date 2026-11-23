"""Recover the wrist cortical contact wall from its original marked volume and current rigid owner."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from classify_connected_boundary import pack
from rebind_ligament_material import banks
from audit_registered_skeleton import closed_solid
from build_carpal_cavity import boundary
from audit_fiber_load_bounds import read_pose
from inspect_shoulder_source import nearest
from partition_carpal_material import solid

def run(h,current,full,carpal,lib,out):
    orig=h/'closed-native/source';new=h/'closed-native/native'
    d=json.loads((orig/'receipt.json').read_text())
    if d['domain_sha256']!=digest(carpal):raise ValueError('Stale carpal domain')
    local=np.load(carpal);w=np.load(orig/'material-source-indices.npz');s=np.load(new/'material-source-indices.npz')
    old_t=np.loadtxt(orig/'mesh.ele',skiprows=int(w['child_begin'])+1,usecols=(1,2,3,4,5),dtype='i4')
    if not np.array_equal(local['role'],old_t[:,4]):raise ValueError('Wrist material identity mismatch')
    wp=np.full(len(local['x']),-1,'i4');wp[local['t'].ravel()]=old_t[:,:4].ravel()-1
    if not np.array_equal(wp[local['t']],old_t[:,:4]-1):raise ValueError('Unregistered wrist topology')
    locrows=np.flatnonzero(np.isin(local['markers'],[115,116]));F=s['original_to_emitted'][wp[local['facets'][locrows]]]
    if np.any(F<0):raise ValueError('Shoulder surgery deleted wrist wall nodes')
    proven=np.load(full/'faces.npz');ki=pack(proven['face']);k=pack(F);j=np.searchsorted(ki,k)
    if not np.array_equal(ki[j],k) or np.any(proven['inherited_kind'][j]!=-1):raise ValueError('Wrist surface ownership changed')
    xyz=np.loadtxt(new/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    xyz0=local['x'][local['facets'][locrows]].astype('f4')
    if not np.array_equal(xyz[F],xyz0):raise ValueError('Material drift at scaphoid wall')
    path=next((h.parent/'supported-bones').glob('*.nhbones'));bank=banks(path);body,sid=next((b,i) for b,i in bank if i==81)
    pose=read_pose(h.parent/'native-ligament-material/stderr.log')['bodyPoses'][body]
    f,v,_=bank[body,sid];v=Rotation.from_quat(pose[3:]).apply(v)+pose[:3];center=v.mean(0)
    lo,hi,cortex=closed_solid((v-center)*1000,f)
    tet=np.loadtxt(new/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    ids=tet[:,:4]-1;near=np.ones(len(ids),bool)
    for axis in range(3):
        col=xyz[:,axis][ids]
        near&=(col.min(1)<=v[:,axis].max())&(col.max(1)>=v[:,axis].min())
    load={}
    for mat in (15,16):
        keep=near&(tet[:,4]==mat);ff=boundary(ids[keep]);vs,ix=np.unique(ff,return_inverse=True)
        tissue=solid((xyz[vs]-center)*1000,ix.reshape(-1,3));over=cortex^tissue
        if over.status()!=cortex.status():raise ValueError('Invalid cortical intersection operand')
        load[str(mat)]={'cells':int(keep.sum()),'penetration_mm3':abs(over.volume())}
    weights=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    pts=np.einsum('ij,fjd->fid',weights,xyz[F]).reshape(-1,3)
    closest,dist,idx=nearest((v,f),pts,lib)
    out.mkdir();np.savez_compressed(out/'cortical-facets.npz',child=proven['cell'][j],side=proven['side'][j],nodes=F,source_marker_rows=locrows,material=local['markers'][locrows]-100,source_bone=sid,body=body,closest_face=idx.reshape(-1,3),closest_point=closest.reshape(-1,3,3))
    result=dict(material_geometry=digest(new/'mesh.node'),original_domain=digest(carpal),bones=digest(path),capture=digest(h.parent/'native-ligament-material/stderr.log'),cortical_rows=len(j),bank=sid,body=body,layers=load,wall_gap_m={'min':float(np.min(dist)),'max':float(np.max(dist)),'median':float(np.median(dist))},remaining_one_sided_skin_faces=int(np.sum((proven['inherited_kind']<0)&(proven['layer']==0)))-int(np.sum(local['markers'][locrows]==115)))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
