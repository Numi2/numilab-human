"""Evaluate current registered cortical solids against the actual skin/fat tetrahedra."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from rebind_ligament_material import banks
from audit_registered_skeleton import closed_solid
from audit_fiber_load_bounds import read_pose
from build_carpal_cavity import boundary
from partition_carpal_material import solid

def run(h,out,capture=None,field=None):
    out.mkdir();vol=h/'closed-native/native';bone=next((h.parent/'supported-bones').glob('*.nhbones'))
    x=np.loadtxt(vol/'mesh.node',skiprows=1,usecols=(1,2,3));cells=np.loadtxt(vol/'mesh.ele',skiprows=1,dtype='i4');tet=cells[:,1:5]-1;mat=cells[:,5]
    if field is not None:
        transport=np.load(field);old=x.copy();x[transport['nodes']]=transport['positions']
        from certify_transported_material import measure
        det,L=measure(x,tet);rest,_=measure(old,tet);ratio=det/rest
        if np.any((det<=6e-18)|(det<L*1e-5)|(ratio<.4)|(ratio>2.5)):raise ValueError('Invalid installed pose')
    native=read_pose(h.parent/'native-ligament-material/stderr.log');p=native['bodyPoses']
    input_capture=h.parent/'native-ligament-material/stderr.log'
    if capture is not None:
        scene=json.loads((capture.parent/'launch.json').read_text());asset=scene['argv'][3]
        if scene['pins'][asset]!=digest(bone):raise ValueError('Native pose and rigid-bank asset differ')
        live=json.loads(capture.read_text())['all_bodies']
        if len(live)!=len(p) or [x['body_index'] for x in live]!=list(range(len(p))):raise ValueError('Incomplete native pose')
        p=[x['current']['position_world_m']+x['current']['orientation_world_xyzw'] for x in live];input_capture=capture
    if native['bonePayloadSha256']!=digest(bone):raise ValueError('Stale source bank')
    lo=np.empty((3,len(tet)),'f4');hi=lo.copy()
    for i in range(3):
        values=x[:,i][tet];lo[i]=values.min(1);hi[i]=values.max(1)
    result={};surfs=banks(bone)
    for (b,s),(f,v,_) in sorted(surfs.items(),key=lambda k:(k[0][0] not in range(41,128),k[0])):
        world=(Rotation.from_quat(p[b][3:]).apply(v)+p[b][:3]).astype('f4').astype(float);origin=world.mean(0)
        try:_,_,cortex=closed_solid((world-origin)*1000,f)
        except ValueError as err:result[s]=dict(body=b,indeterminate=str(err));continue
        near=np.ones(len(tet),bool)
        for i in range(3):near &= (lo[i]<=world[:,i].max())&(hi[i]>=world[:,i].min())
        ids=np.flatnonzero(near);row=dict(body=b)
        for role,k in [('skin',15),('fat',16)]:
            t=tet[ids[mat[ids]==k]]
            if not len(t):row[role]=0.;continue
            ff=boundary(t);verts,inv=np.unique(ff,return_inverse=True)
            try:
                tis=solid((x[verts]-origin)*1000,inv.reshape(-1,3))
                overlap=cortex^tis
                if overlap.status()!=cortex.status():raise ValueError(str(overlap.status()))
                row[role]=max(0.,float(overlap.volume()))
            except ValueError as exc:row[role+'_indeterminate']=str(exc)
        result[s]=row
        if len(result)%20==0:print('banks',len(result),flush=True)
    summary={}
    for name in ('skin','fat'):
        overlap=sorted([(s,r[name]) for s,r in result.items() if r.get(name,0.)>1e-6],key=lambda it:-it[1]);summary[name]=dict(positive_banks=len(overlap),largest_mm3=overlap[:8],unresolved=sum(name not in r for r in result.values()))
    rec=dict(mesh=digest(vol/'mesh.node'),elements=digest(vol/'mesh.ele'),current_displacement=digest(field) if field else None,bone=digest(bone),capture=digest(input_capture),units='mm^3, one value per cortical solid and tissue. Pair volumes can share spatial regions.',results=result,summary=summary)
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
