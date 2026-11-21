"""Compare registered atlas and physical-source shoulder surfaces in the same accepted poses."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from shoulder_inputs import load,digest
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32

def inputs(launch,anchors,motion):
    cfg,c,m,reg,poses=load(launch,anchors,motion)
    doc=json.loads(Path(cfg['argv'][3]).with_suffix('.manifest.json').read_text())
    f=doc['source']['runtime_reference']['manifest']; path=Path(f['file'])
    assert digest(path)==f['sha256']
    default={r['core_body_index']:np.array(r['default_com_position_world_m']+r['default_inertial_quaternion_world_xyzw']) for r in json.loads(path.read_text())['core_tree']['source_body_records']}
    names={}
    for row in doc['source']['anchors']:
        b=row['core_body_index']
        if row['myosim_body'] in {f'{name}_{side}' for side in 'lr' for name in ('scapula','clavicle','humerus','radius','ulna')}:
            matches=[sid for sid,item in c.items() if item['body']==b and item['vertexCount']==row['vertex_count'] and item['indexCount']==3*row['triangle_count']]
            assert len(matches)==1
            names[row['myosim_body']]=(matches[0],b)
    assert len(names)==10
    return m,poses,default,names

def world(p,x):
    return p[:3]+Rotation.from_quat(p[3:]).apply(x)

def audit(launch,anchors,motion,source,out):
    meshes,poses,default,owners=inputs(launch,anchors,motion)
    bones=np.load(source);results=[]
    install_for_float32()
    for side in 'lr':
        names=[s+'_'+side for s in ('scapula','clavicle','humerus','radius','ulna')]
        for name in names:
            _,b=owners[name]
            ref=world(default[b],bones[name+'_local'])
            assert np.max(np.abs(ref-bones[name+'_world']))<1e-6,name
        for key,p in [(0,default),*((k,poses[k]) for k in [32,256,8000,-3])]:
            for i,j in ((0,2),(0,1),(2,3),(2,4)):
                ids=[owners[names[l]] for l in (i,j)]; rows=[]
                for reference in ('physical_source','registered_atlas'):
                    pnts=[];tris=[]
                    for name,(sid,b) in zip((names[i],names[j]),ids):
                        v,f=bones[name+'_local'],bones[name+'_faces']
                        if reference=='registered_atlas':v,f,*_=meshes[sid]
                        pnts.append(world(p[b],v));tris.append(f)
                    pairs=intersections(pnts[0],tris[0],pnts[1],tris[1])
                    rows.append(len(pairs))
                results.append(dict(side=side,step=key,banks=[x[0] for x in ids],physical_source=rows[0],registered_atlas=rows[1]))
    out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': audit(*map(Path,sys.argv[1:]))
