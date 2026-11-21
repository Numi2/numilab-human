"""Emit the four changed transforms through the existing registration and NHBONES1 ABI."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from numilab_human.model import _bodyparts_visual_local_pose
from compare_registration import prepare

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def emit(launch,sites,motion,archive,source,change,out):
    cfg,cat,meshes,anchors,poses,default,data=prepare(launch,sites,motion,archive)
    parent=Path(cfg['argv'][3]);mp=parent.with_suffix('.manifest.json');manifest=json.loads(mp.read_text())
    assert sha(parent)==manifest['payload']['sha256']==anchors['bone_payload_sha256']
    assert sha(source)==manifest['source']['registration']['sha256']
    reg=json.loads(source.read_text());payload=bytearray(parent.read_bytes())
    assert struct.unpack_from('<I',payload,24)[0]==int(sha(source)[:8],16)
    changes=json.loads(change.read_text()); assert {v['sid'] for v in changes}=={8,9,10,11} and len(changes)==4
    out.mkdir(exist_ok=True)
    settings={};geometry={};rows=[]
    for rec in changes:
        sid=rec['sid'];mid,b,raw,f,local,*_=data[sid];anchor=reg['anchors'][sid-1]
        assert b==rec['body']==anchor['target']['core_body_index'] and anchor['source']['member_id']==mid
        offset=60+(sid-1)*60;body,first,n,ii,nind,stable=struct.unpack_from('<6I',payload,offset)
        assert stable==sid and body==b and n==len(raw) and nind==f.size and rec['scale']==1
        assert all(r['stable_id']!=sid for r in manifest['body_local_bakes']['rows'])
        owner=anchor['registration'];old=np.array(owner['source_obj_mm_to_core_inertial_body_m']);dr=R.from_quat(rec['rotation_xyzw']);dt=rec['translation_m']
        predicted=raw@old[:3,:3].T*1000+old[:3,3]
        pre=float(np.max(np.linalg.norm(predicted-local,axis=1)))
        assert pre<1.e-6,(sid,pre)
        new=old.copy();new[:3,:3]=dr.as_matrix()@old[:3,:3];new[:3,3]=dr.apply(old[:3,3])+dt
        owner['source_obj_mm_to_core_inertial_body_m']=new.tolist()
        body_pose=default[b];target=body_pose[:3]+R.from_quat(body_pose[3:]).apply(dr.apply(predicted)+dt)
        owner.update(default_pose_vertex_centroid_world_m=target.mean(axis=0).tolist(),
            vertex_centroid_to_source_com_residual_m=float(np.linalg.norm(target.mean(axis=0)-body_pose[:3])))
        translation,quat,scale=_bodyparts_visual_local_pose(new.tolist(),mid)
        settings[sid]=(offset+24,translation+quat+[scale])
        geometry[sid]=target
        rows.append(dict(stable_id=sid,member_id=mid,member_sha256=anchor['source']['member_sha256'],
            unchanged_parent_transform_max_reconstruction_error_m=pre))
    reg['regional_derivation']['connected_girdle_candidate']=dict(parent_registration_sha256=sha(source),corrections_sha256=sha(change),anchors=rows,
        description="Rigid bilateral shoulder-girdle registration candidate. Original articulated and vascular frames remain fixed. Prior upper-limb admission metrics refer to the parent.")
    rp=out/'candidate-current-source.json'; write(rp,reg)
    struct.pack_into('<I',payload,24,int(sha(rp)[:8],16))
    for off,p in settings.values():struct.pack_into('<8f',payload,off,*p)
    bp=out/parent.name;bp.write_bytes(payload)
    bymember={a['member_id']:i+1 for i,a in enumerate(manifest['source']['anchors'])}
    def coords(n):
        if n in geometry:return geometry[n]
        p=default[cat[n]['body']];return R.from_quat(p[3:]).apply(meshes[n][0])+p[:3]
    for trans in manifest['upper_limb_continuity']['transitions']:
        a,b=map(bymember.__getitem__,trans['source_member_ids'])
        if a in settings or b in settings:
            vs,x=coords(a),coords(b)
            gap=float(cKDTree(vs).query(x)[0].min());trans['minimum_vertex_gap_m']=gap
            assert gap<=trans['maximum_allowed_gap_m'],(trans['name'],gap)
    manifest['source']['registration']={'file':rp.name,'sha256':sha(rp)}
    manifest['source']['connected_girdle_parent']={'bone_sha256':sha(parent),'bone_manifest_sha256':sha(mp),'registration_sha256':sha(source),'changed_anchors':rows}
    manifest['payload'].update(sha256=sha(bp),registration_fingerprint32=sha(rp)[:8])
    manifest['upper_limb_continuity']['group_correction']="Four rigid source transforms corrected on existing body frames. Surface gap witnesses recalculated at default pose."
    manifest['status']='connected_girdle_candidate_requires_coupled_qualification'
    write(out/mp.name,manifest)
    print(sha(bp),sha(rp),[(r['stable_id'],r['unchanged_parent_transform_max_reconstruction_error_m']) for r in rows])
if __name__=='__main__':emit(*map(Path,sys.argv[1:]))
