"""Recompile shoulder base placements in the MyoSim source and independently bind native rest poses."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from myo_sim.build.compose import build_spec

def sha(b):return hashlib.sha256(b).hexdigest()
def states(model):
    state=mujoco.MjData(model)
    mujoco.mj_forward(model,state)
    return state
def bind(base,experiment,frame,archive,ref,out,report):
    control=base.read_bytes();candidate=bytearray(experiment.read_bytes())
    _,_,nr,nb,nj,*_=struct.unpack_from('<10I',control,8)
    assert len(candidate)==len(control) and (nr,nb,nj)==(103,157,156)
    mr=json.loads(ref.read_text())
    assert sha(archive.read_bytes())==mr['source']['archive_sha256']
    specs=build_spec('myofullbody'); model=specs.compile(); plain=states(model)
    table=mr['core_tree']['source_body_records']
    owners=np.frombuffer(control,dtype='<u4',count=nr,offset=len(control)-nr*32)
    oracle=np.frombuffer(control,dtype='<f4',count=nr*7,offset=len(control)-nr*28).reshape(nr,7)
    assert all(r['core_body_index']==owners[i] for i,r in enumerate(table))
    edited=[];affected=set()
    def rot(m):return Rotation.from_matrix(m.reshape(3,3))
    for cal,j in zip(json.loads(frame.read_text()),('l','r')):
        n='scapphant_'+j; p='scapula_'+j
        child=specs.body(n); parent=model.body(p).id
        assert model.body_parentid[model.body(n).id]==parent
        # S is the COM-centred native frame, P and C are source link frames.
        rp=rot(plain.xmat[parent]);rs=rot(plain.ximat[parent])
        c0=plain.xipos[parent];p0=plain.xpos[parent]
        cr=Rotation.from_quat(cal['parent_rotation_xyzw'])
        delta_world=rs*cr*rs.inv(); translation_world=rs.apply(cal['parent_translation_base_m'])
        index=model.body(n).id
        prev=plain.xpos[index];prior_rot=rot(plain.xmat[index])
        new_pos=rp.inv().apply(c0+translation_world+delta_world.apply(prev-c0)-p0)
        new_rot=rp.inv()*delta_world*prior_rot
        edited.append(dict(child=n,parent=p,pos_before=child.pos.tolist(),quat_before_wxyz=child.quat.tolist(),
                           pos_after=new_pos.tolist(),quat_after_wxyz=new_rot.as_quat(scalar_first=True).tolist()))
        child.pos=new_pos.tolist();child.quat=new_rot.as_quat(scalar_first=True).tolist()
        for b in range(1,model.nbody):
            k=b
            while k:
                if k==index:affected.add(b);break
                k=model.body_parentid[k]
    newmodel=specs.compile(); new=states(newmodel)
    stats=[]; residuals=[]
    for i,r in enumerate(table):
        b=model.body(r['name']).id
        p0=plain.xipos[b]; q0=rot(plain.ximat[b]);p1=new.xipos[b];q1=rot(new.ximat[b])
        src=oracle[i]; dr=(q0*Rotation.from_quat(src[3:]).inv()).magnitude()
        err=float(np.linalg.norm(p0-src[:3]))
        residuals.append(dict(name=r['name'],affected=b in affected,dx_m=err,da_rad=float(dr)))
        if b in affected:
            if err>1e-6 or dr>2e-6:raise ValueError(f"prior source default mismatched {r['name']} {err} {dr}")
            row=np.asarray([*p1,*q1.as_quat()],dtype='<f4')
            at=len(candidate)-nr*28+28*i
            candidate[at:at+28]=row.tobytes()
            stats.append(dict(body=r['name'],source_id=b,displacement_m=float(np.linalg.norm(p1-p0))))
        elif (np.linalg.norm(p1-p0)>1e-10 or (q0.inv()*q1).magnitude()>1e-9):
            raise ValueError(f"unrelated source pose changed for {r['name']}")
    assert len(affected)==len(stats) and all(s['displacement_m']>0 for s in stats)
    with out.open('xb') as f:f.write(candidate)
    report.write_text(json.dumps(dict(asset_sha256=sha(candidate),control_sha256=sha(control),source_archive_sha256=mr['source']['archive_sha256'],source_reference_sha256=sha(ref.read_bytes()),joint_candidate_sha256=sha(experiment.read_bytes()),mujoco=mujoco.__version__,source_edits=edited,affected_source_bodies=stats,baseline_residuals=residuals),indent=2)+'\n')
if __name__=='__main__':bind(*map(Path,sys.argv[1:]))
