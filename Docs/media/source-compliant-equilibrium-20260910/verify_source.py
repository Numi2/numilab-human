"""Independent pinned-source kinematics, mass action and gravity; no stepping."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import tarfile


def main():
    import mujoco
    import myo_sim
    import numpy as np
    from myo_sim.build.compose import build_model

    parser=argparse.ArgumentParser()
    parser.add_argument('--certificate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-trace',type=Path)
    args=parser.parse_args()
    base=next(p for p in Path(__file__).resolve().parents if (p/'src/numilab_human').is_dir())
    profile=json.loads((base/'config/myosim-support-stance.v2.json').read_text())
    manifest_path=base/'Docs/media/support-stance-20260908/source-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==profile['source_manifest_sha256']
    archive=base/'Sources/myosim/myo_sim-33c89c2b.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==profile['source']['archive_sha256']
    checkout=base/'Sources/myosim/checkout'
    assert Path(myo_sim.__file__).resolve().is_relative_to(checkout.resolve()) and mujoco.__version__=='3.12.0'
    verified=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            parts=Path(member.name).parts
            if member.isfile() and len(parts)>2 and parts[1]=='myo_sim' and Path(member.name).suffix in {'.py','.xml','.yaml','.yml','.json'}:
                assert (checkout/Path(*parts[1:])).read_bytes()==tar.extractfile(member).read()
                verified+=1
    assert verified>10
    text=args.certificate.read_text()
    def record(prefix):
        values=[json.loads(l.split('=',1)[1]) for l in text.splitlines() if l.startswith(prefix+'=')]
        assert len(values)==1
        return values[0]
    q=np.array(record('compiled_equilibrium_q'),dtype=np.float64)
    result=record('source_compliant_equilibrium')
    model=build_model('myofullbody');data=mujoco.MjData(model)
    root=next(i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
    body=model.jnt_bodyid[root];comR=np.empty(9);inertialR=np.empty(9)
    mujoco.mju_quat2Mat(comR,np.array([q[6],q[3],q[4],q[5]]))
    mujoco.mju_quat2Mat(inertialR,model.body_iquat[body])
    rotation=comR.reshape(3,3)@inertialR.reshape(3,3).T;quat=np.empty(4)
    mujoco.mju_mat2Quat(quat,rotation.ravel())
    data.qpos[:3]=q[:3]-rotation@model.body_ipos[body];data.qpos[3:7]=quat
    transform=np.zeros((model.nv,128));transform[:3,:3]=np.eye(3);transform[3:6,3:6]=rotation.T
    x,y,z=rotation@model.body_ipos[body]
    transform[:3,3:6]=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    for joint in manifest['core_tree']['source_joint_map']:
        source=joint['source_joint_id']
        assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,source)==joint['source_name']
        data.qpos[model.jnt_qposadr[source]]=q[joint['core_q_index']]
        transform[model.jnt_dofadr[source],joint['core_v_index']]=1
    data.qvel[:]=0;mujoco.mj_forward(model,data)
    mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,data,mass)
    # Use explicit matrix-vector contractions and require finite operands
    # and results. The warning-producing BLAS run is retained separately
    # for comparison; its warning cause is not established here.
    acceleration=np.array(result['acceleration'])
    assert all(np.isfinite(v).all() for v in (transform,mass,acceleration,data.qfrc_bias))
    source_acceleration=np.einsum('ij,j->i',transform,acceleration)
    inertial_force=np.einsum('ij,j->i',mass,source_acceleration)
    source_force=np.einsum('ij,i->j',transform,inertial_force)
    source_gravity=np.einsum('ij,i->j',transform,data.qfrc_bias)
    assert np.isfinite(source_force).all() and np.isfinite(source_gravity).all()
    def error(a,b):
        a,b=np.array(a),np.array(b)
        return float(np.max(np.abs(a-b)/(1+np.abs(a)+np.abs(b))))
    mass_error=error(source_force,result['force_residual'])
    gravity_error=error(source_gravity,result['gravity_target'])
    payload=(base/'Docs/media/curved-support-20260908/myosim-fullbody-support-primitives.nhcnt').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==profile['payloads']['support_contact']['sha256']
    plane=np.array(struct.unpack_from('<3f',payload,56));normal=np.array(struct.unpack_from('<3f',payload,68))
    body_ids={b['core_body_index']:b['source_body_id'] for b in manifest['core_tree']['source_body_records']}
    rows=[];force=np.zeros(3);moment=np.zeros(3);index=0
    for i in range(10):
        p=struct.unpack_from('<4I20f',payload,84+96*i);native_body,geom,kind,_=p[:4]
        source_body=body_ids[native_body];G=data.geom_xmat[geom].reshape(3,3);center=data.geom_xpos[geom];size=model.geom_size[geom]
        assert model.geom_bodyid[geom]==source_body
        if kind==2:
            assert model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_CAPSULE
            points=[center-size[1]*G[:,2]-size[0]*normal,center+size[1]*G[:,2]-size[0]*normal]
        else:
            assert kind==3 and model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_ELLIPSOID
            direction=G.T@normal;points=[center-G@(size*size*direction/np.linalg.norm(size*direction))]
        for endpoint,point in enumerate(points):
            local=np.array(p[4:7] if endpoint==0 else p[8:11]);bodyR=data.ximat[source_body].reshape(3,3)
            lowered=data.xipos[source_body]+bodyR@local
            if kind==2:lowered-=p[7]*normal
            else:
                shapeR=np.empty(9);mujoco.mju_quat2Mat(shapeR,np.array([p[19],p[16],p[17],p[18]]))
                shapeR=bodyR@shapeR.reshape(3,3);direction=shapeR.T@normal;radii=np.array(p[20:23])
                lowered-=shapeR@(radii*radii*direction/np.linalg.norm(radii*direction))
            f=result['support_normal_force'][index];force+=f*normal;moment+=np.cross(point,f*normal)
            rows.append(dict(row=index,gap_m=float((point-plane)@normal),force_n=f,lowering_error_m=float(np.max(np.abs(lowered-point)))))
            index+=1
    assert index==18
    gravity=model.body_mass[:,None]*model.opt.gravity[None,:]
    force+=gravity.sum(axis=0);moment+=np.cross(data.xipos,gravity).sum(axis=0)
    report=dict(schema='numi.human.source-compliant-independent-audit.v1',mujoco_version=mujoco.__version__,
        source_files_verified=verified,source_archive_sha256=profile['source']['archive_sha256'],
        certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        maximum_relative_mass_action_error=mass_error,maximum_relative_gravity_error=gravity_error,
        maximum_support_lowering_error_m=max(r['lowering_error_m'] for r in rows),
        minimum_support_gap_m=min(r['gap_m'] for r in rows),
        force_residual_n=force.tolist(),moment_residual_nm=moment.tolist(),contacts=rows,
        boundary='Independent source FK, mass and gravity; source scalar force conformance is qualified separately; no dynamics stepping or experimental calibration.')
    report['passed']=bool(mass_error<=1e-6 and gravity_error<=2e-6 and
        report['maximum_support_lowering_error_m']<=2e-7 and report['minimum_support_gap_m']>=-1e-6 and
        max(np.max(np.abs(force)),np.max(np.abs(moment)))<=1e-3)
    if args.native_trace:
        trace=args.native_trace.read_text();fields={};initial=[]
        for line in trace.splitlines():
            if line.startswith('mrnx_initial_buffer='):
                row=json.loads(line.split('=',1)[1])
                if row['name']=='initial_q':initial.append(row['values'])
            if line.startswith('prepared_recruitment={'):
                row=json.loads(line.split('=',1)[1])
                if row['scenario']=='recruited' and row['root']==1:
                    assert row['schema']=='numi.human.prepared-recruitment-trace.v2'
                    raw=base64.b64decode(row['fp32_le_base64'],validate=True)
                    fields[row['kind']]=struct.unpack('<'+str(len(raw)//4)+'f',raw)
        assert len(initial)==4 and np.array_equal(np.array(initial[0],dtype=np.float32),q.astype(np.float32))
        assert len(fields['path_length'])==len(manifest['muscles'])==416
        rows=[]
        for i,muscle in enumerate(manifest['muscles']):
            source=muscle['source_actuator_index']
            assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,source)==muscle['name']
            source_length=float(data.actuator_length[source]);native_length=fields['path_length'][i]
            rows.append(dict(index=i,name=muscle['name'],source_length_m=source_length,native_length_m=native_length,
                absolute_length_error_m=abs(source_length-native_length),native_path_velocity_m_s=fields['path_velocity'][i],
                native_fiber_velocity_m_s=fields['fiber_velocity'][i],native_normalized_fiber_residual=fields['fiber_residual'][i]))
        report['initial_muscle_path_diagnostic']={'trace_sha256':hashlib.sha256(args.native_trace.read_bytes()).hexdigest(),
            'maximum_length_error_m':max(r['absolute_length_error_m'] for r in rows),
            'maximum_path_speed_m_s':max(abs(r['native_path_velocity_m_s']) for r in rows),
            'maximum_normalized_fiber_residual':max(abs(r['native_normalized_fiber_residual']) for r in rows),
            'rows':rows,'boundary':'Same FP32 initial pose; source/native path comparison and native fibre diagnostics, no calibrated compliant-force equivalence claim.'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='contacts'},indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
