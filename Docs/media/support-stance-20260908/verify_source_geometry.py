from pathlib import Path
import json,hashlib,struct,math,tarfile
import myo_sim
import numpy as np
import mujoco
from myo_sim.build.compose import build_model
from numilab_human.support_stance import assess
base=Path('/Users/home/numilab-human');out=base/'Docs/media/support-stance-20260908'
manifest=json.loads((out/'source-manifest.json').read_text());profile=json.loads((base/'config/myosim-support-stance.v1.json').read_text())
archive=base/'Sources/myosim/myo_sim-33c89c2b.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==profile['source']['archive_sha256']
assert mujoco.__version__=='3.12.0'
checkout=base/'Sources/myosim/checkout'
assert Path(myo_sim.__file__).resolve().is_relative_to(checkout.resolve())
source_files_verified=0
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  parts=Path(member.name).parts
  if member.isfile() and len(parts)>2 and parts[1]=='myo_sim' and Path(member.name).suffix in {'.py','.xml','.yaml','.yml','.json'}:
   relative=Path(*parts[1:]);source=checkout/relative
   assert source.read_bytes()==tar.extractfile(member).read(),str(relative)
   source_files_verified+=1
assert source_files_verified>10
model=build_model('myofullbody');data=mujoco.MjData(model)
assert (model.nq,model.nv,model.nu)==(129,128,416)
state=next(json.loads(l.split('=',1)[1]) for l in (out/'full-capacity-1-0.0001.log').read_text().splitlines() if l.startswith('stand_terminal_state='))
q=state['initial_q']
root_joint=next(i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
root_body=model.jnt_bodyid[root_joint]
com_rotation=np.empty(9);inertial_rotation=np.empty(9)
mujoco.mju_quat2Mat(com_rotation,np.array([q[6],q[3],q[4],q[5]]))
mujoco.mju_quat2Mat(inertial_rotation,model.body_iquat[root_body])
body_rotation=com_rotation.reshape(3,3)@inertial_rotation.reshape(3,3).T
body_quaternion=np.empty(4);mujoco.mju_mat2Quat(body_quaternion,body_rotation.ravel())
data.qpos[:3]=np.array(q[:3])-body_rotation@model.body_ipos[root_body]
data.qpos[3:7]=body_quaternion
for j in manifest['core_tree']['source_joint_map']:
 assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j['source_joint_id'])==j['source_name']
 data.qpos[model.jnt_qposadr[j['source_joint_id']]]=q[j['core_q_index']]
mujoco.mj_forward(model,data)
payload=(base/'Build/completion-20260907/input/myosim-fullbody-support-contact.nhcnt').read_bytes()
assert hashlib.sha256(payload).hexdigest()==profile['payloads']['support_contact']['sha256']
plane=np.array(struct.unpack_from('<3f',payload,56));normal=np.array(struct.unpack_from('<3f',payload,68))
body_ids={b['core_body_index']:b['source_body_id'] for b in manifest['core_tree']['source_body_records']}
metrics,failures=assess((out/'final-certificate/stdout.log').read_text(),0);assert not failures
rows=[];force=np.zeros(3);moment=np.zeros(3)
for i in range(10):
 row=struct.unpack_from('<2I10f',payload,84+48*i);body=body_ids[row[0]];geom=row[1]
 witness=data.xipos[body]+data.ximat[body].reshape(3,3)@np.array(row[2:5])
 R=data.geom_xmat[geom].reshape(3,3);center=data.geom_xpos[geom];size=model.geom_size[geom];kind=model.geom_type[geom]
 if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
  axis=R[:,2];primitive=center-size[1]*axis*np.sign(normal@axis)-size[0]*normal
 elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:primitive=center-size[0]*normal
 elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
  local=R.T@normal;primitive=center-R@(size*size*local/np.linalg.norm(size*local))
 else:raise AssertionError('unexpected source primitive')
 f=float(metrics[f'contact_{i}_normal_force_n']);force+=f*normal;moment+=np.cross(witness,f*normal)
 rows.append({'witness_index':i,'source_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom),'point_world_m':witness.tolist(),'witness_gap_m':float((witness-plane)@normal),'primitive_gap_m':float((primitive-plane)@normal),'normal_force_n':f})
grav=model.body_mass[:,None]*model.opt.gravity[None,:];force+=grav.sum(axis=0);moment+=np.cross(data.xipos,grav).sum(axis=0)
report={'schema':'numi.human.source-stance-oracle.v1','engine':'MuJoCo','version':mujoco.__version__,'numpy_version':np.__version__,'archive_source_files_verified':source_files_verified,'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'input_state_sha256':hashlib.sha256((out/'full-capacity-1-0.0001.log').read_bytes()).hexdigest(),'coordinates':'native accepted initializer converted to source qpos; no source dynamics stepping','source_model_counts':{'nq':model.nq,'nv':model.nv,'muscles':model.nu},'world_force_residual_n':force.tolist(),'world_moment_residual_nm':moment.tolist(),'minimum_witness_gap_m':min(r['witness_gap_m'] for r in rows),'minimum_primitive_gap_m':min(r['primitive_gap_m'] for r in rows),'contacts':rows,'boundary':'Independent pinned source kinematics and external wrench check at the FP32 native initial pose; not material calibration or dynamic contact qualification.'}
report['witness_geometry_within_native_tolerance']=report['minimum_witness_gap_m']>=-1e-6
report['primitive_geometry_within_native_tolerance']=report['minimum_primitive_gap_m']>=-1e-6
report['external_wrench_within_1e_minus3']=bool(max(np.max(np.abs(force)),np.max(np.abs(moment)))<=1e-3)
assert report['witness_geometry_within_native_tolerance'] and report['external_wrench_within_1e_minus3']
(out/'source-oracle.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='contacts'},indent=2))
