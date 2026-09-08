from pathlib import Path
import argparse,json,hashlib,struct,tarfile
import myo_sim
import numpy as np
import mujoco
from myo_sim.build.compose import build_model
from numilab_human.support_stance import assess
parser=argparse.ArgumentParser()
parser.add_argument('--certificate',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--state',type=Path)
args=parser.parse_args()
base=next(p for p in Path(__file__).resolve().parents if (p/'src/numilab_human').is_dir())
profile=json.loads((base/'config/myosim-support-stance.v2.json').read_text())
manifest_path=base/'Docs/media/support-stance-20260908/source-manifest.json'
manifest=json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==profile['source_manifest_sha256']
archive=base/'Sources/myosim/myo_sim-33c89c2b.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==profile['source']['archive_sha256']
assert mujoco.__version__=='3.12.0'
checkout=base/'Sources/myosim/checkout'
assert Path(myo_sim.__file__).resolve().is_relative_to(checkout.resolve())
verified=0
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  parts=Path(member.name).parts
  if member.isfile() and len(parts)>2 and parts[1]=='myo_sim' and Path(member.name).suffix in {'.py','.xml','.yaml','.yml','.json'}:
   assert (checkout/Path(*parts[1:])).read_bytes()==tar.extractfile(member).read()
   verified+=1
assert verified>10
text=args.certificate.read_text()
metrics,failures=assess(text,0,18);assert not failures,failures
poses=[json.loads(l.split(' q=',1)[1]) for l in text.splitlines() if l.startswith('compiled_support_stance=')]
assert len(poses)==2 and poses[0]==poses[1]
# Admission precision of the Metal generalized-state buffers; no source dynamics stepping.
equilibrium=[json.loads(l.split('=',1)[1]) for l in text.splitlines() if l.startswith('compiled_equilibrium_q=')]
assert len(equilibrium)==1
q=np.array(equilibrium[0],dtype=np.float32).astype(np.float64)
if args.state:
 states=[json.loads(l.split('=',1)[1]) for l in args.state.read_text().splitlines() if l.startswith('stand_terminal_state=')]
 assert states and all(t==states[0] for t in states)
 q=np.array(states[0]['q'],dtype=np.float64)
model=build_model('myofullbody');data=mujoco.MjData(model)
root=next(i for i in range(model.njnt) if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
body=model.jnt_bodyid[root];comR=np.empty(9);inertialR=np.empty(9)
mujoco.mju_quat2Mat(comR,np.array([q[6],q[3],q[4],q[5]]));mujoco.mju_quat2Mat(inertialR,model.body_iquat[body])
R=comR.reshape(3,3)@inertialR.reshape(3,3).T;quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
data.qpos[:3]=q[:3]-R@model.body_ipos[body];data.qpos[3:7]=quat
for j in manifest['core_tree']['source_joint_map']:
 assert mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j['source_joint_id'])==j['source_name']
 data.qpos[model.jnt_qposadr[j['source_joint_id']]]=q[j['core_q_index']]
mujoco.mj_forward(model,data)
payload_path=Path(__file__).parent/'myosim-fullbody-support-primitives.nhcnt';payload=payload_path.read_bytes()
assert hashlib.sha256(payload).hexdigest()==profile['payloads']['support_contact']['sha256']
plane=np.array(struct.unpack_from('<3f',payload,56));normal=np.array(struct.unpack_from('<3f',payload,68))
body_ids={b['core_body_index']:b['source_body_id'] for b in manifest['core_tree']['source_body_records']}
rows=[];primitives=[];force=np.zeros(3);moment=np.zeros(3);rowindex=0
for i in range(10):
 p=struct.unpack_from('<4I20f',payload,84+96*i);nativeBody,geom,kind,_=p[:4];sourceBody=body_ids[nativeBody]
 G=data.geom_xmat[geom].reshape(3,3);center=data.geom_xpos[geom];size=model.geom_size[geom]
 assert model.geom_bodyid[geom]==sourceBody
 if kind==2:
  assert model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_CAPSULE
  sourcepoints=[center-size[1]*G[:,2]-size[0]*normal,center+size[1]*G[:,2]-size[0]*normal]
 elif kind==3:
  assert model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_ELLIPSOID
  direction=G.T@normal;sourcepoints=[center-G@(size*size*direction/np.linalg.norm(size*direction))]
 else:raise AssertionError('source geometry kind drift')
 primitives.append({'source_name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom),
                    'gap_m':min(float((x-plane)@normal) for x in sourcepoints)})
 for endpoint,point in enumerate(sourcepoints):
  centre=np.array(p[4:7] if endpoint==0 else p[8:11]);bodyR=data.ximat[sourceBody].reshape(3,3)
  lowered=data.xipos[sourceBody]+bodyR@centre
  if kind==2:lowered-=p[7]*normal
  else:
   shapeR=np.empty(9);mujoco.mju_quat2Mat(shapeR,np.array([p[19],p[16],p[17],p[18]]))
   shapeR=bodyR@shapeR.reshape(3,3);direction=shapeR.T@normal;radii=np.array(p[20:23])
   lowered-=shapeR@(radii*radii*direction/np.linalg.norm(radii*direction))
  f=float(metrics[f'contact_{rowindex}_normal_force_n']);force+=f*normal;moment+=np.cross(point,f*normal)
  rows.append({'row':rowindex,'source_name':primitives[-1]['source_name'],'endpoint':endpoint,
               'gap_m':float((point-plane)@normal),'lowering_error_m':float(np.max(np.abs(lowered-point))),
               'force_n':f});rowindex+=1
assert rowindex==18
grav=model.body_mass[:,None]*model.opt.gravity[None,:];force+=grav.sum(axis=0);moment+=np.cross(data.xipos,grav).sum(axis=0)
report={'schema':'numi.human.curved-support-source-oracle.v1','mujoco_version':mujoco.__version__,
        'source_files_verified':verified,'source_archive_sha256':profile['source']['archive_sha256'],
        'certificate_sha256':hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        'contact_payload_sha256':hashlib.sha256(payload).hexdigest(),
        'minimum_full_primitive_gap_m':min(p['gap_m'] for p in primitives),
        'maximum_lowering_error_m':max(r['lowering_error_m'] for r in rows),
        'force_residual_n':force.tolist(),'moment_residual_nm':moment.tolist(),
        'compiled_q_fp32':q.tolist(),'contacts':rows,'primitives':primitives,
        'boundary':'Independent source FK and wrench at FP32 admission precision; no dynamic standing claim.'}
report['primitive_geometry_passed']=report['minimum_full_primitive_gap_m']>=-1e-6
report['lowering_passed']=report['maximum_lowering_error_m']<=2e-7
report['wrench_passed']=bool(max(np.max(np.abs(force)),np.max(np.abs(moment)))<=1e-3)
if args.state:
 violations=[]
 for j in range(model.njnt):
  if model.jnt_limited[j]:
   value=float(data.qpos[model.jnt_qposadr[j]]);low,high=model.jnt_range[j]
   violation=max(0.0,float(low)-value,value-float(high))
   if violation>0:violations.append({'joint':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j),'violation':violation})
 for row in report['contacts']:row.pop('force_n')
 report.update(force_residual_n=None,moment_residual_nm=None,wrench_passed=None,
               native_state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
               joint_limit_violations=sorted(violations,key=lambda r:r['violation'],reverse=True),
               boundary='Source FK at final native diagnostic state. Dynamic joint limits/contact remain unqualified; static forces are not applied to this state.')
args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('compiled_q_fp32','contacts','primitives')},indent=2))
assert report['primitive_geometry_passed'] and report['lowering_passed'] and (args.state is not None or report['wrench_passed'])
