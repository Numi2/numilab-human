#!/usr/bin/env python3
import hashlib, json, struct, sys
from pathlib import Path
ROOT=Path('/Users/n/numi-human-retained-delivery-20261009')
SRC=ROOT/'skin-source-binding-1202/source-inputs/myosim'
CHECKOUT=SRC/'checkout'
ARCHIVE=SRC/'myo_sim-33c89c2b.tar.gz'
RIGID=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
OUT=Path('/Users/n/numi-human-fds132-m6-1350/attempt-050/arrangement-001/diagnostic-061')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def req(ok,msg):
 if not ok:raise RuntimeError(msg)
sys.path.insert(0,str(CHECKOUT))
import mujoco
from myo_sim.build.compose import build_model
m=build_model('myofullbody');d=mujoco.MjData(m);d.qpos[:]=m.qpos0;mujoco.mj_forward(m,d)
raw=RIGID.read_bytes()
magic,abi,engine_abi,source_count,body_count,joint_count,nq,nv,r0,virtual,r1,embedded=struct.unpack_from('<8s10I32s',raw)
body_off=80+96+48;joint_off=body_off+160*body_count;dof_off=joint_off+144*joint_count;map_off=dof_off+64*nv+4*(nq+nv)
req(magic==b'NHRIGID2' and embedded.hex()==sha(ARCHIVE),'NHRIGID source identity mismatch')
req(source_count==m.nbody-1,'source body count mismatch')
source_to_core=struct.unpack_from(f'<{source_count}I',raw,map_off)
core_by_source={i+1:int(v) for i,v in enumerate(source_to_core)}
def name(obj,i):return mujoco.mj_id2name(m,obj,int(i))
def body_name(i):return name(mujoco.mjtObj.mjOBJ_BODY,i)
def route_points(tendon):
 addr=int(m.tendon_adr[tendon]);count=int(m.tendon_num[tendon]);out=[]
 for ri in range(addr,addr+count):
  typ=int(m.wrap_type[ri]);oid=int(m.wrap_objid[ri]);kind={int(mujoco.mjtWrap.mjWRAP_SITE):'site',int(mujoco.mjtWrap.mjWRAP_SPHERE):'sphere',int(mujoco.mjtWrap.mjWRAP_CYLINDER):'cylinder'}.get(typ,str(typ))
  if kind=='site':
   b=int(m.site_bodyid[oid]);out.append({'kind':kind,'source_id':oid,'name':name(mujoco.mjtObj.mjOBJ_SITE,oid),'source_body_id':b,'source_body_name':body_name(b),'core_body_index':core_by_source.get(b),'position_body_m':[float(x) for x in m.site_pos[oid]],'position_world_m':[float(x) for x in d.site_xpos[oid]]})
  else:
   b=int(m.geom_bodyid[oid]);side=int(round(float(m.wrap_prm[ri])));row={'kind':kind,'source_id':oid,'name':name(mujoco.mjtObj.mjOBJ_GEOM,oid),'source_body_id':b,'source_body_name':body_name(b),'core_body_index':core_by_source.get(b),'position_body_m':[float(x) for x in m.geom_pos[oid]],'position_world_m':[float(x) for x in d.geom_xpos[oid]]}
   if side>=0:
    sb=int(m.site_bodyid[side]);row['side_site']={'source_id':side,'name':name(mujoco.mjtObj.mjOBJ_SITE,side),'source_body_id':sb,'source_body_name':body_name(sb),'core_body_index':core_by_source.get(sb),'position_body_m':[float(x) for x in m.site_pos[side]],'position_world_m':[float(x) for x in d.site_xpos[side]]}
   out.append(row)
 return out
matches=[]
for ai in range(m.nu):
 aname=name(mujoco.mjtObj.mjOBJ_ACTUATOR,ai)
 if not aname or not any(s in aname.lower() for s in ('fds','fdp')):continue
 tendon=int(m.actuator_trnid[ai,0]);nodes=route_points(tendon)
 matches.append({'actuator_index':ai,'actuator_name':aname,'tendon_index':tendon,'tendon_name':name(mujoco.mjtObj.mjOBJ_TENDON,tendon),'route_nodes':nodes,'route_body_names':sorted({n['source_body_name'] for n in nodes}|{n['side_site']['source_body_name'] for n in nodes if 'side_site' in n}),'first_route_endpoint':nodes[0],'last_route_endpoint':nodes[-1]})
req(matches,'no compiled FDS/FDP actuator names found')
report={'schema':'numi.human.fds-fdp-source-route-provenance-061.v1','scope':'Read-only dump of named source FDS/FDP compiled MyoSim muscle route endpoints and all site/wrap body identities, joined through exact source-body-to-NHRIGID core map. Route is the source muscle-tendon path. It is not a tissue-attachment or clinical truth claim.','inputs':{'myosim_archive_path':str(ARCHIVE),'myosim_archive_sha256':sha(ARCHIVE),'nhrigid_path':str(RIGID),'nhrigid_sha256':sha(RIGID),'mujoco_version':mujoco.__version__,'compose_source_path':str(CHECKOUT/'myo_sim/build/compose.py'),'compose_source_sha256':sha(CHECKOUT/'myo_sim/build/compose.py')},'source_model':{'name':'myofullbody','body_count_including_world':int(m.nbody),'actuator_count':int(m.nu),'tendon_count':int(m.ntendon),'source_body_count':int(source_count),'engine_body_count':int(body_count)},'matching_muscles':matches,'interpretation':'These are exact named muscle route sites/wraps from the compiled source model. The separate NHTISS weights in witness 056 remain inferred nearest-four visual route-support weights; the match does not transform them into direct attachments.','status':'complete'}
out=OUT/'fds-fdp-source-routes-061.json';out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'matching_muscles':matches},sort_keys=True))
