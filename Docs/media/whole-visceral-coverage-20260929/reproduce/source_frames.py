from pathlib import Path
import json
import numpy as np
import mujoco
from numilab_human import organ_family_geometry as g
from numilab_human.upper_limb_pose_audit import _pose_qpos
r=Path(__file__).resolve().parent;repo=g.ROOT;c=json.loads((repo/'config/source-organ-family-composite.v2.json').read_text())
_,_,_,_,_,m,d,reg,owners,checks=g.context(repo/'Sources',repo/'Build/myosim-fullbody',repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',repo/'Build/organ-family-coverage-20260929/payload.final'/g.PAYLOAD_NAME,c)
sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'head');world=np.asarray(reg['coordinate_system']['global_source_mm_to_myosim_world_m'])
rows=[]
for a in reg['anchors']:
 if a['target']['name']!='head':continue
 local=np.asarray(a['registration']['source_obj_mm_to_core_inertial_body_m'])
 composed=np.eye(4);composed[:3,:3]=d.ximat[sid].reshape(3,3)@local[:3,:3]
 composed[:3,3]=d.ximat[sid].reshape(3,3)@local[:3,3]+d.xipos[sid]
 rows.append({'source_member_id':a['source']['member_id'],'maximum_coefficient_error':float(np.abs(composed-world).max()),'source_status':a['registration']['status'],'passed':bool(np.allclose(composed,world,rtol=0,atol=1e-12))})
assert len(rows)==8 and all(x['passed'] for x in rows)
chains=[]
for name in ['head','neck','pelvis']:
 sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name);chain=[];k=sid
 while k:
  chain.append({'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,k),'source_body_id':int(k),'local_joint_count':int(m.body_jntnum[k])});k=int(m.body_parentid[k])
 assert chain[0]['local_joint_count']==0
 chains.append({'myosim_body':name,'core_body_index':owners[name][0],'ancestors':chain})
report={'schema':'numi.human.source-head-pelvic-frame-reference.v1','passed':True,'skull_anchor_count':8,'skull_common_frame_checks':rows,'source_body_chains':chains,'rigid_source_program_checks':checks,'independent_cervical_and_eye_motion':False,'clinical_registration':False,'intracranial_containment_and_clearance':'not_assessed','boundary':'Algebraic source atlas/common frame agreement and named source body ancestry only. Head and neck are fixed descendants of torso; pelvis is fixed under source root. No anatomical containment, clearance, cervical/ocular actuation, organ mechanics, or clinical qualification.'}
(r/'source-frame-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'skull_anchors':len(rows),'max_coefficient_error':max(x['maximum_coefficient_error'] for x in rows)}))
