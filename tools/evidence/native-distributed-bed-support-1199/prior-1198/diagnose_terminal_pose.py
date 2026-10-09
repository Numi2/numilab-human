#!/usr/bin/env python3
import hashlib, json, math
from pathlib import Path
import numpy as np

OUT = Path('/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/terminal-anatomy-diagnosis-1198-vs-1191-001')
R1191 = Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run')
R1198 = Path('/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/native-run')
SCAN = Path('/Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/skin-bed-audit-1198-attempt002-terminal-scan')
SCENE = Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-anatomy-manifest.json')
MUSCLES = Path('/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json')
BODY = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json')
BASELINE_GEOM = Path('/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3/lung-geometry-937-full/summary.json')
BASELINE_SKIN_DIFF = Path('/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/native-1191-differential-final-003/summary.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def pose_map(receipt):
    return {int(x['body_index']): x for x in receipt['accepted_registered_body_poses']}
def quat_R(q):
    x,y,z,w = map(float,q)
    n=math.sqrt(x*x+y*y+z*z+w*w)
    x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([
      [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=float)
def angle_deg(R):
    c=max(-1.0,min(1.0,(float(np.trace(R))-1.0)*0.5))
    return math.degrees(math.acos(c))
def rel_pose(poses, parent, child):
    p=poses[parent]; c=poses[child]
    Rp=quat_R(p['quaternion_xyzw']); Rc=quat_R(c['quaternion_xyzw'])
    tp=np.asarray(p['position_m'],float); tc=np.asarray(c['position_m'],float)
    return Rp.T@Rc, Rp.T@(tc-tp)
def pose_distance(a,b):
    return angle_deg(a.T@b)

def load_step(run,step):
    p=run/'accepted-geometry'/('step-{}.receipt.json'.format(step))
    d=read(p)
    assert int(d['accepted_step'])==step, (p,d.get('accepted_step'))
    assert len(d['accepted_registered_body_poses'])==86
    return p,d,pose_map(d)

steps={}
for label,run in [('1191_step0',R1191),('1191_step10000',R1191),('1198_step0',R1198),('1198_step10000',R1198)]:
    step=0 if label.endswith('step0') else 10000
    steps[label]=load_step(run,step)

_,r0_1191,p0_1191=steps['1191_step0']; _,rT_1191,pT_1191=steps['1191_step10000']
_,r0_1198,p0_1198=steps['1198_step0']; pT_path,rT_1198,pT_1198=steps['1198_step10000']
assert r0_1191['accepted_body_state_sha256']==r0_1198['accepted_body_state_sha256']
assert r0_1191['accepted_respiration_state_sha256']==r0_1198['accepted_respiration_state_sha256']
assert r0_1191['accepted_registered_body_poses']==r0_1198['accepted_registered_body_poses']
body_order=read(BODY)['core_tree']['body_order']
body_name=lambda i: body_order[i] if i<len(body_order) else 'body_index_{}'.format(i)

pairs=[
 ('pelvis','femur_r',128,131),('pelvis','femur_l',128,145),
 ('femur_r','tibia_r',131,136),('femur_l','tibia_l',145,150),
 ('tibia_r','talus_r',136,137),('tibia_l','talus_l',150,151),
 ('talus_r','calcn_r',137,138),('talus_l','calcn_l',151,152),
 ('torso','humerus_r',20,41),('torso','humerus_l',20,91),
 ('ulna_r','radius_r',42,43),('ulna_l','radius_l',92,93)]
relative=[]
for label,pname,cname,pi,ci in [(f'{pa}->{ch}',pa,ch,pi,ci) for pa,ch,pi,ci in pairs]:
    rec={'pair':label,'parent_body_index':pi,'child_body_index':ci,'parent_body':body_name(pi),'child_body':body_name(ci)}
    rels={}
    for tag,poses in [('1191_step0',p0_1191),('1191_step10000',pT_1191),('1198_step0',p0_1198),('1198_step10000',pT_1198)]:
        R,t=rel_pose(poses,pi,ci); rels[tag]=(R,t)
        rec[tag]={'relative_rotation_deg_from_identity':angle_deg(R),'relative_translation_parent_frame_mm':(t*1000).tolist()}
    rec['relative_rotation_change_step0_to_terminal_deg']={
       '1191':pose_distance(rels['1191_step0'][0],rels['1191_step10000'][0]),
       '1198':pose_distance(rels['1198_step0'][0],rels['1198_step10000'][0])}
    rec['terminal_1198_minus_1191']={
       'relative_rotation_deg':pose_distance(rels['1191_step10000'][0],rels['1198_step10000'][0]),
       'relative_translation_parent_frame_delta_mm':((rels['1198_step10000'][1]-rels['1191_step10000'][1])*1000).tolist()}
    relative.append(rec)

body_delta=[]
for i in sorted(set(pT_1191)&set(pT_1198)):
    a=pT_1191[i]; b=pT_1198[i]
    dp=(np.asarray(b['position_m'],float)-np.asarray(a['position_m'],float))*1000
    dr=angle_deg(quat_R(a['quaternion_xyzw']).T@quat_R(b['quaternion_xyzw']))
    body_delta.append({'body_index':i,'body':body_name(i),'terminal_translation_delta_mm':dp.tolist(),'terminal_translation_delta_norm_mm':float(np.linalg.norm(dp)),'terminal_orientation_delta_deg':dr})
body_delta.sort(key=lambda x:x['terminal_translation_delta_norm_mm'],reverse=True)

scan_summary=read(SCAN/'summary.json'); scan_result=read(SCAN/'step-10000.result.json')
scene=read(SCENE); muscle=read(MUSCLES)
scene_rows=scene['source_surfaces']; muscle_rows={int(x['stable_id']):x for x in muscle['source']['surfaces']}
targets=[]
for line in (SCAN/'step-10000.targets.jsonl').read_text().splitlines():
    d=json.loads(line)
    n=int(d['intersecting_triangle_pairs'])
    if not n: continue
    sem,sid=map(int,d['surface'])
    if sem==51005:
        src=muscle_rows.get(sid,{})
        binding=src.get('body_bindings',[])
        label=d['source_owner_or_label']
        category='muscle'
        laterality='right' if 'right ' in label.lower() else ('left' if 'left ' in label.lower() else None)
    elif sem==51004:
        label=d['source_owner_or_label']
        body=label.split(' / ')[0]
        member=label.split(' / ',1)[1] if ' / ' in label else None
        bi=body_order.index(body) if body in body_order else None
        src={'source_member':member}
        binding=[{'core_body_index':bi,'myosim_body':body}] if bi is not None else []
        category='bone'
        laterality='right' if 'right' in label.lower() else ('left' if 'left' in label.lower() else None)
    elif sem==51011:
        src=scene_rows.get(str(sid),{})
        binding=[{'core_body_index':src['body_index'],'myosim_body':body_name(src['body_index'])}] if 'body_index' in src else []
        label=src.get('name') or d['source_owner_or_label']
        category='organ_or_vessel_surface'
        laterality='right' if 'right' in label.lower() else ('left' if 'left' in label.lower() else None)
    elif sem==51006:
        src=muscle_rows.get(sid,{})
        binding=src.get('body_bindings',[])
        label=d['source_owner_or_label']; category='tendon'; laterality='right' if 'right ' in label.lower() else ('left' if 'left ' in label.lower() else None)
    else:
        src={}; binding=[]; label=d['source_owner_or_label']; category='other'; laterality=None
    targets.append({'semantic_id':sem,'stable_id':sid,'label':label,'laterality':laterality,'category':category,'intersecting_triangle_pairs':n,'source_binding_member_id':src.get('source_member') or src.get('member_id'),'body_bindings':binding,'pair_coverage_complete':bool(d['pair_coverage_complete'])})
targets.sort(key=lambda x:(x['semantic_id'],x['stable_id']))

counts={}
for t in targets:
    key=str(t['semantic_id']); counts[key]=counts.get(key,0)+t['intersecting_triangle_pairs']

inputs=[
 R1191/'accepted-geometry/step-0.receipt.json',R1191/'accepted-geometry/step-10000.receipt.json',
 R1198/'accepted-geometry/step-0.receipt.json',R1198/'accepted-geometry/step-10000.receipt.json',
 SCAN/'summary.json',SCAN/'step-10000.result.json',SCAN/'step-10000.targets.jsonl',SCAN/'step-10000.crossing-witnesses.jsonl',
 SCENE,MUSCLES,BODY,BASELINE_GEOM,BASELINE_SKIN_DIFF]
source_hashes={str(p):sha(p) for p in inputs if p.is_file()}
source_hashes[str(pT_path)]=sha(pT_path)
report={
 'schema':'numi.human.contoured-bed-terminal-anatomy-diagnosis.v1',
 'scope':'Read-only comparison of accepted registered poses and one exact terminal skin-to-target scan. This is a terminal failure diagnosis, not an eight-pose qualification or causal attribution.',
 'runs':{
  '1191':{'path':str(R1191),'accepted_steps':[0,10000],'initial_body_state_sha256':r0_1191['accepted_body_state_sha256'],'terminal_body_state_sha256':rT_1191['accepted_body_state_sha256'],'terminal_respiration_state_sha256':rT_1191['accepted_respiration_state_sha256'],'terminal_accepted_time_s':rT_1191['accepted_time_s']},
  '1198':{'path':str(R1198),'accepted_steps':[0,10000],'initial_body_state_sha256':r0_1198['accepted_body_state_sha256'],'terminal_body_state_sha256':rT_1198['accepted_body_state_sha256'],'terminal_respiration_state_sha256':rT_1198['accepted_respiration_state_sha256'],'terminal_accepted_time_s':rT_1198['accepted_time_s'],'terminal_pose_receipt_sha256':sha(pT_path)}},
 'initial_pose_identity':{'same_accepted_body_state':True,'same_accepted_respiration_state':True,'same_86_registered_body_poses':True},
 'terminal_body_pose_difference_top10_by_translation':body_delta[:10],
 'relative_segment_pose_comparison':relative,
 'skin_target_terminal_scan':{'summary_path':str(SCAN/'summary.json'),'summary_sha256':sha(SCAN/'summary.json'),'result_path':str(SCAN/'step-10000.result.json'),'result_sha256':sha(SCAN/'step-10000.result.json'),'step':10000,'accepted_time_s':scan_result['accepted_receipt']['accepted_time_s'],'all_859_target_and_skin_self_pair_coverage_complete':scan_result['pair_coverage_complete'],'all_declared_captures_complete':scan_summary['all_declared_captures_complete'],'all_skin_crossing_pair_count':scan_result['all_skin_crossing_pair_count'],'ocular_crossing_pair_count':scan_result['ocular_crossing_pair_count'],'skin_self_crossing_pair_count':scan_result['skin_self_crossing_pair_count'],'targets':targets,'counts_by_semantic_id':counts},
 'baseline_clearance_reference':{'937_full_scan_summary':str(BASELINE_GEOM),'937_summary_sha256':sha(BASELINE_GEOM),'1187_differential_summary':str(BASELINE_SKIN_DIFF),'1187_summary_sha256':sha(BASELINE_SKIN_DIFF),'scope':'1191 target/self clearances are supported by the complete 937 baseline scan and 1187 changed-skin differential; the 1198 report only scanned step 10000 and does not establish full-cycle status.'},
 'interpretation_limits':[
  'The exact step-10000 result is 3,727 skin-to-target triangle-pair intersections across pelvis, bilateral lower-limb muscle surfaces, bilateral posterior tibial veins, right calcaneal tendon, and left forearm/hand muscle surfaces; all declared target pairs were covered and 17 ocular targets had zero hits.',
  'The starting accepted body and respiratory state and all 86 registered body poses are identical between 1191 and 1198. The terminal registered body states differ. Relative 3D segment rotations/translations are reported from accepted body transforms; they are not scalar generalized hip-flexion coordinates.',
  'No q diagnostic was emitted by 1198, so the terminal body-relative hip angle cannot be compared to the separate 1190 scalar hip-fit interval [-12,0]. Do not infer a q limit violation from segment relative rotation.',
  'The evidence shows that terminal pose/contact configuration diverged under the 1198 configuration, but does not isolate the fixed bed as the only cause. It does not support an eight-capture or full-horizon anatomical qualification.'
 ],
 'source_hashes':source_hashes
}
script_path=Path(__file__)
report['analysis_script']={'path':str(script_path),'sha256':sha(script_path)}
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT/'diagnosis.json'),'report_sha256':sha(OUT/'diagnosis.json'),'script_sha256':sha(script_path),'targets':len(targets),'crossing_pairs':scan_result['all_skin_crossing_pair_count'],'relative_pairs':len(relative)},indent=2))
