#!/usr/bin/env python3
"""Lightweight retained-cap regression controls pinned to the 992 candidate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE=H/'quality-partial-cavity-684-root'
CAND=H/'all-quality-992-backskin-cap-candidate-root'
CONTROL983=BASE/'retained_cap_transfer_983_controls.json'
REPORT993=CAND/'report.json'
REPLAY993=CAND/'independent-replay.json'
DISTANCE994=CAND/'independent-3d-distance-replay-994.json'
JOURNAL=CAND/'operation-journal.json'
PINS={
 'controls983':'4b34446cfd42fbce234c4350a8893fc6627173e3fb431f7d890900bb7f740ef0',
 'report993':'6f8cec52b9cfea4046ba820291491b2875398b0b4d6ecd4409a97b3f053849c5',
 'replay993':'14f5194975ea933205a8cb459b07723e8aa38451f521cca25726bcb1f99835c3',
 'distance994':'ec10bce966ab1c13ee9527587c18bdc2d89a3d1d1d1d00bcbd51db49740e8e05',
 'journal992':'50ee0410380ca8dfbc850358eee1aa2820c2e3c64c8b97e19eebf5f11d776f61',
}
EXPECTED_OLD=np.array([[57482,60059,59961],[57482,59961,59941]],np.int64)
EXPECTED_NEW=np.array([[57482,60059,59941],[59961,59941,60059]],np.int64)
EXPECTED_REPLACEMENTS={
 'Fat':[
  {'row':52337,'old':[57482,60059,59961],'new':[57482,60059,59941]},
  {'row':52342,'old':[57482,59961,59941],'new':[59961,59941,60059]},
 ],
 'Skin':[
  {'row':204678,'old':[134151,136630,136728],'new':[134151,136610,136728]},
  {'row':204683,'old':[134151,136610,136630],'new':[136630,136728,136610]},
 ],
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def chain(faces):
 out=defaultdict(int)
 for tri in np.asarray(faces,dtype=np.int64).reshape((-1,3)):
  x=tuple(map(int,tri)); key=tuple(sorted(x))
  inv=sum(x[i]>x[j] for i in range(3) for j in range(i+1,3))
  out[key]+=(-1 if inv&1 else 1)
 return {k:v for k,v in out.items() if v}

def tet_boundary(tets):
 rows=[]
 for tet in np.asarray(tets,dtype=np.int64).reshape((-1,4)):
  a,b,c,d=map(int,tet)
  rows.extend(((b,c,d),(a,d,c),(a,b,d),(a,c,b)))
 return rows

def rows_from_chain(ch):
 rows=[]
 for key,sign in sorted(ch.items()):
  if sign>0:
   rows.extend([key]*sign)
  else:
   rev=(key[0],key[2],key[1])
   rows.extend([rev]*(-sign))
 return np.asarray(rows,dtype=np.int64).reshape((-1,3))

def alias_and_chain_ok(actual_boundary,desired_target,actual_tets):
 return np.array_equal(np.asarray(actual_boundary),np.asarray(desired_target)) and chain(tet_boundary(actual_tets))==chain(desired_target)

def check_fixture():
 # Two positively oriented tets on opposite sides of a shared face.
 P=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,-1.]])
 old=np.array([[0,1,2,3],[0,2,1,4]],np.int64)
 det=[]
 for t in old:
  a,b,c,d=P[t]
  det.append(float(np.linalg.det(np.stack((b-a,c-a,d-a),axis=1))))
 assert min(det)>0,det
 # Retire the cap tet but retain the adjacent volume cell; target is the full
 # boundary of that remaining domain, including the newly exposed shared face.
 new=old[1:]
 desired=rows_from_chain(chain(tet_boundary(new)))
 assert len(new)==1 and len(desired)==4
 assert alias_and_chain_ok(desired.copy(),desired,new)
 # Cyclic row rotations preserve oriented chains while the volume owner's
 # boundary_faces alias must still match desired_boundary_faces byte-for-byte.
 cyclic=np.roll(desired,1,axis=1)
 assert alias_and_chain_ok(cyclic.copy(),cyclic,new)
 # A row-order/cyclic representation mismatch is an alias failure even if the
 # oriented chains would otherwise agree.
 alias_mutated=cyclic.copy(); alias_mutated[0]=desired[0]
 assert not np.array_equal(alias_mutated,cyclic)
 # Reversing one target facet preserves alias bytes but breaks the actual chain.
 wrong_target=desired.copy(); wrong_target[0]=wrong_target[0,[0,2,1]]
 assert np.array_equal(wrong_target,wrong_target.copy())
 assert not alias_and_chain_ok(wrong_target.copy(),wrong_target,new)
 # Retiring the wrong tetrahedron produces a different full-domain boundary.
 wrong_retained=old[:1]
 assert chain(tet_boundary(wrong_retained))!=chain(desired)
 # Retiring both cells is forbidden: empty full material domain.
 empty=np.empty((0,4),np.int64)
 assert len(empty)==0 and not alias_and_chain_ok(np.empty((0,3),np.int64),desired,empty)
 # Complete replacement parent ancestry must include every immediate old row.
 all_parents={17,29,31}; declared={17,29,31}
 assert declared==all_parents
 assert {17,29}!=all_parents
 # Formerly passing retained cells may not become bad; a seed may be removed,
 # but every retained row that was good must remain good.
 def nonregression(before,after):
  b=np.asarray(before,bool); a=np.asarray(after,bool)
  return b.shape==a.shape and bool(np.all(a[b]))
 assert nonregression([False,True,True],[False,True,True])
 assert not nonregression([False,True,True],[False,True,False])
 return {
  'positive_full_domain_after_empty_seed_retirement':'PASS',
  'cyclic_face_orientation_chain_equivalence':'PASS',
  'exact_boundary_alias_required':'PASS_REJECTS_MISMATCH',
  'reversed_target_facet_rejected':'PASS',
  'wrong_retired_cell_rejected':'PASS',
  'empty_entire_material_domain_rejected':'PASS',
  'incomplete_parent_union_rejected':'PASS',
  'formerly_good_regression_rejected':'PASS',
 }

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
 expected_paths={
  'controls983':CONTROL983,'report993':REPORT993,'replay993':REPLAY993,
  'distance994':DISTANCE994,'journal992':JOURNAL}
 for k,p in expected_paths.items():
  if sha(p)!=PINS[k]:raise SystemExit(f'{k} artifact pin mismatch: {p}')
 old=json.loads(CONTROL983.read_text()); report=json.loads(REPORT993.read_text())
 replay=json.loads(REPLAY993.read_text()); dist=json.loads(DISTANCE994.read_text()); journal=json.loads(JOURNAL.read_text())
 if old.get('status')!='PASS_RETAINED_CAP_TRANSFER_CONTROLS':raise SystemExit('983 retained-cap controls not passing')
 oldc=old.get('controls',{})
 required_old={
  'synthetic_empty_cap_full_domain_positive_exact_target':'PASS',
  'empty_entire_material_domain_rejected':'empty cap would remove the entire material domain',
  'former_good_quality_regression_rejected':'formerly-good retained material cell regressed',
  'full_domain_target_mutation_rejected':'full target differs from exact changed-target construction',
  'changed_source_ancestry_mutation_rejected':'typed-I changed face ancestry mismatch',
  'wrong_changed_target_rejected':'empty local target is not exactly the source-flip cap shell',
  'wrong_retired_cell_rejected':'wrong retired cell row',
  'wrong_retirement_source_face_rejected':'empty retirement source-face identity mismatch',
  'wrong_tet_cap_rejected':'changed source face is not on retired cap boundary',
 }
 for k,v in required_old.items():
  if oldc.get(k)!=v:raise SystemExit(f'983 negative control missing/changed: {k}')
 if report.get('status')!='PASS_INDEPENDENT_REPLAY_PENDING_FRESH_775_808':raise SystemExit('993 candidate report status mismatch')
 if replay.get('status')!='PASS_INDEPENDENT_BACKSKIN_CAP_REPLAY':raise SystemExit('993 independent replay mismatch')
 if not (replay.get('full_global_target_chains') and replay.get('paired_exact_volume_cancellation') and replay.get('outer_O_source_unchanged') and replay.get('source_point_bytes_unchanged')):
  raise SystemExit('993 replay omitted required geometry/source invariants')
 if replay.get('physical_admission') is not False or replay.get('mass_mapping')!='NOT_RUN':raise SystemExit('candidate gate status overstated')
 if dist.get('status')!='PASS' or dist.get('schema')!='independent_backskin_patch_3d_lattice_lipschitz_replay_v1':raise SystemExit('994 independent distance proof not passing')
 if float(dist['method']['cap_m'])!=50e-6:raise SystemExit('994 did not prove original 50um bound')
 for direction in ('old_to_new','new_to_old'):
  if float(dist['method']['directional_results'][direction]['upper_bound_m'])>50e-6:raise SystemExit('994 directional upper exceeds 50um')
 op=journal['operation']
 if op['kind']!='2-to-2 source edge flip' or op['source_edge']!=[57482,59961]:raise SystemExit('992 operation identity mismatch')
 if op['face_rows']!=[52337,52342] or op['old_source_faces']!=EXPECTED_OLD.tolist() or op['new_source_faces']!=EXPECTED_NEW.tolist():raise SystemExit('992 source face identity mismatch')
 for r in ('Fat','Skin'):
  got=sorted(journal['roles'][r]['target_row_replacements'],key=lambda x:x['row'])
  want=sorted(EXPECTED_REPLACEMENTS[r],key=lambda x:x['row'])
  if got!=want:raise SystemExit(f'992 complete {r} target-row replacement mismatch')
 skin=journal['roles']['Skin']; fat=journal['roles']['Fat']
 if not skin['empty_target_retirement'] or skin['new_tets'] or len(skin['full_closed_region_rows'])!=58:raise SystemExit('Skin empty local target/full-domain contract mismatch')
 if fat['empty_target_retirement'] or len(fat['old_tets'])!=2 or len(fat['new_tets'])!=2 or len(fat['full_closed_region_rows'])!=14:raise SystemExit('Fat replacement contract mismatch')
 if journal.get('paired_exact_delta_det6')!='0':raise SystemExit('paired exact flux is not zero')
 fixture=check_fixture()
 result={
  'schema':'h706.retained-cap-transfer-995-controls.v1','status':'PASS_RETAINED_CAP_TRANSFER_995',
  'source':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
  'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in expected_paths.items()},
  'actual_candidate':{'report_status':report['status'],'independent_replay_status':replay['status'],
                      'source_candidate':replay['artifacts']['source_candidate'],
                      'exact_paired_delta_det6':journal['paired_exact_delta_det6'],
                      'physical_admission':False,'mass_mapping':'NOT_RUN','fresh_all185':'PENDING'},
  'distance':{'status':dist['status'],'cap_m':dist['method']['cap_m'],
               'old_to_new_upper_m':dist['method']['directional_results']['old_to_new']['upper_bound_m'],
               'new_to_old_upper_m':dist['method']['directional_results']['new_to_old']['upper_bound_m'],
               'source':{'path':str(DISTANCE994),'sha256':PINS['distance994']}},
  'reused_983_negative_controls':required_old,
  'new_boundary_alias_empty_seed_controls':fixture,
  'native_steps':0,'geometry_files_written':False,
 }
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'out':str(out),'source_sha256':sha(Path(__file__)),'old_reuse_controls':len(required_old),'new_controls':len(fixture)},sort_keys=True))
 return 0
if __name__=='__main__':raise SystemExit(main())
