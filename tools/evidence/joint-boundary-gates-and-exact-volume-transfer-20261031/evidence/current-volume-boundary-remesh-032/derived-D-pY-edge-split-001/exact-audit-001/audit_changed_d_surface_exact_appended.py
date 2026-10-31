#!/usr/bin/env python3
"""Exact changed-D-vs-complete O/I/D, paired-wall, and all185 audit.
The candidate remains diagnostic unless its typed ancestry is independently rebuilt.
"""
from pathlib import Path
from fractions import Fraction
import argparse, hashlib, importlib.util, itertools, json, sys, time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fixed-boundary-quality-topology-196')
BASE_DRIVER=P/'run_fixed_boundary_dp_batch_197.py'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise RuntimeError(f'cannot import {path}')
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def tojson(x):
 if isinstance(x,dict):return {str(k):tojson(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [tojson(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.integer):return int(x)
 if isinstance(x,np.floating):return float(x)
 if isinstance(x,Fraction):return str(x)
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return repr(x)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--config',required=True,help='pinned run config for baseline O/I/D')
 ap.add_argument('--candidate-d',required=True);ap.add_argument('--candidate-sha256',required=True)
 ap.add_argument('--changed-rows-key',default='changed_parent_face_rows')
 ap.add_argument('--expected-parent-sha256',default=None)
 ap.add_argument('--candidate-report',default=None);ap.add_argument('--candidate-report-sha256',default=None)
 ap.add_argument('--out',required=True)
 a=ap.parse_args();tic=time.monotonic();config_path=Path(a.config).resolve();candidate_path=Path(a.candidate_d).resolve();out=Path(a.out).resolve()
 config=json.loads(config_path.read_text())
 if config.get('schema')!='h706.fixed-boundary-short-edge-collapse-run-config.v1':raise ValueError('run config schema mismatch')
 if sha(candidate_path)!=a.candidate_sha256:raise ValueError('candidate D SHA mismatch')
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 report=None
 if a.candidate_report:
  rp=Path(a.candidate_report).resolve()
  if not a.candidate_report_sha256 or sha(rp)!=a.candidate_report_sha256:raise ValueError('candidate report SHA missing or mismatched')
  report=json.loads(rp.read_text())
 if (a.candidate_report is None)!=(a.candidate_report_sha256 is None):raise ValueError('candidate report path/hash must be supplied together')
 roles=('O','I','D');cur={r:load(config['current_candidates'][r]['path']) for r in roles}
 for r in roles:
  cp=Path(config['current_candidates'][r]['path']).resolve()
  if sha(cp)!=config['current_candidates'][r]['sha256']:raise ValueError(f'{r} current baseline SHA mismatch')
 parent=cur['D'];cand=load(candidate_path);ppin=str(config['current_candidates']['D']['sha256'])
 if a.expected_parent_sha256 and a.expected_parent_sha256!=ppin:raise ValueError('configured expected parent differs from config D pin')
 # Candidate must expose explicit row correspondences for the full point domain and every retained face.
 pointmap=np.asarray(cand.get('point_output_to_parent_rows',np.empty(0)),np.int64)
 facemap=np.asarray(cand.get('face_output_to_parent_rows',np.empty(0)),np.int64)
 P32=np.asarray(cand.get('points_world_m_f32',np.asarray(cand['points_world_m']).astype(np.float32)),np.float32)
 F=np.asarray(cand['faces'],np.int64);baseP=np.asarray(parent['points_world_m_f32'],np.float32);baseF=np.asarray(parent['faces'],np.int64)
 if pointmap.ndim!=1 or len(pointmap)!=len(P32) or np.any(pointmap < -1) or np.any(pointmap>=len(baseP)):raise ValueError('candidate point-to-parent map malformed')
 mapped_point_rows=pointmap[pointmap>=0]
 if len(np.unique(mapped_point_rows))!=len(mapped_point_rows) or not np.array_equal(np.sort(mapped_point_rows),np.arange(len(baseP),dtype=np.int64)):raise ValueError('candidate must preserve every parent point exactly once and map only explicit appended points to -1')
 appended_point_rows=np.flatnonzero(pointmap<0).astype(np.int64)
 if facemap.ndim!=1 or len(facemap)!=len(F) or np.any(facemap < -1) or np.any(facemap>=len(baseF)):raise ValueError('candidate face-to-parent map malformed')
 if F.ndim!=2 or F.shape[1]!=3 or np.any(F<0) or np.any(F>=len(P32)):raise ValueError('candidate D triangle array invalid')
 point_changed=np.flatnonzero((pointmap<0) | np.any(P32!=baseP[np.maximum(pointmap,0)],axis=1)).astype(np.int64)
 expected_changed=set()
 for outrow,parrow in enumerate(facemap):
  mapped=pointmap[F[outrow]]
  if parrow<0 or np.any(mapped<0) or not np.array_equal(mapped,baseF[int(parrow)]):expected_changed.add(outrow)
 if len(point_changed):
  changed_point_set=set(map(int,point_changed))
  expected_changed.update(int(i) for i,tri in enumerate(F) if any(int(v) in changed_point_set for v in tri))
 if a.changed_rows_key not in cand:raise ValueError(f'candidate does not contain {a.changed_rows_key}')
 changed=np.asarray(cand[a.changed_rows_key],np.int64).reshape(-1)
 if len(np.unique(changed))!=len(changed) or np.any(changed<0) or np.any(changed>=len(F)):raise ValueError('candidate changed row list malformed')
 if set(map(int,changed))!=expected_changed:raise ValueError('declared changed-row set does not exactly cover face-map and F32 point changes')
 if len(changed)==0:raise ValueError('candidate has no changed D rows')
 declared_retired=np.asarray(cand.get('retired_parent_face_rows',np.empty(0)),np.int64).reshape(-1)
 if len(np.unique(declared_retired))!=len(declared_retired) or np.any(declared_retired<0) or np.any(declared_retired>=len(baseF)):raise ValueError('declared retired parent face rows malformed')
 retained_parent_rows=facemap[facemap>=0]
 if len(np.unique(retained_parent_rows))!=len(retained_parent_rows) or not np.array_equal(np.sort(retained_parent_rows),np.setdiff1d(np.arange(len(baseF),dtype=np.int64),declared_retired,assume_unique=False)):raise ValueError('candidate face map must retain each unretired parent face exactly once and explicitly retire only declared parents')
 declared_changed=np.asarray(cand.get('changed_parent_face_rows',np.empty(0)),np.int64).reshape(-1)
 # Candidate ancestry must bind this exact immediate parent when such a field is present.
 parent_fields=('parent_candidate_sha256','immediate_parent_candidate_sha256','source_candidate_sha256')
 observed_parent=[]
 for key in parent_fields:
  if key in cand:
   values=np.asarray(cand[key]).reshape(-1)
   if len(values)==1:observed_parent.append((key,str(values[0])))
 if observed_parent and not any(v==ppin for _,v in observed_parent):raise ValueError(f'candidate parent SHA fields do not include pinned D parent {ppin}: {observed_parent}')
 if report is not None:
  outputs=report.get('output',report.get('candidate_D',{}))
  if isinstance(outputs,dict) and outputs.get('sha256') not in (None,a.candidate_sha256):raise ValueError('candidate report output SHA disagrees')
 # Topology, boundary namespace, and quality diagnostics.
 base=module('dp_batch_197_helpers_198',BASE_DRIVER);helper=base.module('quality_warning_helpers_198',base.HELPER)
 if np.any(F[:,0]==F[:,1]) or np.any(F[:,1]==F[:,2]) or np.any(F[:,2]==F[:,0]):raise ValueError('candidate contains degenerate index face')
 _,top=base.topology(F)
 if top['nonmanifold_edges'] or top['same_direction_edges']:raise ValueError(f'candidate topology invalid: {top}')
 old_boundary=np.sort(helper.boundary(baseF),axis=1)
 new_boundary=np.sort(pointmap[np.asarray(helper.boundary(F),np.int64)],axis=1)
 if not np.array_equal(old_boundary,new_boundary):raise ValueError('candidate D boundary edge set differs after exact point-row mapping')
 # Exact complete context: current baseline O/I/D, current paired walls, all185.
 pred=base.module('quality_owner_loader_198',base.OWNER).load_exact_owner()
 if sha(Path(pred.__file__))!=base.OWNER_IMPL_SHA:raise ValueError('exact predicate implementation SHA changed')
 bone_mod=base.module('bone_barrier_198',base.BONE);filter_mod=base.module('filtered_exact_owner_198',base.FILTER)
 pbase={};cursor=0;combined=[];facebase=base.FACE_BASE.copy()
 for r in ('D','O','I'):
  rp=np.asarray(cur[r]['points_world_m_f32'],np.float32);pbase[r]=cursor;combined.append(rp);cursor+=len(rp)
 combined_points=np.concatenate(combined,axis=0);surface_records=[]
 for r in ('D','O','I'):
  rp=np.asarray(cur[r]['points_world_m_f32'],np.float32);rf=np.asarray(cur[r]['faces'],np.int64)
  raw,_=pred.float32_surface_records(np.asarray(rp,np.float64),rf,source_seam_groups=())
  if len(raw)!=len(rf):raise ValueError(f'{r} exact baseline surface record drop')
  surface_records.extend((q[0],q[1],q[2],facebase[r]+i,tuple(pbase[r]+int(v) for v in q[4])) for i,q in enumerate(raw))
 pointlookup={}
 for r in ('D','O','I'):
  for i,x in enumerate(cur[r]['points_world_m_f32']):
   key=np.ascontiguousarray(x).tobytes()
   if key in pointlookup:raise ValueError(f'cross-role duplicate baseline F32 point {r}/{i}')
   pointlookup[key]=(r,i)
 wall_receipts=[]
 for vol,marker,expected_roles in (('Skin',3,('O','I')),('Fat',6,('I','D'))):
  spath=base.SHELL[vol][0];sh=load(spath);sp=np.asarray(sh['points_world_m'],np.float32);sf=np.asarray(sh['faces'],np.int64);mk=np.asarray(sh['face_marker'],np.int64);wall=np.flatnonzero(mk==marker);used=np.unique(sf[wall]);rm=np.full(len(sp),-1,np.int64)
  for sr in used:
   hit=pointlookup.get(np.ascontiguousarray(sp[int(sr)]).tobytes())
   if hit is None or hit[0] not in expected_roles:raise ValueError(f'{vol} wall row lacks current role map')
   rm[int(sr)]=pbase[hit[0]]+int(hit[1])
  wf=rm[sf[wall]]
  if np.any(wf<0):raise ValueError(f'{vol} wall map incomplete')
  raw,_=pred.float32_surface_records(np.asarray(combined_points,np.float64),wf,source_seam_groups=())
  if len(raw)!=90:raise ValueError(f'{vol} wall exact record census !=90')
  wallbase={'Skin':900000,'Fat':901000}[vol]
  surface_records.extend((q[0],q[1],q[2],wallbase+i,tuple(map(int,q[4]))) for i,q in enumerate(raw))
  wall_receipts.append({'role':vol,'path':str(spath),'sha256':base.SHELL[vol][1],'marker':marker,'wall_faces':len(raw)})
 bones,bone_receipt=bone_mod.load_bone_barrier_records(pred,root=base.ROOT,vertex_offset=len(combined_points),expected_native_sha256=base.BONE_NATIVE,expected_manifest_sha256=base.BONE_MANIFEST)
 if len(bones)!=938272:raise ValueError('all185 F32/F64 barrier record census mismatch')
 exact_index=pred._prepare_surface_aabb(surface_records+bones)
 allowed={facebase[r]+i for r in roles for i in range(len(cur[r]['faces']))}
 filtered=filter_mod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=())
 # Exclude only original D faces retired/replaced by the candidate rows.
 retired_parent_rows=np.unique(np.r_[facemap[changed][facemap[changed]>=0],declared_retired]).astype(np.int64)
 excluded={facebase['D']+int(row) for row in retired_parent_rows};filtered.set_excluded(excluded)
 raw,_=pred.float32_surface_records(np.asarray(P32,np.float64),F[changed],source_seam_groups=())
 if len(raw)!=len(changed):raise ValueError('exact owner dropped a changed candidate D face')
 new_id_base=max(facebase[r]+len(cur[r]['faces']) for r in roles)+1000000
 appended_id_base=1<<40
 if appended_point_rows.size and appended_id_base+int(np.max(appended_point_rows))<max(facebase[r]+len(cur[r]['faces']) for r in roles):raise ValueError('appended-point collision-free ID namespace is not above existing role face identity range')
 def candidate_vertex_id(v):
  v=int(v);par=int(pointmap[v])
  return pbase['D']+par if par>=0 else appended_id_base+v
 recs=[(q[0],q[1],q[2],new_id_base+i,tuple(candidate_vertex_id(v) for v in q[4])) for i,q in enumerate(raw)]
 against=filtered._audit_pair_prepared_first(exact_index,recs,same_surface=True)
 selfpairs=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
 report={'schema':'h706.changed-D-vs-complete-exact-audit.v1','status':'PASS_EXACT_CHANGED_D_VS_COMPLETE_OID_WALLS_ALL185' if against['count']==0 and selfpairs['count']==0 else 'FAIL_EXACT_CHANGED_D_VS_COMPLETE_OID_WALLS_ALL185','physical_admission':False,'native_steps':0,'config':{'path':str(config_path),'sha256':sha(config_path)},'baseline_candidates':config['current_candidates'],'candidate_D':{'path':str(candidate_path),'sha256':sha(candidate_path),'parent_D_sha256':ppin},'candidate_report':None if not a.candidate_report else {'path':str(Path(a.candidate_report).resolve()),'sha256':sha(a.candidate_report)},'changed_rows_key':a.changed_rows_key,'changed_face_rows':changed.tolist(),'changed_face_count':len(changed),'changed_point_rows':point_changed.tolist(),'changed_point_count':len(point_changed),'appended_point_rows':appended_point_rows.tolist(),'appended_point_vertex_id_base':1<<40,'retired_parent_face_rows':declared_retired.tolist(),'face_map_direct_identity':bool(len(F)==len(baseF) and np.array_equal(facemap,np.arange(len(F),dtype=np.int64))),'boundary_edge_count':top['boundary_edges'],'nonmanifold_edges':top['nonmanifold_edges'],'same_direction_edges':top['same_direction_edges'],'changed_vs_complete_OID_SkinFat_walls_all185':tojson(against),'changed_D_internal_self':tojson(selfpairs),'wall_receipts':wall_receipts,'all185_receipt':tojson(bone_receipt),'changed_face_qbound_min':float(np.min(base.qbound(P32,F[changed]))),'changed_face_qbound_bad_count':int(np.count_nonzero(base.qbound(P32,F[changed])<1e-5)),'typed_ancestry_status':'not asserted by this geometry-only owner; a separate typed ancestry/patch-union binding remains required','elapsed_s':time.monotonic()-tic}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('changed_face_rows','changed_point_rows','changed_vs_complete_OID_SkinFat_walls_all185','changed_D_internal_self','all185_receipt')},indent=2,default=str))
 print('pair_counts',against.get('count'),selfpairs.get('count'),'report_sha256',sha(out/'report.json'))
if __name__=='__main__':main()
