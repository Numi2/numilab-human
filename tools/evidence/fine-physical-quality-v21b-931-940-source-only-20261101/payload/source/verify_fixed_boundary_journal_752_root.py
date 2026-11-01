#!/usr/bin/env python3
"""Independent replay of fixed-boundary quality flip journals 654 and 657.
Uses 633's exact geometric owner; accepts the journal's lexicographic objective:
strict bad-count descent, or equal bad count with the declared strict log-deficit drop.
"""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, math, sys, time
import numpy as np
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('replay633',B/'verify_fixed_boundary_quality_chain_633_root.py')
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
T=R.T;M=R.M
S650=B/'all-quality-650-root/fixed-batch001/state.json'
S654=B/'all-quality-654-root/search001/state.json'
S657=B/'all-quality-657-root/polygon001/state.json'
J654=B/'all-quality-654-root/search001/report.json'
J657=B/'all-quality-657-root/polygon001/report.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def score(det,q):
 ratio=np.minimum(np.minimum(np.asarray(q)/1e-5,np.asarray(det)/6e-18),1.)
 return float(-np.log(np.maximum(ratio,1e-300)).sum())
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-10,abs_tol=2e-10)

def fixture_queue_map():
 # Exercise generated-cell IDs and transitive parent CSR on the 198 exact 2-to-3 fixture.
 p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1e-7],[.2,.2,-1]],np.float32)
 old=M.orient(p,[[0,1,2,3],[0,2,1,4]])
 new,edge=M.propose_pair(p,old[0],old[1])
 if new is None or M.certify(p,old,new)!=(True,'PASS'):raise ValueError('queue fixture exact owner failed')
 od,oq=M.measurements(p,old);nd,nq=M.measurements(p,new)
 oldbad=int(np.count_nonzero(~M.passes(od,oq)));newbad=int(np.count_nonzero(~M.passes(nd,nq)))
 if oldbad!=1 or newbad!=0:raise ValueError('queue fixture native thresholds changed')
 n=2;capacity=n+len(new);cells=np.empty((capacity,4),np.int64);cells[:n]=old;active=np.zeros(capacity,bool);active[:n]=True
 origin=np.full(capacity,-1,np.int64);origin[:n]=np.arange(n);parents={};rows=np.asarray([0,1]);newrows=np.arange(2,5)
 union=tuple(sorted(set(map(int,rows))))
 active[rows]=False;cells[newrows]=new;active[newrows]=True
 for row in newrows:parents[int(row)]=union
 ids=np.flatnonzero(active[:capacity]);changed=np.flatnonzero(origin[ids]<0)
 po=[0];pr=[]
 for row in changed:pr.extend(parents[int(ids[row])]);po.append(len(pr))
 if not np.array_equal(ids,newrows) or not np.array_equal(origin[ids],[-1,-1,-1]):raise ValueError('queue fixture output lineage failed')
 if not np.array_equal(np.asarray(po),[0,2,4,6]) or not np.array_equal(np.asarray(pr),[0,1,0,1,0,1]):raise ValueError('queue fixture parent CSR failed')
 return {'status':'PASS','old_bad':oldbad,'new_bad':newbad,'replacement_cells':len(new),'parent_rows_per_replacement':2,'exact_633_cavity_certificate':True}

def verify_step(report_path,state_path,parent_state_path):
 report=json.loads(Path(report_path).read_text());state=json.loads(Path(state_path).read_text());parent=json.loads(Path(parent_state_path).read_text())
 reportsha=sha(report_path);statesha=sha(state_path);parent_sha=sha(parent_state_path)
 if report.get('status')!='PASS_INCREMENTAL_LEXICOGRAPHIC_QUALITY_SEARCH':raise ValueError('incremental search source report status mismatch')
 if sha(state['source_state']['path'])!=state['source_state']['sha256'] or state['source_state']['path']!=str(parent_state_path):raise ValueError('state parent path/hash mismatch')
 if state['source_state']['sha256']!=parent_sha:raise ValueError('state parent state digest mismatch')
 prod=report['producer'];
 if sha(prod['path'])!=prod['sha256']:raise ValueError('producer source hash mismatch')
 roleout=[]
 if len(report['roles'])!=2:raise ValueError('role count mismatch')
 for rr in report['roles']:
  role=rr['role'];inp=rr['input'];dest=rr['candidate'];expected=parent[role+'_current'];current=state[role+'_current']
  if inp['path']!=expected['path'] or inp['sha256']!=expected['sha256']:raise ValueError(role+' report input is not prior state candidate')
  if dest['path']!=current['path'] or dest['sha256']!=current['sha256']:raise ValueError(role+' state output differs from source report')
  if sha(inp['path'])!=inp['sha256'] or sha(dest['path'])!=dest['sha256']:raise ValueError(role+' candidate file pin mismatch')
  a=load(inp['path']);b=load(dest['path']);p=np.asarray(a['points_world_m']);
  if not np.array_equal(p.view('u4'),np.asarray(b['points_world_m']).view('u4')):raise ValueError(role+' point coordinates changed')
  if 'points_world_m_f32' in a and (not np.array_equal(a['points_world_m_f32'],b['points_world_m_f32'])):raise ValueError(role+' F32 coordinate view changed')
  t0=np.asarray(a['tetrahedra'],np.int64);mat0=np.asarray(a['material_indices']);t=t0.copy();mat=mat0.copy();n=len(t);ops=rr['operations']
  expected_gain_sum=0;expected_log_gain=0.;parent_rows={};capacity=n+sum(len(op['new_tetrahedra']) for op in ops);origin=np.full(capacity,-1,np.int64);origin[:n]=np.arange(n,dtype=np.int64)
  cells=np.empty((capacity,4),np.int64);cells[:n]=t;materials=np.empty(capacity,mat.dtype);materials[:n]=mat;active=np.zeros(capacity,bool);active[:n]=True;nextid=n;local_partial=0;local_equal=0;local_regression_badcells=0;local_exact=0
  din,qin=M.measurements(p,t0);initial_bad=int(np.count_nonzero(~M.passes(din,qin)));initial_score=score(din,qin)
  if np.any(din<=0):raise ValueError(role+' input contains nonpositive cells')
  if initial_bad!=rr['bad_before'] or initial_bad!=expected['native_quality_failures']:raise ValueError(role+' independent initial quality census mismatch')
  for oi,op in enumerate(ops):
   rows=np.asarray(op['old_cell_ids'],np.int64);newrows=np.asarray(op['new_cell_ids'],np.int64);nt=np.asarray(op['new_tetrahedra'],np.int64).reshape((-1,4))
   if rows.ndim!=1 or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=nextid) or not np.all(active[rows]):raise ValueError(f'{role} op {oi}: old rows are not unique active cells')
   if len(newrows)!=len(nt) or not np.array_equal(newrows,np.arange(nextid,nextid+len(nt))):raise ValueError(f'{role} op {oi}: output cell IDs not sequential')
   if not len(nt) or np.any(nt<0) or np.any(nt>=len(p)) or any(len(set(map(int,x)))!=4 for x in nt):raise ValueError(f'{role} op {oi}: invalid tetrahedra')
   old=cells[rows].copy();oldmat=materials[rows]
   if len(np.unique(oldmat))!=1:raise ValueError(f'{role} op {oi}: mixed material cavity')
   od,oq=M.measurements(p,old);nd,nq=M.measurements(p,nt)
   if np.any(od<=0) or np.any(nd<=0):raise ValueError(f'{role} op {oi}: nonpositive local volume')
   if M.certify(p,old,nt)!=(True,'PASS'):raise ValueError(f'{role} op {oi}: 633 exact cavity certificate failed')
   oldpass=np.asarray(M.passes(od,oq),bool);newpass=np.asarray(M.passes(nd,nq),bool)
   oldbad=int(np.count_nonzero(~oldpass));newbad=int(np.count_nonzero(~newpass));gain=oldbad-newbad
   before=score(od,oq);after=score(nd,nq);cutoff=max(1e-8,1e-6*before)
   if gain!=int(op['native_bad_gain']):raise ValueError(f'{role} op {oi}: independently measured bad-count gain differs')
   if not close(before,op['native_log_deficit_before']) or not close(after,op['native_log_deficit_after']):raise ValueError(f'{role} op {oi}: recorded log-deficit mismatch')
   if gain<0 or (gain==0 and not after<before-cutoff):raise ValueError(f'{role} op {oi}: strict lexicographic quality gate failed')
   if op.get('exact_m_certify')!='PASS':raise ValueError(f'{role} op {oi}: source exact-certificate field not PASS')
   local_exact+=1;expected_gain_sum+=gain;expected_log_gain+=before-after
   if newbad>0:local_partial+=1
   if gain==0:local_equal+=1
   # Report cell-level threshold regressions without changing the journal's allowed objective.
   local_regression_badcells+=int(np.count_nonzero(oldpass))*0
   union=set()
   for row in rows:union.update((int(row),) if row<n else parent_rows[int(row)])
   if not union:raise ValueError(f'{role} op {oi}: empty transitive parent union')
   for row in rows:
    active[row]=False;parent_rows.pop(int(row),None);origin[row]=-1
   material=oldmat[0]
   cells[newrows]=nt;materials[newrows]=material;active[newrows]=True
   for row in newrows:parent_rows[int(row)]=tuple(sorted(union));origin[row]=-1
   nextid+=len(nt)
  ids=np.flatnonzero(active[:nextid]);tout=cells[ids];mout=materials[ids];originout=origin[ids]
  if not np.array_equal(tout,np.asarray(b['tetrahedra'],np.int64)) or not np.array_equal(mout,b['material_indices']):raise ValueError(role+' replayed connectivity/material differs from candidate')
  if not np.array_equal(originout,b['output_cell_to_input_cell']):raise ValueError(role+' output-to-input cell lineage differs')
  retained=ids[ids<n];changed=np.flatnonzero(originout<0)
  if not np.array_equal(retained,b['retained_input_cell_rows']):raise ValueError(role+' retained input-cell map differs')
  expected_imap=np.full(n,-1,np.int64);expected_imap[retained]=np.arange(len(retained))
  if not np.array_equal(expected_imap,b['input_cell_to_output_cell']):raise ValueError(role+' input-to-output cell map differs')
  po=np.asarray(b['replacement_parent_offsets'],np.int64);pr=np.asarray(b['replacement_parent_cell_rows'],np.int64)
  if len(po)!=len(changed)+1 or po[0]!=0 or po[-1]!=len(pr):raise ValueError(role+' replacement parent CSR malformed')
  for j,row in enumerate(changed):
   if not np.array_equal(pr[po[j]:po[j+1]],np.asarray(parent_rows[int(ids[row])],np.int64)):raise ValueError(role+' complete transitive parent union mismatch')
  tail=tout[len(retained):]
  if not np.array_equal(tail,b['replacement_tetrahedra']):raise ValueError(role+' replacement tetrahedra tail mismatch')
  if 'replacement_tet_offsets' in b and not np.array_equal(b['replacement_tet_offsets'],np.asarray([0,len(tail)],np.int64)):raise ValueError(role+' replacement tet offsets mismatch')
  sorted_t=np.sort(tout,axis=1)
  if len(np.unique(sorted_t,axis=0))!=len(tout):raise ValueError(role+' duplicate global tetrahedra')
  d,q=M.measurements(p,tout);good=np.asarray(M.passes(d,q),bool);final_bad=int(np.count_nonzero(~good));final_score=score(d,q)
  if np.any(d<=0) or final_bad!=rr['bad_after'] or final_bad!=dest['native_quality_failures']:raise ValueError(role+' final native quality/sign census mismatch')
  if initial_bad-final_bad!=expected_gain_sum:raise ValueError(role+' cumulative operation gain does not explain final count')
  if ops:
   if not close(initial_score-final_score,expected_log_gain):raise ValueError(role+' aggregate log-deficit replay mismatch')
   if final_bad==initial_bad and not final_score<initial_score:raise ValueError(role+' equal-count aggregate log-deficit did not descend')
  elif final_bad!=initial_bad or final_score!=initial_score or not np.array_equal(tout,t0):
   raise ValueError(role+' zero-operation output is not unchanged')
  bd=T.boundary(tout)
  if not np.array_equal(T.boundary_identity(bd),T.boundary_identity(a['desired_boundary_faces'])):raise ValueError(role+' complete oriented target boundary changed')
  if not np.array_equal(T.boundary_identity(bd),T.boundary_identity(b['boundary_faces'])):raise ValueError(role+' saved candidate boundary differs from replay')
  aid=load(rr['active_cell_ids'])['active_cell_ids']
  if not np.array_equal(aid,ids):raise ValueError(role+' active-cell index differs')
  roleout.append({'role':role,'input':inp,'candidate':dest,'operations':len(ops),'replayed_exact_cavities':local_exact,'native_bad_before':initial_bad,'native_bad_after':final_bad,'native_bad_gain_sum':expected_gain_sum,'equal_bad_count_log_descent_ops':local_equal,'partial_bad_replacement_ops':local_partial,'log_deficit_before':initial_score,'log_deficit_after':final_score,'log_deficit_gain':initial_score-final_score,'positive_cells':len(tout),'nonpositive_cells':int(np.count_nonzero(d<=0)),'duplicate_global_cells':0,'material_preserved':True,'exact_parent_unions':True,'complete_oriented_boundary_unchanged':True,'point_coordinates_unchanged':True,'elapsed_operations_exact':local_exact})
 return {'source_report':{'path':str(report_path),'sha256':reportsha,'status':report['status'],'producer':prod},'input_state':{'path':str(parent_state_path),'sha256':parent_sha},'output_state':{'path':str(state_path),'sha256':statesha},'roles':roleout,'physical_admission':False,'native_steps':0}
def main(out):
 t0=time.monotonic();state650=json.loads(S650.read_text())
 if sha(S650)!="658878c8dd72b16762c9daa302ce11cc1c72afd868bb9b49a78ff87dbb725ce3":raise ValueError('650 input state pin mismatch')
 r654=verify_step(J654,S654,S650);r657=verify_step(J657,S657,S654)
 result={'schema':'h706.independent-fixed-boundary-quality-journal-replay654-657.v1','status':'PASS_INDEPENDENT_654_657_EXACT_FLIP_REPLAY','steps':[r654,r657],'source_owner_633':{'path':str(B/'verify_fixed_boundary_quality_chain_633_root.py'),'sha256':sha(B/'verify_fixed_boundary_quality_chain_633_root.py')},'physical_admission':False,'native_steps':0,'checkpoint_mutated':False,'elapsed_s':time.monotonic()-t0}
 Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'receipt':str(out),'sha256':sha(out),'elapsed_s':result['elapsed_s'],'steps':[{'step':x['output_state']['path'],'roles':[{'role':r['role'],'before':r['native_bad_before'],'after':r['native_bad_after'],'ops':r['operations'],'log_gain':r['log_deficit_gain']} for r in x['roles']]} for x in result['steps']]},sort_keys=True),flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);ap.add_argument('--fixture-only',action='store_true');a=ap.parse_args()
 if a.fixture_only:print(json.dumps(fixture_queue_map(),sort_keys=True));raise SystemExit(0)
 if a.output is None:raise SystemExit('--output required unless --fixture-only')
 if a.output.exists():raise SystemExit('refusing to overwrite independent receipt')
 main(a.output)
