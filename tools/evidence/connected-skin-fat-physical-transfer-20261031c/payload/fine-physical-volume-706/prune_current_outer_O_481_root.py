"""Replay a witness-pruned subset of a disjoint outer-Skin batch; all final gates retained."""
from pathlib import Path
import json,hashlib,argparse,importlib.util,numpy as np,subprocess,time,ast
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
spc=importlib.util.spec_from_file_location('prune379',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(spc);spc.loader.exec_module(T)
ap=argparse.ArgumentParser();ap.add_argument('--failed-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--current-context-config',type=Path,required=True);ap.add_argument('--current-context-config-sha256',required=True);args=ap.parse_args()
root=args.failed_root;out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
rp=root/'run001/report.json';r=json.loads(rp.read_text())
assert r['status']=='PASS_BOUNDED_OUTER_O_QUOTIENT_PHYSICAL_PREFLIGHT'
audit=root/'qualification435001/audit354/report.json';a=json.loads(audit.read_text())
assert a['status']=='FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
assert a['candidate_changed_record_count_by_role']['I']==a['candidate_changed_record_count_by_role']['D']==0
typedproof=root/'typed434001/receipt.json';tr=json.loads(typedproof.read_text())
assert tr['input_report_sha256']==sha(rp) and tr['status']=='PASS_BOUNDED_OUTER_O_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY'
typed=Path(tr['candidate']);assert sha(typed)==tr['candidate_sha256']==a['roles']['O']['candidate']['sha256']
at=load(typed)
# Reconstruct exact354 record IDs from its unchanged execution rule and fixed197 face bases.
owner=H/'combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py'
assert 'next_id = max(face_ids, default=0) + 1_000_000' in owner.read_text()
tree=ast.parse((H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py').read_text())
fb=None
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FACE_BASE' for x in node.targets):fb=ast.literal_eval(node.value)
assert fb=={'D':0,'O':300000,'I':600000}
assert a['all185_receipt']['negative_face_index'].startswith('global index is -1')
baseid=max([901089]+[fb[k]+a['roles'][k]['baseline_face_count']-1 for k in ('D','O','I')])+1000000
changed=a['roles']['O']['changed_candidate_face_rows']
qmap={baseid+i:int(at['source_collapse_operation_rows'][int(fr)]) for i,fr in enumerate(changed)}
assert all(x>=0 for x in qmap.values()) and len(changed)==a['changed_face_total']
drop=set();why=[]
for pair in a['changed_vs_complete_retained_OID_both_walls_all185']['triangle_pairs']:
 hits=[qmap[int(i)] for i in pair if int(i) in qmap];assert hits,pair
 drop.update(hits);why.append({'kind':'cross-structure-or-retained','pair':pair,'removed_operations':hits})
for pair in a['changed_OID_joint_self_pairs']['triangle_pairs']:
 hits=sorted(set(qmap[int(i)] for i in pair if int(i) in qmap));assert hits,pair
 if set(hits)&drop:continue
 chosen=min(hits,key=lambda j:(r['operations'][j]['combined_native_quality_gain'],-j))
 drop.add(chosen);why.append({'kind':'changed-self','pair':pair,'removed_operations':[chosen]})
assert drop and len(drop)<len(r['operations'])
selection=[i for i in range(len(r['operations'])) if i not in drop]
ops=[r['operations'][i] for i in selection]
source=Path(r['source_O']);assert sha(source)==r['source_O_sha256']
su=load(source);p=su['points_world_m_f32'];f=su['faces'];mp=np.arange(len(p))
for op in ops:mp[op['collapsed_input_point_row']]=op['retained_input_point_row']
nf=mp[f];deg=np.any(np.diff(np.sort(nf,axis=1),axis=1)==0,axis=1)
run=out/'run001';run.mkdir(exist_ok=False)
raw=run/'candidate-O-outer-quotient-raw.npz'
np.savez_compressed(raw,points_world_m=p,faces=nf[~deg],source_input_face_rows=np.flatnonzero(~deg),source_input_vertex_quotient=mp)
roles=[]
for oldrole in r['roles']:
 assert oldrole['role']=='Skin'
 inp=Path(oldrole['input']);assert sha(inp)==oldrole['input_sha256']
 v=load(inp);t=v['tetrahedra'].copy();active=np.ones(len(t),bool);newentries=[]
 for oi,oldoi in enumerate(selection):
  e=oldrole['operations'][oldoi];op=r['operations'][oldoi]
  assert e['operation']==oldoi
  rows=np.asarray(e['source_cell_rows'],np.int64)
  old=t[rows].copy();nt=np.where(old==e['retired_volume_node'],e['kept_volume_node'],old)
  degen=np.any(np.diff(np.sort(nt,axis=1),axis=1)==0,axis=1);t[rows]=nt;active[rows[degen]]=False
  newentries.append(dict(e,operation=oi))
 tt=t[active];d,q=T.m.measurements(v['points_world_m'],tt);assert np.all(d>0)
 v.update(tetrahedra=tt,material_indices=v['material_indices'][active],boundary_faces=T.boundary(tt),retained_input_cell_rows=np.flatnonzero(active),source_O_vertex_quotient=mp,source_interface_collapse_rows=np.arange(len(ops)))
 dest=run/'skin-outer-O-quotient-volume-proposal.npz';np.savez_compressed(dest,**v)
 roles.append(dict(oldrole,output=str(dest),output_sha256=sha(dest),operations=newentries,bad_after=int(np.count_nonzero(~T.m.passes(d,q)))))
newr=dict(r,candidate_O=str(raw),candidate_O_sha256=sha(raw),operations=ops,roles=roles,passing_operations=len(ops),accumulated_absolute_volume_change_m3=sum(abs(x['signed_volume_change_m3']) for x in ops),producer_sha256=sha(__file__),witness_pruning={'failed_report':str(audit),'failed_report_sha256':sha(audit),'failed_batch_report':str(rp),'failed_batch_report_sha256':sha(rp),'retained_operation_rows':selection,'removed_operation_rows':sorted(drop),'witnesses':why,'exact354_source_sha256':sha(owner)})
nr=run/'report.json';nr.write_text(json.dumps(newr,indent=2)+'\n')
def call(tag,argv):
 with (out/(tag+'.log')).open('w') as log:subprocess.run([PY,*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,check=True)
call('typed434',[H/'bind_outer_O_quality_exchange_479_root.py','--report',nr,'--output',out/'typed434001'])
call('qualify439',[H/'qualify_current_outer_O_480_root.py','--root',out,'--current-context-config',args.current_context_config,'--current-context-config-sha256',args.current_context_config_sha256])
proof=out/'qualification435001/receipt.json';pr=json.loads(proof.read_text())
assert pr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
receipt={'status':'PASS_EXACT_WITNESS_PRUNED_OUTER_SKIN_SUBSET_AND_FULL_REAUDIT','physical_admission':False,'native_steps':0,'source_failed_root':str(root),'failed_context_sha256':sha(audit),'removed_operations':sorted(drop),'retained_operations':len(ops),'bad_before':roles[0]['bad_before'],'bad_after':roles[0]['bad_after'],'qualification':str(proof),'qualification_sha256':sha(proof),'candidate':pr['candidate'],'candidate_sha256':pr['candidate_sha256'],'elapsed_s':time.monotonic()-started,'producer_sha256':sha(__file__)}
(out/'witness-pruning-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
