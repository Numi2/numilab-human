#!/usr/bin/env python3
"""Use the retained connected fixed-boundary remesher on measured core slivers."""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys,time,signal
from contextlib import contextmanager
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
O=B/'minimum-thickness-minkowski-328/attempt-006-anchored-local-topology'
IN=P/'deep-core-void-coalescence-366/candidate-f32-typed.npz'
CLOSED=P/'deep-core-void-coalescence-366/D-closed-csg-only.npz'
PREF=P/'deep-core-connected-patch-preflight-369/patch-preflight.json'
OUT=P/'deep-core-connected-patch-filter-375'
ID=62000200

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@contextmanager
def trial_bound(seconds):
 def timeout(signum,frame):raise TimeoutError('patch triangulation exceeded per-trial budget')
 previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds)
 try:yield
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False);a=load(IN);closed=load(CLOSED);x=a['points_world_m'];f=a['faces'];caps=closed['faces'][len(f):]
 helper=mod('connected_patch_owner',O/'run-029-patch-remesh/condition_outer_patch_remesh.py')
 parent=P/'deep-core-connected-patch-remesh-371';g=load(parent/'geometry-checkpoint.npz');r=json.loads((parent/'receipt.json').read_text());audit=json.loads((P/'deep-core-change-coverage-374/receipt.json').read_text())
 if audit['manifest_banks_checked']!=185 or any(not q['overlapping_patch_components'] for q in audit['positive_lost_bone_witnesses']):raise ValueError('incomplete occupancy witness coverage')
 unsafe=set(audit['unsafe_patch_components']);accepted=[];removed=set();skipped=[];base=len(g['kept_input_face_rows']);nf=g['faces'];off=g['patch_new_face_offsets']
 for i,q in enumerate(r['accepted_components']):
  if q['component'] in unsafe:skipped.append({'component':q['component'],'reason':'removed_due_to_all185_positive_lost_bone_witness_overlap'});continue
  v=dict(q);v['removed_rows']=np.asarray(v['removed_rows'],np.int64);v['new_faces']=nf[base+off[i]:base+off[i+1]];accepted.append(v);removed.update(map(int,v['removed_rows']))
 if not accepted:raise ValueError('no safe patch left')
 print(json.dumps({'retained_patches':len(accepted),'removed_patches':len(skipped),'all185_change_domain_witness_gate':True}),flush=True)
 kept=np.asarray([i for i in range(len(f)) if i not in removed],np.int64);nf=np.concatenate([f[kept]]+[v['new_faces'] for v in accepted]);nnew=len(nf)-len(kept)
 # Save the recovered geometry before packaging, preserving full patch operation support.
 np.savez_compressed(OUT/'geometry-checkpoint.npz',points_world_m=x,faces=nf,kept_input_face_rows=kept,removed_input_face_rows=np.asarray(sorted(removed)),patch_removed_offsets=np.r_[0,np.cumsum([len(v['removed_rows']) for v in accepted])],patch_removed_input_face_rows=np.concatenate([v['removed_rows'] for v in accepted]),patch_new_face_offsets=np.r_[0,np.cumsum([len(v['new_faces']) for v in accepted])])
 off=a['face_ancestry_offsets'];ak=[k for k in a if k.startswith('face_ancestry_') and k!='face_ancestry_offsets'];takes=[np.arange(off[r],off[r+1]) for r in kept]
 # Whole patch support is explicit operation ancestry; it is not a nearest-face measurement.
 for v in accepted:
  support=np.unique(np.concatenate([np.arange(off[r],off[r+1]) for r in v['removed_rows']]));takes.extend([support]*len(v['new_faces']))
 take=np.concatenate(takes);lens=np.asarray([len(v) for v in takes],np.int64);out={}
 for k,v in a.items():
  if k in ak:out[k]=v[take]
  elif k=='face_ancestry_offsets':out[k]=np.r_[0,np.cumsum(lens)]
  elif k=='faces':out[k]=nf
  elif k.startswith('face_') and v.ndim and len(v)==len(f):
   dtype=v.dtype;newvalues=np.full(nnew,-1,dtype=dtype)
   if k=='face_source_original_id':newvalues[:]=ID
   elif k=='face_source_layer':newvalues[:]=2
   elif k in ('face_source_id','face_source_face_id','face_source_identity0'):newvalues=np.arange(nnew,dtype=dtype)
   out[k]=np.concatenate([v[kept],newvalues])
  else:out[k]=v.copy()
 used=np.unique(np.r_[nf.ravel(),caps.ravel()]);remap=np.full(len(x),-1,np.int64);remap[used]=np.arange(len(used));out['points_world_m']=x[used];out['faces']=remap[nf]
 for k in ('source_f64_vertex_rows','emitted_vertex_input_row'):
  if k in out:out[k]=out[k][used]
 if 'vertex_input_old_to_emitted' in out:
  old=out['vertex_input_old_to_emitted'];good=old>=0;new=np.full(old.shape,-1,np.int64);new[good]=remap[old[good]];out['vertex_input_old_to_emitted']=new
 out['cap_boundary_vertex_rows']=remap[out['cap_boundary_vertex_rows']]
 ds=OUT/'D-derived-source-62000200.npz';dface=out['faces'][len(kept):];du=np.unique(dface);dm=np.full(len(used),-1,np.int64);dm[du]=np.arange(len(du))
 darrays={k:out[k][len(kept):] for k in ['face_source_original_id','face_source_face_id','face_source_identity0','face_source_identity1']}
 np.savez_compressed(ds,points_world_m=out['points_world_m'][du],source_faces=dm[dface],**darrays,parent_candidate_sha256=np.asarray([sha(IN)]),source_vertex_parent_candidate_row=used[du],operation=np.asarray(['bounded_connected_fixed_boundary_sliver_remesh']))
 schema=json.loads(str(out['face_source_identity_schema_by_original_id_json'][0]));schema[str(ID)]={k:v for k,v in schema['62000100'].items() if not k.startswith('parent_patch')};schema[str(ID)].update(path=str(ds),sha256=sha(ds),identity_original_id=ID,face_count=nnew,derived_operation='bounded_connected_fixed_boundary_sliver_remesh',derived_parent_assets=[{'path':str(IN),'sha256':sha(IN)}],derived_parent_source_ids=sorted(set(map(int,a['face_source_original_id'][np.asarray(sorted(removed))]))),source_geometry_scope='local_D_connected_sliver_patches')
 out['face_source_identity_schema_by_original_id_json']=np.asarray([json.dumps(schema,sort_keys=True,separators=(',',':'))]);target=OUT/'candidate-f32-typed.npz';np.savez_compressed(target,**out)
 np.savez_compressed(OUT/'D-closed-csg-only.npz',points_world_m=out['points_world_m'],faces=np.r_[out['faces'],remap[caps]])
 np.savez_compressed(OUT/'vertex-compaction-map.npz',old_to_new=remap,new_to_old=used)
 sys.path.insert(0,str(P/'derived-lineage-producer-260'));from export_authoring_registered_surface_sidecars_230 import validate_candidate
 valid=validate_candidate(target,'D',require_f32=True);valid={k:v for k,v in valid.items() if k not in ('points','faces','faceArrays','sourceSchema','faceAncestry')}
 report={'status':'PRODUCER_VALIDATED_OCCUPANCY_FILTERED_CANDIDATE','input':str(IN),'input_sha256':sha(IN),'candidate':str(target),'candidate_sha256':sha(target),'accepted_components':[{k:v for k,v in x.items() if k not in ('new_faces',)} for x in accepted],'skipped':skipped,'input_quality':helper.global_quality(x,f),'output_quality':helper.global_quality(out['points_world_m'],out['faces']),'points_before':len(x),'points_after':len(used),'faces_before':len(f),'faces_after':len(nf),'all_retained_point_coordinates_bitwise_unchanged':True,'producer260_validation':valid,'full_exact_self_audit_run':False,'all185_bank_audit_run':False,'maximum_sampled_patch_deviation_m':max(v['sampled_deviation_m'] for v in accepted),'physical_admission':False,'elapsed_s':time.monotonic()-t}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');print(json.dumps({k:report[k] for k in ['status','candidate_sha256','input_quality','output_quality','points_before','points_after','faces_before','faces_after','elapsed_s']},sort_keys=True))
if __name__=='__main__':main()
