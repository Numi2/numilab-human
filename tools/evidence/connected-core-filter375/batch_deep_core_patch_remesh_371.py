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
OUT=P/'deep-core-connected-patch-remesh-371'
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
 t=time.monotonic();OUT.mkdir(exist_ok=False);a=load(IN);closed=load(CLOSED);pref=json.loads(PREF.read_text())
 if sha(IN)!=pref['input_sha256']:raise ValueError('inventory pin mismatch')
 x=a['points_world_m'];f=a['faces'];caps=closed['faces'][len(f):]
 if not np.array_equal(closed['points_world_m'],x) or not np.array_equal(closed['faces'][:len(f)],f):raise ValueError('cap coordinate domain mismatch')
 helper=mod('connected_patch_owner',O/'run-029-patch-remesh/condition_outer_patch_remesh.py')
 ear=mod('bounded_patch_owner',O/'run-030-global-derived-patch-remesh/patch_triangulation.py');flip=helper.loadmod('flip_owner',helper.FLIP)
 exact=flip.Flips(x,f);exact.exact=flip.exact
 em=helper.edges_of(f);nbr=[set() for _ in f]
 for rr in em.values():
  if len(rr)==2:nbr[rr[0]].add(rr[1]);nbr[rr[1]].add(rr[0])
 capverts=set(map(int,caps.ravel()));removed=set();records=[];accepted=[];skipped=[];deadline=time.monotonic()+150
 for item in pref['components']:
  ci=item['component'];core=set(item['core_face_rows']);patch=set(core);log=[];chosen=None
  if core&removed:continue
  for depth in range(4):
   if depth:
    patch=patch|set().union(*(nbr[r] for r in patch))
   if patch&removed:log.append({'depth':depth,'reason':'overlaps_accepted_patch'});continue
   loops,directed=helper.boundary_loops(f,patch);chi,nv,ne=helper.comp_euler(patch,f)
   if chi!=1 or len(loops)!=1 or set(loops[0])&capverts:
    log.append({'depth':depth,'reason':'not_disk_or_protected_cap','chi':chi,'loops':len(loops)});continue
   # Grow until every fixed boundary edge is above the construction target.
   loop=loops[0];bd=x[np.asarray(loop)].astype(np.float64);bdlength=np.linalg.norm(np.roll(bd,-1,axis=0)-bd,axis=1)
   if bdlength.min()<helper.MIN_EDGE_M:
    log.append({'depth':depth,'reason':'short_fixed_boundary_edge','min_edge_m':float(bdlength.min())});continue
   rows=np.asarray(sorted(patch),np.int64);tri=x[f[rows]].astype(np.float64);guide=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]).sum(0)
   if np.linalg.norm(guide)==0:log.append({'depth':depth,'reason':'zero_guide'});continue
   # Reject broad curved patches before an expensive planar approximation attempt.
   used_patch=np.unique(f[rows]);qq=x[used_patch].astype(np.float64);center=qq.mean(0);_,_,vt=np.linalg.svd(qq-center,full_matrices=False)
   plane_max=float(np.abs((qq-center)@vt[-1]).max())
   if len(rows)>32 and plane_max>2e-5:log.append({'depth':depth,'reason':'broad_nonplanar_patch','plane_max_m':plane_max});continue
   try:
    with trial_bound(2.0):
     new,point_map,tm=ear.remesh_patch_disk(x,f[rows],loop,guide,coalesce_threshold_m=5e-6,protected_vertices=capverts,max_deviation_m=1e-5,max_constraint_flips=256)
    new=np.asarray(new,np.int64)
    ndedges=helper.edges_of(new)
    boundary_edges={tuple(sorted((loop[j],loop[(j+1)%len(loop)]))) for j in range(len(loop))}
    if any(e not in boundary_edges and any(r not in patch for r in em.get(e,())) for e in ndedges):
     log.append({'depth':depth,'reason':'new_internal_edge_occupied_outside_patch'});continue
   except Exception as ex:log.append({'depth':depth,'reason':'triangulation','error':str(ex)});continue
   nloop,ndir=helper.boundary_loops(new,set(range(len(new))))
   if len(nloop)!=1 or set(ndir)!=set(directed):log.append({'depth':depth,'reason':'boundary_changed'});continue
   edge,alt,_=helper.qmetrics(x,new)
   if edge.min()<helper.MIN_EDGE_M or alt.min()<helper.MIN_ALT_M:
    log.append({'depth':depth,'reason':'quality','min_edge_m':float(edge.min()),'min_altitude_m':float(alt.min())});continue
   dev,dab,dba=helper.hausdorff_samples(x,f[rows],new)
   if dev>1e-5:log.append({'depth':depth,'reason':'deviation','sampled_bidirectional_max_m':dev});continue
   try:npairs,nrecords=helper.exact_candidate_records(exact,new,removed|patch,records)
   except Exception as ex:log.append({'depth':depth,'reason':'exact_owner_error','error':str(ex)});continue
   if npairs!=0:log.append({'depth':depth,'reason':'exact_new_self_pairs','pairs':npairs});continue
   chosen={'component':ci,'depth':depth,'removed_rows':rows,'new_faces':new,'sampled_deviation_m':dev,'old_to_new_m':dab,'new_to_old_m':dba,'min_new_edge_m':float(edge.min()),'min_new_altitude_m':float(alt.min()),'local_exact_new_pairs':0,'trials':log};records.extend(nrecords);break
  if chosen is None:skipped.append({'component':ci,'trials':log})
  else:
   accepted.append(chosen);removed.update(map(int,chosen['removed_rows']));print(json.dumps({'component':ci,'removed':len(chosen['removed_rows']),'new':len(chosen['new_faces']),'deviation_um':chosen['sampled_deviation_m']*1e6}),flush=True)
  if time.monotonic()>deadline:break
 if not accepted:
  r={'status':'NO_CANDIDATE','input_sha256':sha(IN),'accepted':0,'skipped':skipped,'elapsed_s':time.monotonic()-t,'physical_admission':False};(OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));return
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
 report={'status':'PRODUCER_VALIDATED_CANDIDATE','input':str(IN),'input_sha256':sha(IN),'candidate':str(target),'candidate_sha256':sha(target),'accepted_components':[{k:v for k,v in x.items() if k not in ('new_faces',)} for x in accepted],'skipped':skipped,'input_quality':helper.global_quality(x,f),'output_quality':helper.global_quality(out['points_world_m'],out['faces']),'points_before':len(x),'points_after':len(used),'faces_before':len(f),'faces_after':len(nf),'all_retained_point_coordinates_bitwise_unchanged':True,'producer260_validation':valid,'full_exact_self_audit_run':False,'all185_bank_audit_run':False,'maximum_sampled_patch_deviation_m':max(v['sampled_deviation_m'] for v in accepted),'physical_admission':False,'elapsed_s':time.monotonic()-t}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');print(json.dumps({k:report[k] for k in ['status','candidate_sha256','input_quality','output_quality','points_before','points_after','faces_before','faces_after','elapsed_s']},sort_keys=True))
if __name__=='__main__':main()
