from pathlib import Path
import numpy as np,json,hashlib,collections,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209'); B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'; out=P/'canonical-O005-lineage-352'; out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def asset(p):return {'path':str(p),'sha256':sha(p)}
oldp=B/'O-exact-witness-repair-286/candidate.npz'; cp=B/'minimum-thickness-minkowski-328/attempt-005-postbuffer-simplify/candidate-O-set_tolerance-f64.npz'; mp=cp.with_name('candidate-O-set_tolerance-face-parent-one-to-many.npz'); bp=B/'minimum-thickness-minkowski-328/attempt-004-prebuffer-simplify/skin-buffer-after-pre-simplify-f64.npz'; capfile=B/'inputs/O-temporary-closed.npz'; oz=B/'emitted-OID-joint-nesting-309/O-closed-F32-stage.npz'; cb=cp.with_name('skin-buffer-set_tolerance-f64.npz')
a=load(oldp); c=load(cp); m=load(mp); cap=load(capfile); b=load(bp)
schema=json.loads(a['face_source_identity_schema_by_original_id_json'].item()); n=len(c['faces_f64']); orig=np.repeat(c['run_original_id'],np.diff(c['run_index'])//3).astype(np.int64); fid=c['face_id'].astype(np.int64)
assert len(orig)==n and np.array_equal(orig,m['candidate_face_origin']) and np.array_equal(fid,m['candidate_face_id'])
assert sha(bp)==str(m['buffer_parent_surface_sha256'][0]); assert sha(cb)==str(m['conditioned_buffer_sha256'][0])
fields=('face_source_layer','face_source_body','face_source_bone','face_source_identity0','face_source_identity1')
lookup={}; ambiguous=[]
for row,key in enumerate(zip(a['face_source_original_id'].tolist(),a['face_source_face_id'].tolist())):
 value=tuple(int(a[k][row]) for k in fields)
 if key in lookup and lookup[key]!=value:ambiguous.append((key,lookup[key],value))
 else:lookup[key]=value
assert not ambiguous,ambiguous[:5]
# Explicit source view for temporary construction caps, never an original anatomical face.
capid=120000101; bufferid=1134834; assert str(capid) not in schema and str(bufferid) not in schema
caprows=cap['temporary_cap_face_rows'].astype(np.int64); assert len(caprows)==41
capview=out/'O-temporary-cap-source.npz'; np.savez_compressed(capview,points_world_m=cap['points_world_m'],faces=cap['faces'][caprows],face_id=caprows,face_source_identity0=caprows,face_source_identity1=np.full(41,-1,np.int64),source_input_face_rows=caprows,temporary_construction_only=np.asarray([True]))
schema[str(capid)]={'role':'O','derived_operation':'exact_temporary_closure_caps_for_CSG_only','path':str(capview),'sha256':sha(capview),'face_count':41,'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_id','derived_parent_source_ids':[1],'derived_parent_assets':[asset(capfile)],'temporary_construction_only':True}
# Buffer union output is an explicit derived surface, not a measured source face.
rows=np.flatnonzero(orig==bufferid); bv=out/'O005-buffer-union-derived-source.npz'; off=m['buffer_parent_offsets']; counts=off[rows+1]-off[rows]; parentrows=np.concatenate([m['buffer_parent_rows'][off[r]:off[r+1]] for r in rows]); bo=np.repeat(b['run_original_id'],np.diff(b['run_index'])//3); bf=b['face_id']; newoff=np.r_[0,np.cumsum(counts)]
np.savez_compressed(bv,points_world_m_f64=c['points_world_m_f64'],faces_f64=c['faces_f64'][rows],face_id=rows,face_source_identity0=rows,face_source_identity1=np.full(len(rows),-1,np.int64),O005_input_face_rows=rows,parent_buffer_offsets=newoff,parent_buffer_rows=parentrows,parent_buffer_original_ids=bo[parentrows],parent_buffer_face_ids=bf[parentrows],parent_buffer_surface_sha256=np.asarray([sha(bp)]),source_O005_sha256=np.asarray([sha(cp)]),parent_support_semantics=np.asarray(['tag-derived one-to-many support; empty rows are newly constructed offset/union faces with full operation inputs as parents']))
schema[str(bufferid)]={'role':'O','derived_operation':'conditioned_skin_buffer_union_output','path':str(bv),'sha256':sha(bv),'face_count':len(rows),'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_id','derived_parent_source_ids':sorted(map(int,schema.keys())),'derived_parent_assets':[asset(cp),asset(cb),asset(bp),asset(mp),asset(oz),asset(oldp)],'source_geometry_scope':'referenced_descendant_faces_only','parent_support_semantics':'new operation faces have no fabricated single source face; complete operation inputs retained'}
result=np.full((n,5),-999999,np.int64); outorig=orig.copy(); outfid=fid.copy(); missing=[]; capset=set(map(int,caprows))
for r,(o,f) in enumerate(zip(orig.tolist(),fid.tolist())):
 if o==bufferid:
  result[r]=[0,-1,-1,r,-1];outfid[r]=r
 elif o==1 and f in capset:
  result[r]=[0,-1,-1,f,-1];outorig[r]=capid
 elif (o,f) in lookup:result[r]=lookup[(o,f)]
 else:missing.append([r,o,f])
assert not missing,missing[:20]
assert np.all(result[:,3]>=0)
# Validate every referenced schema identity pair and derived face triple against real arrays.
validation=[]
for o in np.unique(outorig):
 info=schema[str(int(o))];mask=outorig==o
 if info.get('role')=='BONE':
  assert np.all(result[mask,4]==-1) and np.all(result[mask,3]>=0) and np.all(result[mask,3]<info['face_count'])
  validation.append({'original_id':int(o),'output_rows':int(mask.sum()),'source_schema_kind':'retained declared BONE envelope source from verified O286 packet','source_schema':info});continue
 src=Path(info['path']);assert sha(src)==info['sha256'],str(src)
 z=np.load(src,allow_pickle=False);k0,k1=info['identity_keys'];assert k0 in z and k1 in z
 select=np.ones(len(z[k0]),bool)
 if info.get('identity_original_id_key'):select=np.asarray(z[info['identity_original_id_key']])==int(o)
 have=set(zip(np.asarray(z[k0])[select].tolist(),np.asarray(z[k1])[select].tolist()));need=set(map(tuple,result[mask,3:5].tolist()));assert not need-have,(o,list(need-have)[:3])
 validation.append({'original_id':int(o),'output_rows':int(mask.sum()),'source_path':str(src),'source_sha256':sha(src)})
output=out/'O005-canonical-face-lineage.npz';data={k:result[:,i] for i,k in enumerate(fields)};data.update(face_source_original_id=outorig,face_source_face_id=outfid,legacy_face_source_original_id=orig,legacy_face_source_face_id=fid,source_cache_rows=np.arange(n,dtype=np.int64),source_cache_sha256=np.asarray([sha(cp)]),temporary_cap_descendant_face_mask=np.isin(outorig,[capid]),face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True)]));np.savez_compressed(output,**data)
receipt={'status':'PASS','scope':'metadata crosswalk only; not geometry or physical qualification','input':asset(cp),'output':asset(output),'input_faces':n,'buffer_derived_faces':len(rows),'temporary_caps':int(np.count_nonzero(outorig==capid)),'ambiguous_inherited_identity_pairs':len(ambiguous),'missing_parent_rows':len(missing),'referenced_sources':validation,'current_derived_source_id_reserved':120000100,'legacy_collision':{'id':60000010,'O286_source_preserved':schema.get('60000010'),'later_run008_id_reuse':'must not use legacy current tag alone; assign fresh derived source with complete O005 CSR'}}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='referenced_sources'},indent=2))
