from pathlib import Path
import numpy as np,json,hashlib
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union';out=P/'canonical-I328-lineage-356';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asset(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
oldp=B/'I-expanded-boundary-remesh-296-producer260-ready/candidate.npz';cp=B/'minimum-thickness-minkowski-328/attempt-004-prebuffer-simplify/candidate-I-after-fat-f64.npz';bp=cp.with_name('fat-buffer-after-pre-simplify-f64.npz');capfile=B/'inputs/I-temporary-closed.npz';istage=B/'emitted-OID-joint-nesting-309/I-closed-F32-stage.npz';dstage=B/'emitted-OID-joint-nesting-309/D-closed-F32-stage.npz';manifest=cp.with_name('source-derived-manifest.json');originalD=B/'inputs/D-temporary-closed.npz'
a=load(oldp);c=load(cp);b=load(bp);cap=load(capfile);schema=json.loads(a['face_source_identity_schema_by_original_id_json'].item());orig=np.repeat(c['run_original_id'],np.diff(c['run_index'])//3).astype(np.int64);fid=c['face_id'].astype(np.int64);n=len(fid);fields=('face_source_layer','face_source_body','face_source_bone','face_source_identity0','face_source_identity1');lookup={};amb=[]
for r,key in enumerate(zip(a['face_source_original_id'].tolist(),a['face_source_face_id'].tolist())):
 v=tuple(int(a[k][r]) for k in fields)
 if key in lookup and lookup[key]!=v:amb.append((key,lookup[key],v))
 else:lookup[key]=v
assert not amb
capid=120000201;bufid=841487;assert str(capid) not in schema and str(bufid) not in schema;caprows=cap['temporary_cap_face_rows'].astype(np.int64);assert len(caprows)==41
capview=out/'I-temporary-cap-source.npz';np.savez_compressed(capview,points_world_m=cap['points_world_m'],faces=cap['faces'][caprows],face_id=caprows,face_source_identity0=caprows,face_source_identity1=np.full(41,-1,np.int64),source_input_face_rows=caprows,temporary_construction_only=np.asarray([True]));schema[str(capid)]={'role':'I','derived_operation':'exact_temporary_closure_caps_for_CSG_only','path':str(capview),'sha256':sha(capview),'face_count':41,'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_id','derived_parent_source_ids':[3],'derived_parent_assets':[asset(capfile)],'temporary_construction_only':True}
rows=np.flatnonzero(orig==bufid);bo=np.repeat(b['run_original_id'],np.diff(b['run_index'])//3);bf=b['face_id'].astype(np.int64);byid={}
for r,i in enumerate(bf):byid.setdefault(int(i),[]).append(r)
off=[0];parents=[]
for r in rows:parents.extend(byid.get(int(fid[r]),[]));off.append(len(parents))
view=out/'I328-fat-buffer-union-derived-source.npz';np.savez_compressed(view,points_world_m_f64=c['points_world_m_f64'],faces_f64=c['faces_f64'][rows],face_id=rows,face_source_identity0=rows,face_source_identity1=np.full(len(rows),-1,np.int64),I328_input_face_rows=rows,parent_buffer_offsets=np.asarray(off,np.int64),parent_buffer_rows=np.asarray(parents,np.int64),parent_buffer_original_ids=bo[parents],parent_buffer_face_ids=bf[parents],parent_buffer_surface_sha256=np.asarray([sha(bp)]),source_I328_sha256=np.asarray([sha(cp)]),parent_support_semantics=np.asarray(['one-to-many offset-face support through retained Manifold face tags; not a unique original anatomical face']))
# Parent inputs explicitly identify the offset/union operation, including the
# D stage used to construct it. No nearest-source triangle is fabricated.
schema[str(bufid)]={'role':'I','derived_operation':'minimum_fat_thickness_buffer_union_output','path':str(view),'sha256':sha(view),'face_count':len(rows),'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_id','derived_parent_source_ids':sorted(map(int,schema.keys())),'derived_parent_assets':[asset(cp),asset(bp),asset(istage),asset(dstage),asset(oldp),asset(originalD),asset(manifest)],'source_geometry_scope':'referenced_descendant_faces_only','parent_D_stage':asset(dstage),'parent_support_semantics':'whole declared offset operation plus tag-derived buffer-face support; not a fabricated measured-person or original single-face match'}
result=np.full((n,5),-999999,np.int64);outorig=orig.copy();outfid=fid.copy();missing=[];caps=set(caprows.tolist())
for r,(o,i) in enumerate(zip(orig.tolist(),fid.tolist())):
 if o==bufid:result[r]=[1,-1,-1,r,-1];outfid[r]=r
 elif o==3 and i in caps:result[r]=[1,-1,-1,i,-1];outorig[r]=capid
 elif (o,i) in lookup:result[r]=lookup[(o,i)]
 else:missing.append([r,o,i])
assert not missing,missing[:10];assert np.all(result[:,3]>=0)
validation=[]
for sid in np.unique(outorig):
 info=schema[str(int(sid))];mask=outorig==sid
 if info.get('role')=='BONE':assert np.all(result[mask,4]==-1) and np.all(result[mask,3]>=0) and np.all(result[mask,3]<info['face_count']);continue
 src=Path(info['path']);assert sha(src)==info['sha256'];z=load(src);k0,k1=info['identity_keys'];selected=np.ones(len(z[k0]),bool)
 if 'identity_original_id_key' in info:selected=np.asarray(z[info['identity_original_id_key']])==sid
 have=set(zip(z[k0][selected].tolist(),z[k1][selected].tolist()));need=set(map(tuple,result[mask,3:5].tolist()));assert not need-have,(sid,list(need-have)[:3])
 if info.get('identity_face_id_key'):
  have=set(zip(z[info['identity_face_id_key']][selected].tolist(),z[k0][selected].tolist(),z[k1][selected].tolist()));need=set(zip(outfid[mask].tolist(),result[mask,3].tolist(),result[mask,4].tolist()));assert not need-have,(sid,list(need-have)[:3])
 validation.append({'source_id':int(sid),'rows':int(mask.sum()),'missing_identity_pairs_or_triples':0})
output=out/'I328-canonical-face-lineage.npz';data={k:result[:,i] for i,k in enumerate(fields)};data.update(face_source_original_id=outorig,face_source_face_id=outfid,legacy_face_source_original_id=orig,legacy_face_source_face_id=fid,source_cache_rows=np.arange(n,dtype=np.int64),source_cache_sha256=np.asarray([sha(cp)]),temporary_cap_descendant_face_mask=outorig==capid,face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True)]));np.savez_compressed(output,**data)
receipt={'status':'PASS','scope':'metadata crosswalk only; not geometry or physical qualification','input':asset(cp),'output':asset(output),'input_faces':n,'buffer_derived_faces':len(rows),'temporary_caps':int(np.count_nonzero(outorig==capid)),'ambiguous_inherited_identity_pairs':len(amb),'missing_parent_rows':len(missing),'validated_sources':validation,'new_operation_source_id_reserved':120000200};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='validated_sources'},indent=2))
