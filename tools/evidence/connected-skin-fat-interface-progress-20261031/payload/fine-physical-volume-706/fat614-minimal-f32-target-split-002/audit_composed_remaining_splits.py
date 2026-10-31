from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,sys,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OWNER_SCRIPT=H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py'
PARENT602=H/'fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz'
PARENT602_SHA='602036c9b164bd1eec83b344675b8cabf453665920e45b558b50fe8115c67493'
SPLIT=H/'fat614-minimal-f32-target-split-002/run-005/candidate-D-remaining-yy-target-edge-splits.npz'
SPLIT_SHA='4ce01a4bd397e7cf2b3c469ee33a3937d9173ff7d585a2687a1a09cf98377ab2'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def f32(a,label):
 p=np.asarray(a.get('points_world_m_f32',np.asarray(a['points_world_m']).astype(np.float32)))
 if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError(f'{label} malformed F32 points')
 return p
def same(a,b):
 x,y=np.asarray(a),np.asarray(b);return x.shape==y.shape and x.dtype==y.dtype and np.ascontiguousarray(x).tobytes()==np.ascontiguousarray(y).tobytes()
def boundary(f):
 ed={}
 for t in np.asarray(f,np.int64):
  for u,v in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):ed.setdefault(tuple(sorted((u,v))),[]).append((u,v))
 out=set()
 for k,d in ed.items():
  if len(d)==1:out.add(d[0])
  elif len(d)==2:
   if d[0]!=d[1][::-1]:raise ValueError(f'same-direction interior edge {k}')
  else:raise ValueError(f'nonmanifold edge {k} incidence={len(d)}')
 return out
def hjson(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def orientkey(t):
 a,b,c=map(int,t);return min((a,b,c),(b,c,a),(c,a,b))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 tic=time.monotonic()
 current={
 'O':(H/'joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz','4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20'),
 'I':(H/'joint-native-OI-ears-279-root/I-batch002-run001/typed-280-002/candidate-I-typed-current.npz','d1f897d88dddbf07bcf9c48388bd38ebed3423cc3be0ee1242f7816a76cb3c66'),
 'D':(H/'native-apex-D-singleton-after-paired-293-root/run-001/typed-001/candidate-D-typed-current.npz','214f888227a38da66bbbbeff70932927f393f9d6b34b1a2690f328a01b6a4dca')}
 base={}; paths={}
 for r,(p,want) in current.items():
  if sha(p)!=want:raise ValueError(f'{r} baseline hash changed')
  paths[r]=p;base[r]=load(p)
 if sha(SPLIT)!=SPLIT_SHA:raise ValueError('split candidate hash mismatch')
 if sha(PARENT602)!=PARENT602_SHA:raise ValueError('intermediate 602036 candidate hash mismatch')
 mid=load(PARENT602);cand=load(SPLIT);P0={r:f32(base[r],r) for r in ('D','O','I')};Pmid=f32(mid,'602036');P1=f32(cand,'candidate D')
 F0=np.asarray(base['D']['faces'],np.int64);Fmid=np.asarray(mid['faces'],np.int64);F1=np.asarray(cand['faces'],np.int64)
 pmap_i=np.asarray(cand.get('point_output_to_parent_rows'),np.int64);fmap_i=np.asarray(cand.get('face_output_to_parent_rows'),np.int64)
 if pmap_i.shape!=(len(P1),) or fmap_i.shape!=(len(F1),):raise ValueError('immediate row map dimensions invalid')
 if not np.array_equal(np.asarray(cand['immediate_parent_candidate_sha256']).reshape(-1),np.asarray([sha(PARENT602)])):raise ValueError('candidate immediate 602 parent pin mismatch')
 pmap_mid=np.asarray(mid['point_output_to_parent_rows'],np.int64);fmap_mid=np.asarray(mid['face_output_to_parent_rows'],np.int64)
 if len(pmap_mid)!=len(Pmid) or len(fmap_mid)!=len(Fmid):raise ValueError('602 parent map dimensions invalid')
 if not np.array_equal(np.asarray(mid['immediate_parent_candidate_sha256']).reshape(-1),np.asarray([sha(paths['D'])])):raise ValueError('602 candidate does not directly parent the pinned final D baseline')
 if not np.array_equal(pmap_i,np.r_[np.arange(len(Pmid),dtype=np.int64),np.full(len(P1)-len(Pmid),-1,np.int64)]):raise ValueError('new candidate point map is not identity plus appended points')
 if np.count_nonzero(pmap_i<0)!=9 or np.count_nonzero(fmap_i<0)!=32:raise ValueError('immediate split batch row delta dimensions changed')
 if not np.array_equal(fmap_i[:len(Fmid)][np.setdiff1d(np.arange(len(Fmid)),np.flatnonzero(fmap_i[:len(Fmid)]<0))],np.setdiff1d(np.arange(len(Fmid)),np.flatnonzero(fmap_i[:len(Fmid)]<0))):raise ValueError('immediate retained face map changed identity')
 pmap=np.full(len(P1),-1,np.int64);pkeep=np.flatnonzero(pmap_i>=0);pmap[pkeep]=pmap_mid[pmap_i[pkeep]]
 fmap=np.full(len(F1),-1,np.int64);fkeep=np.flatnonzero(fmap_i>=0);fmap[fkeep]=fmap_mid[fmap_i[fkeep]]
 newrows=np.flatnonzero(pmap<0);oldrows=np.flatnonzero(pmap>=0)
 if len(newrows)!=11 or not np.array_equal(newrows,np.arange(len(P0['D']),len(P1))):raise ValueError('expected two 602036 plus nine new appended D points')
 if not np.array_equal(pmap[oldrows],np.arange(len(P0['D']),dtype=np.int64)):raise ValueError('retained D points do not map in exact baseline prefix order')
 if not np.array_equal(P1[oldrows].view(np.uint32),P0['D'].view(np.uint32)):raise ValueError('retained D F32 point bits changed')
 if not np.array_equal(np.asarray(cand['points_world_m'])[:len(P0['D'])].view(np.uint64),np.asarray(base['D']['points_world_m']).view(np.uint64)):raise ValueError('retained D F64 point bytes changed')
 if not np.array_equal(np.asarray(cand['points_world_m'])[newrows].view(np.uint64),P1[newrows].astype(np.float64).view(np.uint64)):raise ValueError('new D points are not widened F32')
 if np.any(np.all(P1[newrows,None,:].view(np.uint32)==np.concatenate([P0['D'],P0['O'],P0['I']],axis=0)[None,:,:].view(np.uint32),axis=2)):raise ValueError('new point duplicates a current O/I/D point')
 changed=np.flatnonzero(fmap<0);old_face_rows=fmap[fmap>=0];retired=np.flatnonzero(~np.isin(np.arange(len(F0)),old_face_rows))
 declared_retired=np.asarray(cand.get('retired_parent_face_rows'),np.int64);declared_changed=np.asarray(cand.get('changed_parent_face_rows'),np.int64)
 if len(retired)!=18 or len(changed)!=40:raise ValueError(f'composed face delta dimensions differ: retired={len(retired)} changed={len(changed)}')
 if not np.array_equal(np.sort(declared_retired),retired):raise ValueError('composed retired baseline faces differ from declared retirement set')
 if not np.array_equal(np.sort(declared_changed),changed):raise ValueError('composed changed output faces differ from declared changed set')
 if np.any(fmap < -1) or np.any(fmap>=len(F0)):raise ValueError('composed face parent map out of range')
 if not np.array_equal(fmap[old_face_rows],old_face_rows):raise ValueError('retained D face rows do not retain exact baseline identity')
 adds_path=Path(str(np.asarray(cand['face_source_identity_schema_additions_json']).item()))
 if sha(adds_path)!=str(np.asarray(cand['face_source_identity_schema_additions_sha256']).item()):raise ValueError('identity additions JSON hash mismatch')
 adds=json.loads(adds_path.read_text())
 prev_add_path=Path(str(np.asarray(cand['face_source_identity_schema_additions_parent_path']).item()))
 if sha(prev_add_path)!=str(np.asarray(cand['face_source_identity_schema_additions_parent_sha256']).item()):raise ValueError('parent identity additions hash mismatch')
 if str(prev_add_path)!=str(np.asarray(mid['face_source_identity_schema_additions_json']).item()):raise ValueError('candidate additions do not extend the exact 602036 registry')
 if sha(prev_add_path)!=str(np.asarray(mid['face_source_identity_schema_additions_sha256']).item()):raise ValueError('602036 registry pin mismatch')
 # Validate every derived face in the composed delta against its immutable baseline parent and full typed union.
 lanes=['layer','body','bone','face_id','original_id','identity0','identity1'];uoff=np.asarray(cand['quality_star_patch_union_offsets'],np.int64);auoff=np.asarray(base['D']['quality_star_patch_union_offsets'],np.int64);aoff=np.asarray(cand['face_ancestry_offsets'],np.int64);baseaoff=np.asarray(base['D']['face_ancestry_offsets'],np.int64);anc_fields=['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']
 source_ids=np.asarray(cand['face_source_original_id'],np.int64);changed_set=set(map(int,changed));synthetic_ids=set(int(source_ids[r]) for r in changed)
 if not synthetic_ids.issubset(set(map(int,adds['additions'].keys()))):raise ValueError('changed derived face lacks registry addition')
 if len(synthetic_ids)!=18:raise ValueError('expected 18 face-level derived source identities')
 for sid in sorted(synthetic_ids):
  entry=adds['additions'][str(sid)];parent=int(entry['parent_face_row'])
  if not 0<=parent<len(F0):raise ValueError(f'synthetic ID {sid} parent row is outside baseline D')
  mid_additions=json.loads(prev_add_path.read_text())['additions']
  if str(sid) in mid_additions:
   if int(mid_additions[str(sid)]['parent_face_row'])!=parent:raise ValueError(f'synthetic ID {sid} inherited parent row changed')
  elif int(fmap_mid[parent])!=parent:raise ValueError(f'new synthetic ID {sid} does not refer to a retained baseline parent face')
  rows=np.flatnonzero(source_ids==sid)
  if not set(map(int,rows)).issubset(changed_set):raise ValueError(f'synthetic ID {sid} appears outside the declared composed face delta')
  lo,hi=map(int,auoff[parent:parent+2]);parent_union=(sorted(set(tuple(map(int,row)) for row in np.column_stack([base['D'][f'quality_star_patch_union_face_source_{lane}'][lo:hi] for lane in lanes]))) if hi>lo else [tuple(int(base['D'][f'face_source_{lane}'][parent]) for lane in lanes)])
  declared_union=sorted(set(tuple(map(int,row)) for row in entry['complete_typed_parent_union']))
  if declared_union!=parent_union:raise ValueError(f'synthetic ID {sid} registry union differs from baseline typed parent')
  if len(parent_union)!=int(entry['parent_union_tuple_count']) or hjson([list(t) for t in parent_union])!=str(entry['parent_union_tuple_sha256']):raise ValueError(f'synthetic ID {sid} parent union digest/count mismatch')
  alo,ahi=map(int,baseaoff[parent:parent+2]);parent_anc=sorted(set(tuple(int(base['D'][k][j]) for k in anc_fields) for j in range(alo,ahi)))
  tri_vertices=set(map(int,F0[parent]))
  for r in rows:
   lo,hi=map(int,uoff[r:r+2]);child_union=[tuple(int(cand[f'quality_star_patch_union_face_source_{lane}'][j]) for lane in lanes) for j in range(lo,hi)]
   if child_union!=parent_union:raise ValueError(f'synthetic ID {sid} child row {r} loses full typed parent union')
   alo,ahi=map(int,aoff[r:r+2]);child_anc=sorted(set(tuple(int(cand[k][j]) for k in anc_fields) for j in range(alo,ahi)))
   if child_anc!=parent_anc:raise ValueError(f'synthetic ID {sid} child row {r} loses full source-face ancestry union')
   if int(cand['face_source_face_id'][r])!=-1 or int(cand['face_source_identity0'][r])!=int(r):raise ValueError(f'synthetic ID {sid} face identity tuple malformed')
   if (int(cand['face_source_layer'][r]),int(cand['face_source_body'][r]),int(cand['face_source_bone'][r]))!=(int(base['D']['face_source_layer'][parent]),int(base['D']['face_source_body'][parent]),int(base['D']['face_source_bone'][parent])):raise ValueError(f'synthetic ID {sid} scalar source descriptors changed from baseline parent')
   if not set(map(int,F1[r])).issubset(tri_vertices|set(map(int,newrows))):raise ValueError(f'synthetic ID {sid} output face is outside its baseline parent/inserted-point support')
  id1=np.unique(np.asarray(cand['face_source_identity1'])[rows])
  if len(id1)!=1:raise ValueError(f'synthetic ID {sid} has inconsistent patch identity')
  if len(rows)<2:raise ValueError(f'synthetic ID {sid} does not identify a split parent patch')
 # All rows that lost exact baseline identity must be registry-backed derived faces.
 if not set(map(int,changed)).issubset(set(np.flatnonzero(np.isin(source_ids,list(synthetic_ids))))):raise ValueError('changed face rows include unregistered face identities')
 # Shared exact-owner namespace: all original role points retain baseline IDs; new D points receive unique fresh IDs.
 pointbase={};cursor=0
 for r in ('D','O','I'):pointbase[r]=cursor;cursor+=len(P0[r])
 new_pointbase=cursor;extra=P1[newrows];d_face_point_map=np.r_[np.arange(len(P0['D']),dtype=np.int64),new_pointbase+np.arange(len(newrows),dtype=np.int64)]
 mapped_candidate_D=d_face_point_map[F1]
 if boundary(F0)!=boundary(mapped_candidate_D):raise ValueError('D directed boundary signature differs from full composed baseline')
 # Load the existing exact owner and assemble full current O/I/D + both walls + all185 context.
 base_mod=module('split_exact_owner_197',H/'fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py')
 pred=base_mod.module('quality_owner_loader_198',base_mod.OWNER).load_exact_owner()
 if sha(Path(pred.__file__))!=base_mod.OWNER_IMPL_SHA:raise ValueError('exact predicate implementation hash changed')
 bone_mod=base_mod.module('bone_barrier_198',base_mod.BONE);filter_mod=base_mod.module('filtered_exact_owner_198',base_mod.FILTER)
 role_order=('D','O','I');pointbase={};cursor=0;parts=[]
 for r in role_order:pointbase[r]=cursor;parts.append(P0[r]);cursor+=len(P0[r])
 new_pointbase=cursor;extra=P1[newrows];parts.append(extra);combined_points=np.concatenate(parts,axis=0)
 facebase=base_mod.FACE_BASE.copy();baseline_records=[]
 for r in role_order:
  raw,_=pred.float32_surface_records(P0[r].astype(np.float64),np.asarray(base[r]['faces'],np.int64),source_seam_groups=())
  if len(raw)!=len(base[r]['faces']):raise ValueError(f'{r}: baseline exact-owner face record drop')
  baseline_records.extend((q[0],q[1],q[2],facebase[r]+i,tuple(pointbase[r]+int(v) for v in q[4])) for i,q in enumerate(raw))
 # Map both unchanged 90-face walls to existing baseline role-local coordinates.
 lookup={}
 for r in role_order:
  for i,q in enumerate(P0[r]):
   k=np.ascontiguousarray(q,dtype='<f4').tobytes()
   if k in lookup:raise ValueError(f'baseline O/I/D cross-role point alias at {r}/{i}')
   lookup[k]=(r,i)
 wall_records=[];wall_receipts=[]
 for volume,marker,expected_roles in (('Skin',3,('O','I')),('Fat',6,('I','D'))):
  spath,expected_sha=base_mod.SHELL[volume]
  if sha(spath)!=expected_sha:raise ValueError(f'{volume} wall asset hash mismatch')
  sh=load(spath);sp=np.asarray(sh['points_world_m'],np.float32);sf=np.asarray(sh['faces'],np.int64);mk=np.asarray(sh['face_marker'],np.int64);rows=np.flatnonzero(mk==marker);remap=np.full(len(sp),-1,np.int64)
  for sr in np.unique(sf[rows]):
   hit=lookup.get(np.ascontiguousarray(sp[int(sr)],dtype='<f4').tobytes())
   if hit is None or hit[0] not in expected_roles:raise ValueError(f'{volume} wall point lacks expected current role binding')
   remap[int(sr)]=pointbase[hit[0]]+hit[1]
  wf=remap[sf[rows]]
  if np.any(wf<0):raise ValueError(f'{volume} wall mapping incomplete')
  raw,_=pred.float32_surface_records(combined_points.astype(np.float64),wf,source_seam_groups=())
  if len(raw)!=90:raise ValueError(f'{volume} expected exact wall count90')
  start={'Skin':900000,'Fat':901000}[volume]
  rr=[(q[0],q[1],q[2],start+i,tuple(map(int,q[4]))) for i,q in enumerate(raw)];wall_records.extend(rr);wall_receipts.append({'role':volume,'path':str(spath),'sha256':expected_sha,'face_count':len(rr)})
 bones,bone_receipt=bone_mod.load_bone_barrier_records(pred,root=base_mod.ROOT,vertex_offset=len(combined_points),expected_native_sha256=base_mod.BONE_NATIVE,expected_manifest_sha256=base_mod.BONE_MANIFEST)
 if len(bones)!=938272:raise ValueError('all185 barrier record count mismatch')
 context=baseline_records+wall_records+list(bones)
 ids=[int(q[3]) for q in context]
 if len(set(ids))!=len(ids):raise ValueError('context face record ID collision')
 exact_index=pred._prepare_surface_aabb(context)
 allowed={facebase[r]+i for r in role_order for i in range(len(base[r]['faces']))}
 excluded={facebase['D']+int(r) for r in retired}
 filtered=filter_mod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=excluded)
 raw,_=pred.float32_surface_records(P1.astype(np.float64),F1[changed],source_seam_groups=())
 if len(raw)!=len(changed):raise ValueError('exact owner dropped a changed candidate face')
 changed_records=[];idbase=max(ids)+1000000
 for i,q in enumerate(raw):
  verts=[]
  for v in q[4]:
   v=int(v)
   if v<len(P0['D']):verts.append(pointbase['D']+v)
   else:verts.append(new_pointbase+(v-len(P0['D'])))
  changed_records.append((q[0],q[1],q[2],idbase+i,tuple(verts)))
 against=filtered._audit_pair_prepared_first(exact_index,changed_records,same_surface=True)
 selfpairs=pred._audit_record_pairs(itertools.combinations(changed_records,2),same_surface=True)
 # Exact joint native-apex necessary-bound check on all eight split triangles.
 jmod=module('split_joint_apex_272',H/'audit_joint_native_apex_272_root.py')
 scores=jmod.score(P1,F1[changed]);bad_exact=[];ip=jmod.M.integer_points(P1)
 for i,row in enumerate(changed):
  reason,gap=jmod.exact_terms(ip,F1[int(row)])
  if reason:bad_exact.append({'face_row':int(row),'reason':reason,'gap':str(gap)})
 zero=against.get('count')==0 and selfpairs.get('count')==0 and not bad_exact
 out.mkdir(parents=True)
 report={'schema':'h706.fat614-minimally-rounded-d-edge-split-complete-audit.v1','status':'PASS_EXACT_CHANGED_D_VS_COMPLETE_OID_WALLS_ALL185_AND_APEX' if zero else 'FAIL_EXACT_CHANGED_D_VS_COMPLETE_OID_WALLS_ALL185_OR_APEX','physical_admission':False,'native_steps':0,'candidate_D':{'path':str(SPLIT),'sha256':sha(SPLIT),'immediate_parent_path':str(PARENT602),'immediate_parent_sha256':sha(PARENT602),'baseline_parent_path':str(paths['D']),'baseline_parent_sha256':sha(paths['D'])},'current_surface_baselines':{r:{'path':str(paths[r]),'sha256':sha(paths[r])} for r in role_order},'point_map':{'retained_D_prefix_exact':True,'appended_D_rows':newrows.tolist(),'appended_volume_global_indices_in_audit_namespace':list(range(new_pointbase,len(combined_points))),'new_points_bits':[ [f'0x{int(y):08x}' for y in row] for row in extra.view(np.uint32)]},'changed_D_face_rows':changed.tolist(),'retired_D_parent_face_rows':retired.tolist(),'changed_vs_complete_current_OID_both_walls_all185':against,'changed_D_joint_self_pairs':selfpairs,'joint_native_apex':{'scores':scores.tolist(),'minimum':float(np.min(scores)),'all_exact_terms_pass':not bad_exact,'exact_failures':bad_exact,'native_gate_unchanged':True},'wall_receipts':wall_receipts,'all185_receipt':bone_receipt,'exact_owner':{'path':str(Path(pred.__file__)),'sha256':sha(Path(pred.__file__)),'expected_sha256':base_mod.OWNER_IMPL_SHA},'elapsed_s':time.monotonic()-tic,'limitation':'Only changed D faces were collision-tested against complete retained current O/I/D, both wall assets, and all185; no volume remesh/positive-cell/mass/native admission is established.'}
 rp=out/'report.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n')
 print(json.dumps({'status':report['status'],'against_count':against.get('count'),'self_count':selfpairs.get('count'),'apex_min':report['joint_native_apex']['minimum'],'apex_failures':len(bad_exact),'report_path':str(rp),'report_sha256':sha(rp)},indent=2))
if __name__=='__main__':main()
