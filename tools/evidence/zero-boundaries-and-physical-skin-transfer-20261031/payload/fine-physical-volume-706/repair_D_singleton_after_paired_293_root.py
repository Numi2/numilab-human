#!/usr/bin/env python3
"""Bounded connected D-surface repair of joint native-apex obstructions; fixed F32 coordinates."""
from pathlib import Path
import numpy as np,json,hashlib,argparse,sys,importlib.util,itertools,time,collections
H=Path(__file__).resolve().parent;S=H/'fixed-boundary-quality-topology-196'
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
base=module('base197for262',S/'run_fixed_boundary_dp_batch_197.py')
ears=module('joint_ears283for294',H/'joint_native_lazy_ears_283_root.py')
J=module('joint272for276',H/'audit_joint_native_apex_272_root.py')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 inputs={
  'O':H/'joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz',
  'I':H/'joint-native-OI-ears-279-root/I-batch002-run001/typed-280-002/candidate-I-typed-current.npz',
  'D':H/'paired-inferred-envelope-resolution-reduction-293-root/candidate-D-paired-resolution-reduced.npz'}
 expected={'O':'4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20',
  'I':'d1f897d88dddbf07bcf9c48388bd38ebed3423cc3be0ee1242f7816a76cb3c66',
  'D':'9a66be5a75d097647d8ff6c0b8a7ee447faa687f2434d146709d6ae5fa316ff5'}
 for r,p in inputs.items(): assert sha(p)==expected[r],(r,sha(p),expected[r])
 arrays={r:base.load(p) for r,p in inputs.items()};pts={r:np.asarray(z['points_world_m'],np.float32) for r,z in arrays.items()};faces={r:np.asarray(z['faces'],np.int64) for r,z in arrays.items()}
 pred=base.module('exact042for262',base.OWNER).load_exact_owner();assert sha(Path(pred.__file__))==base.OWNER_IMPL_SHA
 helper=base.module('helper037for262',base.HELPER);disk=base.module('disk031for262',base.DISK);bone=base.module('bone609for262',base.BONE);filtermod=base.module('filter030for262',base.FILTER)
 pbase={};chunks=[];cursor=0
 for r in ('D','O','I'):pbase[r]=cursor;chunks.append(pts[r]);cursor+=len(pts[r])
 combined=np.concatenate(chunks);records=[];facebase=base.FACE_BASE
 for r in ('D','O','I'):
  raw,_=pred.float32_surface_records(pts[r].astype(float),faces[r],source_seam_groups=());assert len(raw)==len(faces[r])
  records.extend((z[0],z[1],z[2],facebase[r]+i,tuple(pbase[r]+int(v) for v in z[4])) for i,z in enumerate(raw))
 lookup={}
 for r in pts:
  for i,p in enumerate(pts[r]):
   key=p.tobytes();assert key not in lookup;lookup[key]=(r,i)
 for vol,marker,roles,wbase in (('Skin',3,('O','I'),900000),('Fat',6,('I','D'),901000)):
  path,pin=base.SHELL[vol];assert sha(path)==pin;z=base.load(path);sf=z['faces'][z['face_marker']==marker];mp={}
  for i in np.unique(sf):
   r,j=lookup[np.asarray(z['points_world_m'][i],np.float32).tobytes()];assert r in roles;mp[int(i)]=pbase[r]+j
  mapped=np.asarray([[mp[int(v)] for v in f] for f in sf],np.int64);raw,_=pred.float32_surface_records(combined.astype(float),mapped,source_seam_groups=());assert len(raw)==90
  records.extend((z[0],z[1],z[2],wbase+i,tuple(map(int,z[4]))) for i,z in enumerate(raw))
 bones,br=bone.load_bone_barrier_records(pred,root=H.parent,vertex_offset=len(combined),expected_native_sha256=base.BONE_NATIVE,expected_manifest_sha256=base.BONE_MANIFEST);assert len(bones)==938272
 index=pred._prepare_surface_aabb(records+bones);allowed={facebase[r]+i for r in faces for i in range(len(faces[r]))}
 filtered=filtermod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=())
 role='D';P=pts[role];F=faces[role];q=J.score(P,F);bad=np.flatnonzero(q<J.GATE)
 protected=set()
 for v in json.loads(base.PROT.read_text())['surface_routes']:
  if v['role']==role:
   hits=np.flatnonzero(np.all(P==np.asarray(v['target_world_m_f32'],np.float32),axis=1));assert len(hits)==1;protected.add(int(hits[0]))
 edgeindex=helper.edgeindex(F,len(P));vv=F.ravel();fr=np.repeat(np.arange(len(F)),3);o=np.argsort(vv,kind='stable');u,off,cnt=np.unique(vv[o],return_index=True,return_counts=True);adj={int(v):fr[o[off[i]:off[i]+cnt[i]]] for i,v in enumerate(u)}
 lattice=tuple(pred.float32_point_lattice_key(v) for v in P);proxy=helper.Proxy(filtered,pbase[role],len(P))
 accepted=[];failed=[];retired=set();used=set();previous=[];nextid=5000000
 deadline=tic+a.total_seconds;budget_exhausted=False
 # One authenticated six-face singleton disk from D282, rebased through the paired-retirement map.
 maps=base.load(H/'paired-inferred-envelope-resolution-reduction-293-root/paired-retirement-maps.npz')
 parent_rows=[119513,119514,238473,238572,238573,238574]
 rows=np.asarray([int(maps['D_parent282_face_to_output_row'][r]) for r in parent_rows],np.int64)
 if np.any(rows<0) or len(np.unique(rows))!=6:raise RuntimeError('singleton patch row-map missing or duplicated')
 seed=int(maps['D_parent282_face_to_output_row'][119514])
 if int(J.score(P,F)[seed])>=J.GATE:raise RuntimeError('authenticated singleton seed no longer has a joint-apex obstruction')
 loop=helper.patchloop(F,rows)
 if loop is None or len(loop)!=4:raise RuntimeError(f'expected closed four-vertex singleton disk, got {loop}')
 verts=set(map(int,F[rows].ravel()))
 if verts&protected:raise RuntimeError(f'singleton patch touches protected surface points {sorted(verts&protected)}')
 candidates=[(len(rows),len(loop),seed,tuple(map(int,rows)),loop,verts)]
 seen=set()
 for _,__,seed,rowkey,loop,verts in sorted(candidates,key=lambda z:(z[0],z[1],z[2],z[3])):
  if time.monotonic()>=deadline:budget_exhausted=True;break
  if rowkey in seen:continue
  seen.add(rowkey);rows=np.asarray(rowkey,np.int64)
  if retired.intersection(rowkey) or used&verts:continue
  if q[seed]>=J.GATE:continue
  filtered.set_excluded({facebase[role]+i for i in retired.union(rowkey)})
  ext=helper.external(loop,edgeindex,rows,retired)
  try:
   remaining=deadline-time.monotonic()
   if remaining<=0.1:budget_exhausted=True;break
   nf,recs,detail=ears.collision_aware_lazy_ears(loop,P,ext,index,proxy,filtered,lattice,previous,nextid,disk,max_nodes=30000,max_seconds=min(6.0,remaining),minimum_altitude_m=20e-9,max_solutions=64,quality_floor=a.surface_floor)
   nf=np.asarray(nf,np.int64);nq=J.score(P,nf)
   before=int(sum(q[rows]<J.GATE));after=int(sum(nq<J.GATE))
   assert after<before and np.all(nq>=a.surface_floor), 'native-apex/floor reduction gate'
   qa,qg=helper.stableq(P,nf);assert np.all(qa>=20e-9) and np.all(qg>=0)
   extgate=filtered._audit_pair_prepared_first(index,recs,same_surface=True);selfgate=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True);assert extgate['count']==selfgate['count']==0
   op={'role':role,'seed_face':seed,'retired_parent_face_rows':rows.tolist(),'parent_face_rows':rows.tolist(),'new_faces':nf.tolist(),'typed_parent_union':base.union_tuples(arrays[role],rows),'source_face_ancestry_union':base.ancestry_union(arrays[role],rows),'q_before':q[rows].tolist(),'q_after':nq.tolist(),'native_apex_obstructions_before':before,'native_apex_obstructions_after':after,'exact_changed_vs_complete_pairs':0,'exact_new_internal_and_previous_pairs':0,'vertices_unchanged':True,'ear_search':detail}
   op['typed_parent_union_sha256']=base.union_digest(op['typed_parent_union']);accepted.append(op);retired.update(rowkey);used.update(verts);previous.extend(recs);nextid+=20000
   print(json.dumps({'stage':'accepted','seed':seed,'retired_faces':len(rows),'new_faces':len(nf),'removed_impossible':before-after}),flush=True)
  except Exception as e:failed.append({'seed':seed,'rows':rows.tolist(),'error':repr(e)})
 filtered.set_excluded({facebase[role]+i for i in retired})
 joint=filtered._audit_pair_prepared_first(index,previous,same_surface=True);js=pred._audit_record_pairs(itertools.combinations(previous,2),same_surface=True);assert joint['count']==js['count']==0
 keep=np.flatnonzero(~np.isin(np.arange(len(F)),np.asarray(sorted(retired),np.int64)));new=np.asarray([f for op in accepted for f in op['new_faces']],np.int64).reshape(-1,3);G=np.concatenate((F[keep],new));fm=np.concatenate((keep,np.full(len(new),-1,np.int64)))
 b0,t0=base.topology(F);b1,t1=base.topology(G);assert np.array_equal(b0,b1) and not t1['nonmanifold_edges'] and not t1['same_direction_edges'];assert len(np.unique(np.sort(G,axis=1),axis=0))==len(G)
 unions=[];offs=[0];patchids=[];cursor=len(keep)
 for j,op in enumerate(accepted):
  n=len(op['new_faces']);op['new_output_face_rows']=list(range(cursor,cursor+n));cursor+=n
  for f in op['new_faces']:unions.extend(op['typed_parent_union']);offs.append(len(unions));patchids.append('native-apex-singleton-after-paired-293-D-'+str(j))
 U=np.asarray(unions,np.int64).reshape(-1,7);path=a.output/'candidate-D-native-apex-connected-raw.npz'
 np.savez_compressed(path,points_world_m=arrays[role]['points_world_m'],points_world_m_f32=P,faces=G,point_output_to_parent_rows=np.arange(len(P),dtype=np.int64),face_output_to_parent_rows=fm,retired_parent_face_rows=np.asarray(sorted(retired),np.int64),new_faces_parent_point_rows=new,new_face_patch_union_offsets=np.asarray(offs,np.int64),new_output_face_patch_id=np.asarray(patchids),immediate_parent_candidate_sha256=np.asarray([sha(inputs[role])]),**{base.UNION[k].replace('quality_star_patch_union_face_source_','new_face_patch_union_'):U[:,k] for k in range(7)})
 qq=J.score(P,G)
 report={'status':'PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL' if accepted else 'NO_QUALIFIED_PROGRESS','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'ear_helper_sha256':sha(Path(ears.__file__)),'joint_bound_owner_sha256':sha(Path(J.__file__)),'quality_metric':'joint_volume_shape_apex_upper_bound','construction_surface_floor':a.surface_floor,'native_gate_unchanged':True,'all_point_arrays_unchanged':True,'joint_complete_surface_walls_all185_pairs':joint['count'],'joint_changed_self_pairs':js['count'],'accepted_operations':accepted,'failed_attempts':failed,'outputs':{'D':{'path':str(path),'sha256':sha(path),'parent':str(inputs[role]),'parent_sha256':sha(inputs[role]),'no_native_apex_before':len(bad),'no_native_apex_after':int(sum(qq<J.GATE)),'floor_bad_after':int(sum(qq<a.surface_floor))}},'paired_retirement_input':{'receipt_path':str(H/'paired-inferred-envelope-resolution-reduction-293-root/receipt.json'),'receipt_sha256':sha(H/'paired-inferred-envelope-resolution-reduction-293-root/receipt.json'),'maps_path':str(H/'paired-inferred-envelope-resolution-reduction-293-root/paired-retirement-maps.npz'),'maps_sha256':sha(H/'paired-inferred-envelope-resolution-reduction-293-root/paired-retirement-maps.npz'),'singleton_D282_parent_face_rows':parent_rows,'singleton_D0_parent_face_rows':rows.tolist()},'context_inputs':{r:{'path':str(p),'sha256':sha(p)} for r,p in inputs.items()},'elapsed_s':time.monotonic()-tic,'total_time_budget_s':a.total_seconds,'time_budget_exhausted':budget_exhausted}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('accepted_operations','failed_attempts')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--surface-floor',type=float,default=1.1e-5);p.add_argument('--total-seconds',type=float,default=90.0);run(p.parse_args())
