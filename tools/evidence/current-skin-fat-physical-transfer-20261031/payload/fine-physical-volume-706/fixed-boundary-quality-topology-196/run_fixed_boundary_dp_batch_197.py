#!/usr/bin/env python3
"""One-index bounded connected-disk retriangulation for native-fixed-face witnesses.
Coordinates are immutable. This produces raw geometry proposals only; volume remeshing,
full typed metadata, mass mapping, and native admission remain separate gates.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,itertools,json,sys,time,traceback
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
ROOT=H.parent
P=H/'fixed-boundary-quality-topology-196'
W=P/'all-1855-impossible-fixed-boundary-face-witnesses.npz'; W_SHA='7816cacfa5eec5814c319a3597a780ed47aa775ae88954f8b5090292d51a9215'
WREC=P/'receipt.json'; WREC_SHA='52694683b2b93a38e0fa88d404d2dfe440f9ce999c2eb3f1c2524ab16b2d166d'
SRC_CLASS=P/'geometry-source-classification.json'
PROT=H/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json'; PROT_SHA='ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71'
CAND={
 'O':(H/'quality-warning-connected-union-038-typed-metadata-005-001/O-merged-run004-run005-001/typed-output/candidate-O-connected-union-typed.npz','b35eb99aab790d861263b6dffa605f781287d81d682e3fb34dac8810d2c92b66'),
 'I':(H/'quality-warning-connected-union-038-typed-metadata-005-001/I-run-002/candidate-I-connected-union-typed.npz','a445e482f83616bd863e194377896d09d98327ff474683f5f687a4ac52d059f7'),
 'D':(H/'fat-inferred-boundary-D-refinement-128-v2/candidate-D-inferred-boundary-refined-typed.npz','ff73291bce77dda81eff985fecdbba1b198f24e95a7f8f9af1d6058172d4da45'),
}
SOURCE={
 'O':(H/'microedge-csr-canonical-011/O/candidate-O-csr-canonical.npz','578cee97a3bbecb73c0e1a6d7c26f45ac6cc111a552996f3910866a4a3652a9b'),
 'I':(H/'local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz','9721102247157eeb42c59633fda59d63fd5ba48ad11ba4e63987834136711e31'),
 'D':(H/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz','15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f'),
}
SHELL={
 'Skin':(H/'complete-typed-shell-owner-074/skin-shell.npz','e18930bf18863cb97b4d98590a7c25d0d043c3e6f4f783676dac196ae812d55a'),
 'Fat':(H/'fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz','65a13549560123b31ecbc698680edaff48ab843e834ef127e72f59d6aaffc04a'),
}
HELPER=H/'quality-warning-connected-union-037/remesh_warning_union_037.py'; HELPER_SHA='f6daea315a8ec7aa150c8171924682f54968b19950359f66328de4597d47f2b3'
OWNER=ROOT/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'; OWNER_SHA='f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad'; OWNER_IMPL_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
BONE=ROOT/'bone-barrier-remeshing-609/bone_barrier.py'; BONE_SHA='734527f060dcae186ae1b9609eca1ab95768d952aa0e9e8c7af76cae882b236b'; BONE_NATIVE='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5'; BONE_MANIFEST='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c'
DISK=H/'quality-disk-eligibility-031/constraint_disk_quality.py'; DISK_SHA='38e7a1ab5caa9044b86de931555590061339ed11388a84bf302ee78eeb53e97f'
CHORD=ROOT/'exact-chord-pruning-618/chord_pruning.py'; CHORD_SHA='c59f57487dc180db31f8820db23b7e355d69585de3b363eb39c95edaadedbebc'
FILTER=H/'quality-exact-filter-adapter-030/filtered_prepared_exact_owner.py'; FILTER_SHA='c3f427432096719b9fe3b91cbd2fc1a6f779228c03456cef85dc8bbbae9da16d'
FEASIBLE=H/'quality-disk-feasibility-vectorized-054/disk_feasibility.py'; FEASIBLE_SHA='6f889b1d631af51bcb186976fc78cddd546f81f2e858e88f982da05941413909'
FEASIBLE_REC=H/'quality-disk-feasibility-vectorized-054/receipt.json'; FEASIBLE_REC_SHA='35a4ba33e5d2c3d647645087baa6506544252a427dc4c8c9178b19a5d3a4b2fe'
FACE_BASE={'D':0,'O':300000,'I':600000}; ALT=20e-9; ANGLE=0.1; MAX_FACES=240; MAX_LOOP=22; PATCH_BUDGET=6.0; MAX_PATCHES_PER_ROLE=1800; MAX_TOTAL_SECONDS=300.0
TYPED=('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')
UNION=('quality_star_patch_union_face_source_layer','quality_star_patch_union_face_source_body','quality_star_patch_union_face_source_bone','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_original_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise RuntimeError(f'cannot import {path}')
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def qbound(points,faces):
 tri=np.asarray(points,np.float32)[np.asarray(faces,np.int64)].astype(np.float64)
 a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];c=tri[:,2]-tri[:,1]
 area2=np.linalg.norm(np.cross(a,b),axis=1);long=np.maximum.reduce((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)))
 return np.divide(area2,long*long,out=np.zeros_like(area2),where=long>0)
def union_tuples(A,rows):
 vals=set()
 for row in map(int,rows):
  if 'quality_star_patch_union_offsets' in A and all(k in A for k in UNION):
   off=A['quality_star_patch_union_offsets'];lo,hi=map(int,(off[row],off[row+1]))
  else:lo=hi=0
  if hi>lo:
   for j in range(lo,hi):vals.add(tuple(int(A[k][j]) for k in UNION))
  else:vals.add(tuple(int(A[k][row]) for k in TYPED))
 return sorted(vals)
def ancestry_union(A,rows):
 out=set()
 keys=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows')
 if 'face_ancestry_offsets' not in A or any(k not in A for k in keys):return []
 off=A['face_ancestry_offsets']
 for row in map(int,rows):
  for j in range(int(off[row]),int(off[row+1])):out.add(tuple(int(A[k][j]) for k in keys))
 return [list(x) for x in sorted(out)]
def union_digest(vals):
 payload=json.dumps([list(map(int,x)) for x in vals],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf8')
 return hashlib.sha256(payload).hexdigest()
def topology(f):
 f=np.asarray(f,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));s=np.sort(e,axis=1);u,inv,c=np.unique(s,axis=0,return_inverse=True,return_counts=True);fw=e[:,0]==s[:,0];fc=np.bincount(inv[fw],minlength=len(u));rc=np.bincount(inv[~fw],minlength=len(u));return u[c==1],{'faces':int(len(f)),'boundary_edges':int(np.count_nonzero(c==1)),'nonmanifold_edges':int(np.count_nonzero(c>2)),'same_direction_edges':int(np.count_nonzero((c==2)&((fc!=1)|(rc!=1))))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 t0=time.monotonic();deadline=t0+MAX_TOTAL_SECONDS;status={'stage':'start','patch_proposals':0,'dp_attempts':0,'accepted':0,'failed':0}
 def progress(stage,**extra):
  status['stage']=stage;d={'schema':'h706.fixed-boundary-dp-196.progress.v1','elapsed_s':time.monotonic()-t0,'deadline_s':MAX_TOTAL_SECONDS,'counters':dict(status),**extra};(out/'progress.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(json.dumps(d),flush=True)
 def check_budget(stage):
  if time.monotonic()>=deadline:raise TimeoutError('bounded total DP wall time reached at '+stage)
 try:
  pins=[(W,W_SHA,'1855 witness'),(WREC,WREC_SHA,'witness receipt'),(PROT,PROT_SHA,'protected149'),(HELPER,HELPER_SHA,'037 helpers'),(OWNER,OWNER_SHA,'exact owner loader'),(BONE,BONE_SHA,'bone helper'),(DISK,DISK_SHA,'DP disk'),(CHORD,CHORD_SHA,'chord pruning'),(FILTER,FILTER_SHA,'exact filter'),(FEASIBLE,FEASIBLE_SHA,'disk feasibility'),(FEASIBLE_REC,FEASIBLE_REC_SHA,'disk feasibility receipt')]
  for r,(p,h) in CAND.items():pins.append((p,h,'candidate '+r))
  for r,(p,h) in SOURCE.items():pins.append((p,h,'015 source '+r))
  for r,(p,h) in SHELL.items():pins.append((p,h,'shell '+r))
  for p,h,n in pins:
   got=sha(p)
   if got!=h:raise ValueError(f'{n} pin changed: {got} != {h}')
  if not SRC_CLASS.exists():raise ValueError('source classification report missing')
  check_budget('pinned inputs')
  W0=load(W); current={r:load(CAND[r][0]) for r in ('O','I','D')}; source={r:load(SOURCE[r][0]) for r in ('O','I','D')}
  points={r:np.asarray(current[r]['points_world_m'],np.float32) for r in current}; p64={r:np.asarray(current[r]['points_world_m'],np.float64) for r in current}; faces={r:np.asarray(current[r]['faces'],np.int64) for r in current}
  # Input role domains must be complete and exact to their pinned boundary witnesses.
  _wr=np.asarray(W0['surface_role']).astype(str);_fr=np.asarray(W0['candidate_face_rows'],np.int64);
  if [len(np.unique(_fr[_wr==r])) for r in ('O','I','D')]!=[18,55,1727]:raise ValueError('unique witness counts differ')
  protected={r:set() for r in faces};pr=json.loads(PROT.read_text())
  for rec in pr['surface_routes']:
   role=rec['role'];xyz=np.asarray(rec['target_world_m_f32'],np.float32);hits=np.flatnonzero(np.all(points[role]==xyz[None,:],axis=1))
   if len(hits)!=1:raise ValueError(f'{role} route F32 coordinate maps {len(hits)} current rows')
   protected[role].add(int(hits[0]))
  witness_protected=np.asarray(W0['protected_rows_by_surface_role'],np.int64);wo=np.asarray(W0['protected_rows_by_surface_role_offsets'],np.int64)
  for i,r in enumerate(('O','I','D')):
   if sorted(protected[r])!=sorted(map(int,witness_protected[wo[i]:wo[i+1]])):raise ValueError(f'{r} protected route row mismatch to witness')
  # Import the already-tested owner algorithms; no copied exact predicate or remesher.
  h=module('quality_warning_helpers_037_196',HELPER);pred=module('quality_owner_042_196',OWNER).load_exact_owner()
  if sha(Path(pred.__file__))!=OWNER_IMPL_SHA:raise ValueError('exact predicate implementation SHA mismatch')
  bone_mod=module('bone_barrier_609_196',BONE);disk_mod=module('disk_eligibility_031_196',DISK);chord_mod=module('chord_pruning_618_196',CHORD);filter_mod=module('filtered_exact_owner_030_196',FILTER);feas_mod=module('disk_feasibility_054_196',FEASIBLE)
  if json.loads(FEASIBLE_REC.read_text()).get('status')!='PASS':raise ValueError('disk feasibility receipt not PASS')
  check_budget('helpers loaded')
  # Build a shared point namespace D,O,I and add both authenticated 90-face rim walls.
  role_order=('D','O','I');pbase={};cursor=0;combined=[]
  for r in role_order:pbase[r]=cursor;combined.append(points[r]);cursor+=len(points[r])
  combined_points=np.concatenate(combined,axis=0);facebase=dict(FACE_BASE)
  surf=[];edgeidx={};lattice={};initial_boundary={};before_quality={}
  for r in role_order:
   F=faces[r];raw,_=pred.float32_surface_records(np.asarray(points[r],np.float64),F,source_seam_groups=())
   if len(raw)!=len(F):raise ValueError(f'{r} exact record count drop')
   surf.extend((q[0],q[1],q[2],facebase[r]+i,tuple(pbase[r]+int(v) for v in q[4])) for i,q in enumerate(raw))
   edgeidx[r]=h.edgeindex(F,len(points[r]));lattice[r]=tuple(pred.float32_point_lattice_key(x) for x in points[r])
   initial_boundary[r]=h.boundary(F);before_quality[r]=qbound(points[r],F)
  pointlookup={}
  for r in role_order:
   for i,x in enumerate(points[r]):
    k=np.ascontiguousarray(x).tobytes()
    if k in pointlookup:raise ValueError(f'cross-role duplicate exact point coordinate {r}/{i}')
    pointlookup[k]=(r,i)
  wall_receipts=[];wall_facebase={'Skin':900000,'Fat':901000}
  for vol,shpath,marker,roles in (('Skin',SHELL['Skin'][0],3,('O','I')),('Fat',SHELL['Fat'][0],6,('I','D'))):
   z=load(shpath);sp=np.asarray(z['points_world_m'],np.float32);sf=np.asarray(z['faces'],np.int64);mk=np.asarray(z['face_marker'],np.int64);fr=np.flatnonzero(mk==marker);used=np.unique(sf[fr]);rowmap=np.full(len(sp),-1,np.int64)
   for sr in used:
    hit=pointlookup.get(np.ascontiguousarray(sp[int(sr)]).tobytes())
    if hit is None or hit[0] not in roles:raise ValueError(f'{vol} wall vertex shell row {sr} not uniquely in expected current roles')
    rowmap[int(sr)]=pbase[hit[0]]+int(hit[1])
   wf=rowmap[sf[fr]]
   if np.any(wf<0):raise ValueError(f'{vol} wall point map incomplete')
   wr,_=pred.float32_surface_records(np.asarray(combined_points,np.float64),wf,source_seam_groups=())
   if len(wr)!=len(wf):raise ValueError(f'{vol} wall exact record count')
   surf.extend((q[0],q[1],q[2],wall_facebase[vol]+i,tuple(map(int,q[4]))) for i,q in enumerate(wr))
   wall_receipts.append({'volume_role':vol,'shell_path':str(shpath),'shell_sha256':SHELL[vol][1],'wall_marker':marker,'wall_face_count':len(fr),'exact_wall_faces_indexed':len(wr),'source_roles':list(roles)})
  bone_records,bone_receipt=bone_mod.load_bone_barrier_records(pred,root=ROOT,vertex_offset=len(combined_points),expected_native_sha256=BONE_NATIVE,expected_manifest_sha256=BONE_MANIFEST)
  if len(bone_records)!=938272:raise ValueError('all185 record count mismatch')
  progress('RECORDS_READY',role_surface_records=len(surf),wall_records=sum(x['exact_wall_faces_indexed'] for x in wall_receipts),all185_records=len(bone_records),wall_receipts=wall_receipts)
  exact_index=pred._prepare_surface_aabb(surf+bone_records);check_budget('complete exact index built')
  allowed={facebase[r]+i for r in faces for i in range(len(faces[r]))}
  filtered=filter_mod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=())
  # Build bounded, disjoint one-ring disk candidates from every current impossible face.
  candidates={r:[] for r in faces}
  for r in ('O','I','D'):
   F=faces[r];vv=np.concatenate((F[:,0],F[:,1],F[:,2]));fr=np.tile(np.arange(len(F),dtype=np.int64),3);oo=np.argsort(vv,kind='mergesort');sv=vv[oo];uv,off,cnt=np.unique(sv,return_index=True,return_counts=True);adj={int(v):fr[oo[off[i]:off[i]+cnt[i]]] for i,v in enumerate(uv)}
   wids=np.flatnonzero(np.asarray(W0['surface_role']).astype(str)==r);wrows=np.asarray(W0['candidate_face_rows'],np.int64)[wids];wrats=np.asarray(W0['bound_ratio'],np.float64)[wids]
   patchseen=set()
   for wi,seed,ratio in zip(wids,wrows,wrats):
    tri=F[int(seed)];rows=np.unique(np.concatenate([adj[int(v)] for v in tri])).astype(np.int64)
    if len(rows)>MAX_FACES:continue
    verts=set(map(int,F[rows].reshape(-1)))
    if verts & protected[r]:continue
    loop=h.patchloop(F,rows)
    if loop is None or len(loop)>MAX_LOOP:continue
    key=tuple(map(int,rows))
    if key in patchseen:continue
    patchseen.add(key)
    union=union_tuples(current[r],rows)
    groups={(int(t[0]),int(t[1])) for t in union}
    if len(groups)!=1:continue
    bone_set={int(t[2]) for t in union}
    candidates[r].append({'role':r,'seed_face':int(seed),'witness_index':int(wi),'bound_ratio':float(ratio),'rows':rows,'loop':list(map(int,loop)),'vertices':verts,'typed_union':union,'union_sha256':union_digest(union),'source_face_ancestry':ancestry_union(current[r],rows),'parent_layer_body':list(next(iter(groups))),'parent_bones':sorted(bone_set)})
   candidates[r].sort(key=lambda c:(len(c['rows']),len(c['loop']),c['bound_ratio'],c['seed_face']))
  forced={'D':128922,'O':63608,'I':160951};selected=[];selected_by={r:0 for r in faces};used_rows={r:set() for r in faces};used_verts={r:set() for r in faces}
  for r in ('D','O','I'):
   hit=next((c for c in candidates[r] if c['seed_face']==forced[r]),None)
   if hit is None:raise ValueError(f'forced seed {r}:{forced[r]} lost eligible homogeneous disk preflight')
   selected.append(hit);selected_by[r]+=1;used_rows[r].update(map(int,hit['rows']));used_verts[r].update(hit['vertices'])
  # Greedy maximal disjoint batch over all eligible witnesses in deterministic order.
  # One shared exact index and a single joint gate; no full-body rebuild per patch.
  for r in ("D","O","I"): 
   for c in candidates[r]:
    if selected_by[r]>=MAX_PATCHES_PER_ROLE:break
    if (set(map(int,c["rows"]))&used_rows[r]) or (c["vertices"]&used_verts[r]):continue
    selected.append(c);selected_by[r]+=1;used_rows[r].update(map(int,c["rows"]));used_verts[r].update(c["vertices"])
  status['patch_proposals']=len(selected)
  progress('DISJOINT_PATCHES_SELECTED',selected=[{'role':c['role'],'seed_face':c['seed_face'],'patch_face_count':len(c['rows']),'loop_vertex_count':len(c['loop']),'ratio':c['bound_ratio'],'source_patch_group':c['parent_layer_body'],'parent_bones':c['parent_bones']} for c in selected],selected_by_role=selected_by)
  accepted=[];failed=[];retired={r:set() for r in faces};accepted_records=[];next_id=3000000
  for idx,c in enumerate(selected):
   if time.monotonic()>=deadline:failed.append({'role':c['role'],'seed_face':c['seed_face'],'status':'TOTAL_WALL_BUDGET_REACHED'});break
   r=c['role'];F=faces[r];P32=points[r];rows=np.asarray(c['rows'],np.int64);loop=c['loop'];proxy=h.Proxy(filtered,pbase[r],len(points[r]))
   excl={facebase[q]+int(x) for q in faces for x in retired[q]};excl.update(facebase[r]+int(x) for x in rows);filtered.set_excluded(excl)
   ext=h.external(loop,edgeidx[r],rows,retired[r]);chord_report=None
   try:
    check_budget('patch chord pruning')
    chords,chord_report=chord_mod.prune_chords(loop,lattice[r],exact_index,proxy,max_seconds=2.0,cache={})
    if chord_report.get('blocked_boundary_edges'):raise ValueError('blocked patch boundary edge')
    newfaces,recs,dp=disk_mod.collision_aware_dp_disk(loop,P32,P32,ext|set(chords),exact_index,proxy,lattice[r],accepted_records,next_id,max_seconds=PATCH_BUDGET,max_branches=512,chunk_size=64,minimum_altitude_m=ALT,minimum_angle_degrees=ANGLE)
    newfaces=np.asarray(newfaces,np.int64).reshape((-1,3));qa,qg=h.stableq(P32,newfaces);afterbound=qbound(P32,newfaces)
    if len(newfaces)!=len(loop)-2 or np.any(qa<ALT) or np.any(qg<ANGLE):raise ValueError('postquality/triangle count gate')
    if np.any(afterbound<1e-5):raise ValueError('new fixed face still violates the normalized-det necessary bound')
    local_before=qbound(P32,F[rows]);bad_before=int(np.count_nonzero(local_before<1e-5))
    if bad_before<1:raise ValueError('selected patch no longer contains a current impossible face')
    # Repeat the changed-vs-complete query through the canonical exact owner.
    changed_audit=filtered._audit_pair_prepared_first(exact_index,recs,same_surface=True)
    if changed_audit.get('count')!=0:raise ValueError('changed patch intersects retained O/I/D/rim/all185 records')
    internal=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
    if internal.get('count')!=0:raise ValueError('new faces intersect within patch')
    parentrows=set(map(int,rows));orig_map=np.asarray(current[r].get('quality_star_face_output_to_input_row',np.full(len(F),-1)),np.int64)
    origrows=set(map(int,orig_map[rows][orig_map[rows]>=0]));anc=c['source_face_ancestry']
    if np.any(np.asarray(current[r]['quality_star_point_output_to_input_row'],np.int64)[newfaces]<0):raise ValueError('new face uses point without immediate 015 ancestry')
    opid=f'196-{r}-fixed-boundary-dp-{idx:03d}-seed{int(c["seed_face"])}'
    record={'patch_id':opid,'role':r,'seed_current_face_row':int(c['seed_face']),'witness_bound_ratio_before':float(c['bound_ratio']),'retired_parent_face_rows':rows.astype(int).tolist(),'retired_source015_face_rows':sorted(origrows),'loop_current_point_rows':loop,'new_faces_current_point_rows':newfaces.astype(int).tolist(),'new_faces_source015_point_rows':np.asarray(current[r]['quality_star_point_output_to_input_row'],np.int64)[newfaces].astype(int).tolist(),'complete_typed_parent_union':c['typed_union'],'typed_parent_union_sha256':c['union_sha256'],'parent_layer_body':c['parent_layer_body'],'parent_bone_values':c['parent_bones'],'source_face_ancestry_union':anc,'old_patch_impossible_face_count':bad_before,'new_patch_impossible_face_count':int(np.count_nonzero(afterbound<1e-5)),'minimum_new_face_bound_ratio':float(afterbound.min()),'minimum_new_face_altitude_m':float(qa.min()),'minimum_new_face_angle_degrees':float(qg.min()),'exact_changed_vs_complete_surface_and_all185_pairs':int(changed_audit['count']),'exact_changed_aabb_candidates':int(changed_audit.get('aabb_candidate_pairs',0)),'exact_patch_self_pairs':int(internal['count']),'exact_dp':dp,'exact_chord_pruning':chord_report,'coordinates_changed':False}
    accepted.append({'role':r,'candidate':c,'newfaces':newfaces,'records':recs,'report':record,'record_id_start':next_id})
    retired[r].update(map(int,rows));accepted_records.extend(recs);next_id+=20000
    status['accepted']=len(accepted);status['dp_attempts']=idx+1
    progress('PATCH_ACCEPTED',patch=record,accepted=len(accepted),failed=len(failed))
   except Exception as exc:
    failed.append({'role':r,'seed_face':int(c['seed_face']),'status':'PATCH_NOT_ACCEPTED','error':repr(exc),'patch_face_rows':rows.astype(int).tolist(),'loop_vertex_rows':loop,'chord_pruning':chord_report})
    status['failed']=len(failed);status['dp_attempts']=idx+1
    progress('PATCH_FAILED_CONTINUING',role=r,seed_face=int(c['seed_face']),error=repr(exc),accepted=len(accepted),failed=len(failed))
  if not accepted:raise RuntimeError('no fixed-boundary disk patch passed the bounded exact trial')
  # Verify accepted changed faces jointly, after excluding all retired parent rows.
  excl={facebase[q]+int(x) for q in faces for x in retired[q]};filtered.set_excluded(excl)
  joint_audit=filtered._audit_pair_prepared_first(exact_index,accepted_records,same_surface=True)
  joint_self=pred._audit_record_pairs(itertools.combinations(accepted_records,2),same_surface=True)
  if joint_audit.get('count')!=0 or joint_self.get('count')!=0:raise ValueError('joint accepted batch exact gate')
  outroles={};output_faces={};quality_after={};patch_union_entries=[]
  for r in ('O','I','D'):
   F=faces[r];parent_ret=np.asarray(sorted(retired[r]),np.int64);keep=np.flatnonzero(~np.isin(np.arange(len(F),dtype=np.int64),parent_ret));patches=[x for x in accepted if x['role']==r];newf=np.concatenate([x['newfaces'] for x in patches],axis=0) if patches else np.empty((0,3),np.int64);fo=np.concatenate((F[keep],newf),axis=0);output_faces[r]=fo
   b0=np.sort(h.boundary(F),axis=1);b1=np.sort(h.boundary(fo),axis=1)
   if not np.array_equal(b0,b1):raise ValueError(f'{r} unoriented boundary edge set changed')
   st_before=h.patchloop(F,[]) if False else None
   _,topo=topology(fo)
   if topo['nonmanifold_edges'] or topo['same_direction_edges']:raise ValueError(f'{r} output topology invalid: {topo}')
   qbefore=qbound(points[r],F);qafter=qbound(points[r],fo)
   quality_after[r]={'input_faces':int(len(F)),'output_faces':int(len(fo)),'input_impossible_fixed_face_count':int(np.count_nonzero(qbefore<1e-5)),'output_impossible_fixed_face_count':int(np.count_nonzero(qafter<1e-5)),'minimum_input_bound_ratio':float(qbefore.min()),'minimum_output_bound_ratio':float(qafter.min()),'boundary_edge_count':int(topo['boundary_edges']),'nonmanifold_edges':topo['nonmanifold_edges'],'same_direction_edges':topo['same_direction_edges']}
   if patches and quality_after[r]['output_impossible_fixed_face_count']>=quality_after[r]['input_impossible_fixed_face_count']:raise ValueError(f'{r} batch failed to reduce impossible fixed-boundary face count')
   if not np.array_equal(p64[r],np.asarray(current[r]['points_world_m'],np.float64)) or not np.array_equal(points[r],np.asarray(current[r]['points_world_m'],np.float32)):raise ValueError(f'{r} point coordinates changed')
   source_face_map=np.asarray(current[r].get('quality_star_face_output_to_input_row',np.full(len(F),-1)),np.int64)
   source_point_map=np.asarray(current[r]['quality_star_point_output_to_input_row'],np.int64)
   faceoutparent=np.concatenate((keep,np.full(len(newf),-1,np.int64)))
   faceoutsource=np.concatenate((source_face_map[keep],np.full(len(newf),-1,np.int64)))
   new_input_points=source_point_map[newf] if len(newf) else np.empty((0,3),np.int64)
   if len(newf) and np.any(new_input_points<0):raise ValueError(f'{r} added face point source ancestry missing')
   new_union_rows=[];new_union_offsets=[0];new_patchids=[];new_patchseeds=[];new_patch_output_rows=[]
   cursor=len(keep)
   for q in patches:
    un=q['candidate']['typed_union'];n=len(q['newfaces']);start=cursor;rowsout=list(range(cursor,cursor+n));cursor+=n
    for _ in range(n):new_union_rows.extend(un);new_union_offsets.append(len(new_union_rows))
    new_patchids.extend([q['report']['patch_id']]*n);new_patchseeds.extend([q['report']['seed_current_face_row']]*n);new_patch_output_rows.append({'patch_id':q['report']['patch_id'],'output_face_rows':rowsout,'retired_parent_face_rows':q['report']['retired_parent_face_rows']})
    q['report']['new_output_face_rows']=rowsout
   unarray=np.asarray(new_union_rows,np.int64).reshape((-1,7)) if new_union_rows else np.empty((0,7),np.int64)
   union_digest_lanes={UNION[j]:unarray[:,j] for j in range(7)}
   output=out/f'candidate-{r}-fixed-boundary-dp-raw.npz'
   np.savez_compressed(output,points_world_m=p64[r],points_world_m_f32=points[r],faces=fo,source_candidate_sha256=np.asarray([SOURCE[r][1]]),immediate_parent_candidate_sha256=np.asarray([CAND[r][1]]),point_output_to_parent_rows=np.arange(len(points[r]),dtype=np.int64),point_output_to_input_rows=source_point_map,face_output_to_parent_rows=faceoutparent,face_output_to_input_rows=faceoutsource,retired_parent_face_rows=parent_ret,retired_input_face_rows=np.unique(source_face_map[parent_ret][source_face_map[parent_ret]>=0]).astype(np.int64),new_faces_parent_point_rows=newf,new_faces_input_point_rows=new_input_points,new_face_patch_union_offsets=np.asarray(new_union_offsets,np.int64),**{k.replace('quality_star_patch_union_face_source_','new_face_patch_union_'):v for k,v in union_digest_lanes.items()},new_output_face_patch_id=np.asarray(new_patchids,dtype='U80'),new_output_face_seed_parent_row=np.asarray(new_patchseeds,np.int64))
   outroles[r]={'path':str(output),'sha256':sha(output),'input_candidate_path':str(CAND[r][0]),'input_candidate_sha256':CAND[r][1],'source_015_path':str(SOURCE[r][0]),'source_015_sha256':SOURCE[r][1],'point_count':len(points[r]),'input_face_count':len(F),'output_face_count':len(fo),'retired_parent_face_count':len(parent_ret),'new_face_count':len(newf),'output_patch_maps':new_patch_output_rows,'quality':quality_after[r],'protected_target_count':len(protected[r]),'protected_target_rows_unchanged':True}
  # Per-operation documents include the exact full patch-level parent union, not a fake one-face overlap.
  for q in accepted:
   r=q['role'];op=q['report'];op['schema']='h706.fixed-boundary-quality-dp-operation.v1';op['lineage_semantics']='complete deduplicated seven-field parent union for the consumed disk patch; not per-triangle geometric overlap';op['input_candidate_path']=str(CAND[r][0]);op['input_candidate_sha256']=CAND[r][1];op['source_015_path']=str(SOURCE[r][0]);op['source_015_sha256']=SOURCE[r][1];op['physical_admission']=False;op['native_steps']=0
   opath=out/(op['patch_id']+'.json');opath.write_text(json.dumps(op,indent=2,sort_keys=True)+'\n');op['operation_json_path']=str(opath);op['operation_json_sha256']=sha(opath)
  if len(accepted_records)!=sum(len(q['newfaces']) for q in accepted):raise ValueError('accepted record census mismatch')
  final={'schema':'h706.fixed-boundary-quality-dp-batch-196.v1','status':'PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL','physical_admission':False,'native_steps':0,'coordinates_changed':False,'point_arrays_unchanged':True,'quality_gate':'normalized_det >= 1e-5; boundary face necessary ratio is 2*area/longest_edge^2','input_witness':{'path':str(W),'sha256':W_SHA,'receipt_path':str(WREC),'receipt_sha256':WREC_SHA,'unique_surface_witness_counts':{'O':18,'I':55,'D':1727},'volume_role_face_incidences':{'Skin':73,'Fat':1782},'protected_bad_face_vertices':0,'all_role_incidence_count':1855},'input_candidates':{r:{'path':str(CAND[r][0]),'sha256':CAND[r][1]} for r in CAND},'source015':{r:{'path':str(SOURCE[r][0]),'sha256':SOURCE[r][1]} for r in SOURCE},'protected149':{'path':str(PROT),'sha256':PROT_SHA,'routes_by_role':{r:len(protected[r]) for r in protected},'all_target_points_and_rows_preserved':True},'current_exact_index':{'predicate_owner_path':str(OWNER),'predicate_owner_loader_sha256':OWNER_SHA,'predicate_implementation_sha256':OWNER_IMPL_SHA,'role_surface_records':sum(len(faces[r]) for r in faces),'authenticated_skin_fat_rim_walls':wall_receipts,'all185_bone_records':len(bone_records),'bone_record_receipt':bone_receipt},'selected_patch_count':len(selected),'selected_patches':[{'role':c['role'],'seed_face':c['seed_face'],'witness_ratio':c['bound_ratio'],'input_patch_face_count':len(c['rows']),'loop_vertex_count':len(c['loop']),'input_parent_layer_body':c['parent_layer_body'],'input_parent_bones':c['parent_bones'],'parent_patch_rows':c['rows'].astype(int).tolist()} for c in selected],'accepted_patch_count':len(accepted),'failed_patch_count':len(failed),'accepted_operations':[q['report'] for q in accepted],'failed_candidates':failed,'joint_changed_faces_vs_complete_OID_walls_all185':{'pair_count':int(joint_audit['count']),'aabb_candidates':int(joint_audit.get('aabb_candidate_pairs',0)),'exact_query_records':int(joint_audit.get('complete_query_records',0)),'face_id_ordering_filter':joint_audit.get('face_id_ordering_filter'),'joint_new_face_self_pairs':int(joint_self['count'])},'quality_after':quality_after,'outputs':outroles,'scope':'Exact source-prepared topology proposals only. Current volume cells were not retetrahedralized; any surface topology change requires a new matching PLC/volume mesh and full downstream audits. No mass, native, runtime, or physical admission claim.','elapsed_seconds':time.monotonic()-t0,'wall_budget_seconds':MAX_TOTAL_SECONDS}
  (out/'report.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n');progress('COMPLETE',status=final['status'],accepted=len(accepted),failed=len(failed),outputs=outroles)
  print(json.dumps({'status':final['status'],'out':str(out),'accepted':len(accepted),'failed':len(failed),'quality_after':quality_after,'joint_pairs':joint_audit['count'],'elapsed':final['elapsed_seconds']},indent=2),flush=True)
 except BaseException as exc:
  (out/'failure.json').write_text(json.dumps({'schema':'h706.fixed-boundary-quality-dp-batch-196.failure.v1','status':'BOUNDED_PROPOSAL_FAILED_OR_INCOMPLETE','error':repr(exc),'traceback':traceback.format_exc(),'counters':status,'elapsed_s':time.monotonic()-t0,'physical_admission':False,'native_steps':0},indent=2,sort_keys=True)+'\n');raise
if __name__=='__main__':main()
