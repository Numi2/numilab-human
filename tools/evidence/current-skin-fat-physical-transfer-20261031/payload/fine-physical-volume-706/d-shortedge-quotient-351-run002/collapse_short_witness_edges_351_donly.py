#!/usr/bin/env python3
"""Bounded exact short-edge contractions for fixed-boundary quality witnesses.
Raw topology proposals only; all point coordinates remain byte-identical. Each
collapse is link-checked, source-union recorded, and exact-audited against one
shared O/I/D + paired-wall + all185 index before batch composition.
"""
from pathlib import Path
import hashlib, importlib.util, itertools, json, sys, time, traceback
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
P=H/'fixed-boundary-quality-topology-196'
BASE_DRIVER=P/'run_fixed_boundary_dp_batch_197.py'
ROLE_ORDER=('O','I','D')
MAX_EDGE_M=2.5e-4
MAX_ACCEPTED=2000
MAX_TRIALS=100000
MAX_SECONDS=300.0
MIN_QBOUND=1.0e-5
MIN_ALTITUDE_M=20e-9
MIN_ANGLE_DEG=0.0
QBOUND_POLICY='all_new_faces_pass'
BATCH_ID='unset'
INPUT={}
TYPED={}
LINEAGE_LAYERS=[]
QUALITY_PREPASS=None
QUALITY_PREPASS_SHA=''
BACK_SCOPE_REPORT=None
BACK_SCOPE_REPORT_SHA=''
EXPECTED={}
AUTHORIZED_EDGE_PATH=None
AUTHORIZED_EDGE_SHA=''
CONFIG_PATH=None
CONFIG_SHA=''
UNION_LANES=('layer','body','bone','face_id','original_id','identity0','identity1')
TYPED_KEYS=tuple('face_source_'+k for k in UNION_LANES)
RAW_UNION_KEYS=tuple('new_face_patch_union_'+k for k in UNION_LANES)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 import argparse
 ap=argparse.ArgumentParser(description='Parameterized exact short-edge contraction driver')
 ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 global BATCH_ID,INPUT,TYPED,LINEAGE_LAYERS,QUALITY_PREPASS,QUALITY_PREPASS_SHA,BACK_SCOPE_REPORT,BACK_SCOPE_REPORT_SHA,EXPECTED
 global MAX_EDGE_M,MAX_ACCEPTED,MAX_TRIALS,MAX_SECONDS,MIN_QBOUND,MIN_ALTITUDE_M,MIN_ANGLE_DEG,QBOUND_POLICY,CONFIG_PATH,CONFIG_SHA
 global AUTHORIZED_EDGE_PATH,AUTHORIZED_EDGE_SHA
 CONFIG_PATH=Path(args.config).resolve();CONFIG_SHA=sha(CONFIG_PATH);cfg=json.loads(CONFIG_PATH.read_text())
 if cfg.get('schema')!='h706.fixed-boundary-short-edge-collapse-run-config.v1':raise ValueError('run config schema mismatch')
 BATCH_ID=str(cfg['batch_id']);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 def source_map(name):
  table=cfg[name]
  if set(table)!=set(ROLE_ORDER):raise ValueError(f'{name} must declare exactly O/I/D')
  return {r:(Path(table[r]['path']).resolve(),str(table[r]['sha256'])) for r in ROLE_ORDER}
 INPUT=source_map('current_candidates');TYPED=source_map('typed_ancestry_parents')
 LINEAGE_LAYERS=cfg['lineage_layers']
 if not LINEAGE_LAYERS:raise ValueError('lineage_layers cannot be empty')
 QUALITY_PREPASS=Path(cfg['quality_prepass_report']['path']).resolve();QUALITY_PREPASS_SHA=str(cfg['quality_prepass_report']['sha256'])
 BACK_SCOPE_REPORT=Path(cfg['back_scope_report']['path']).resolve();BACK_SCOPE_REPORT_SHA=str(cfg['back_scope_report']['sha256'])
 EXPECTED={r:int(cfg['expected_bad_face_counts'][r]) for r in ROLE_ORDER}
 auth=cfg.get('authorized_edge_map')
 authorized_edges_by_role={r:[] for r in ROLE_ORDER};authorized_orientations_by_role={r:set() for r in ROLE_ORDER}
 if auth:
  AUTHORIZED_EDGE_PATH=Path(auth['path']).resolve();AUTHORIZED_EDGE_SHA=str(auth['sha256'])
  if sha(AUTHORIZED_EDGE_PATH)!=AUTHORIZED_EDGE_SHA:raise ValueError('authorized D edge map SHA mismatch')
  amap=load(AUTHORIZED_EDGE_PATH)
  edges=np.asarray(amap['D_edge_rows'],np.int64);seeds=np.asarray(amap['D_seed_face_rows'],np.int64)
  orientations=np.asarray(amap['D_allowed_orientation_edge_rows'],np.int64)
  drops=np.asarray(amap['D_allowed_collapsed_point_rows'],np.int64);keeps=np.asarray(amap['D_allowed_retained_point_rows'],np.int64)
  if edges.ndim!=2 or edges.shape[1]!=2 or len(edges)!=len(seeds):raise ValueError('authorized edge/seed dimensions mismatch')
  if orientations.ndim!=2 or orientations.shape[1]!=2 or len(orientations)!=len(drops) or len(drops)!=len(keeps):raise ValueError('authorized orientation dimensions mismatch')
  if len({tuple(map(int,e)) for e in edges})!=len(edges):raise ValueError('duplicate authorized D edge')
  authorized_edges_by_role['D']=[(tuple(map(int,e)),int(fi)) for e,fi in zip(edges,seeds)]
  authorized_orientations_by_role['D']={(int(a),int(b)) for a,b in zip(drops,keeps)}
 params=cfg.get('parameters',{})
 MAX_EDGE_M=float(params.get('max_edge_length_m',MAX_EDGE_M));MAX_ACCEPTED=int(params.get('max_accepted',MAX_ACCEPTED));MAX_TRIALS=int(params.get('max_trials',MAX_TRIALS));MAX_SECONDS=float(params.get('wall_budget_seconds',MAX_SECONDS));MIN_QBOUND=float(params.get('normalized_det_minimum',MIN_QBOUND));MIN_ALTITUDE_M=float(params.get('minimum_altitude_m',MIN_ALTITUDE_M));MIN_ANGLE_DEG=float(params.get('minimum_angle_degrees',MIN_ANGLE_DEG));QBOUND_POLICY=str(params.get('replacement_qbound_policy','all_new_faces_pass'))
 if QBOUND_POLICY not in ('all_new_faces_pass','strict_local_net_reduction'):raise ValueError('unknown qbound construction policy')
 if BATCH_ID in ('','unset') or any(x<=0 for x in (MAX_EDGE_M,MAX_ACCEPTED,MAX_TRIALS,MAX_SECONDS,MIN_QBOUND,MIN_ALTITUDE_M)):raise ValueError('invalid batch id or nonpositive construction limits')
 t0=time.monotonic();deadline=t0+MAX_SECONDS
 status={'stage':'start','trials':0,'accepted':0,'rejected':0,'candidate_edges':0}
 failures={}
 phase_times={}
 def progress(stage,**extra):
  status['stage']=stage;d={'schema':'h706.fixed-boundary-short-edge-collapse-generic.progress.v1','batch_id':BATCH_ID,'elapsed_s':time.monotonic()-t0,'deadline_s':MAX_SECONDS,'counters':dict(status),'rejection_counts':dict(failures),**extra};(out/'progress.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(json.dumps(d),flush=True)
 def reject(code):failures[code]=failures.get(code,0)+1;status['rejected']+=1
 def check_budget():
  if time.monotonic()>=deadline:raise TimeoutError('bounded edge-collapse batch wall time reached')
 try:
  base=module('dp_batch_197_helpers_198',BASE_DRIVER)
  pins=[(CONFIG_PATH,CONFIG_SHA,'run configuration'),(QUALITY_PREPASS,QUALITY_PREPASS_SHA,'current qbound witness census'),(BACK_SCOPE_REPORT,BACK_SCOPE_REPORT_SHA,'back412 spatial scope report')]
  if AUTHORIZED_EDGE_PATH is not None:pins.append((AUTHORIZED_EDGE_PATH,AUTHORIZED_EDGE_SHA,'authorized D short-edge volume-preflight map'))
  for li,layer in enumerate(LINEAGE_LAYERS):
   rp=Path(layer['report']['path']).resolve();pins.append((rp,str(layer['report']['sha256']),f'lineage layer {li} report'))
   if set(layer['candidates'])!=set(ROLE_ORDER):raise ValueError(f'lineage layer {li} must declare exactly O/I/D candidates')
   for r in ROLE_ORDER:
    cand=layer['candidates'][r];pins.append((Path(cand['path']).resolve(),str(cand['sha256']),f'lineage layer {li} candidate {r}'))
  for r,(p,h) in INPUT.items():pins.append((p,h,'current input '+r))
  for r,(p,h) in TYPED.items():pins.append((p,h,'typed ancestry '+r))
  seenpins={}
  for p,h,n in pins:
   p=Path(p).resolve();h=str(h);oldpin=seenpins.get(str(p))
   if oldpin is not None and oldpin!=h:raise ValueError(f'conflicting SHA pins for {p}: {oldpin} vs {h}')
   seenpins[str(p)]=h;got=sha(p)
   if got!=h:raise ValueError(f'{n} SHA mismatch {got} != {h}')
  for r in ROLE_ORDER:
   cc=LINEAGE_LAYERS[0]['candidates'][r]
   if Path(cc['path']).resolve()!=INPUT[r][0] or str(cc['sha256'])!=INPUT[r][1]:raise ValueError(f'current input {r} differs from lineage layer 0')
  outarr={r:load(INPUT[r][0]) for r in ROLE_ORDER};typed={r:load(TYPED[r][0]) for r in ROLE_ORDER}
  points={r:np.asarray(outarr[r]['points_world_m_f32'],np.float32) for r in ROLE_ORDER};p64={r:np.asarray(outarr[r]['points_world_m'],np.float64) for r in ROLE_ORDER};faces={r:np.asarray(outarr[r]['faces'],np.int64) for r in ROLE_ORDER}
  point015={r:np.asarray(outarr[r]['point_output_to_input_rows'],np.int64) for r in ROLE_ORDER};parent_face={r:np.asarray(outarr[r]['face_output_to_parent_rows'],np.int64) for r in ROLE_ORDER};source_face={r:np.asarray(outarr[r]['face_output_to_input_rows'],np.int64) for r in ROLE_ORDER}
  for r in ROLE_ORDER:
   if len(points[r])!=len(p64[r]) or not np.array_equal(points[r],p64[r].astype(np.float32)):raise ValueError(f'{r} input F32/F64 point view mismatch')
   if len(faces[r])!=len(source_face[r]) or len(faces[r])!=len(parent_face[r]):raise ValueError(f'{r} face-map lengths differ')
  # Resolve all faces through immediate input maps, with complete op unions at each configured layer.
  layer_reports=[];layer_arrays=[];layer_ops=[];effective_parent_maps=[];crosswalk_rows=[];crosswalk_arrays={}
  for li,layer in enumerate(LINEAGE_LAYERS):
   rep=json.loads(Path(layer['report']['path']).read_text());layer_reports.append(rep)
   if rep.get('status') not in ('PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL','PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL'):
    raise ValueError(f'lineage layer {li} report status is not an accepted frozen proposal: {rep.get("status")}')
   arrs=outarr if li==0 else {r:load(Path(layer['candidates'][r]['path']).resolve()) for r in ROLE_ORDER}
   for r in ROLE_ORDER:
    declared=rep.get('outputs',{}).get(r,{})
    if str(Path(declared.get('path','')).resolve())!=str(Path(layer['candidates'][r]['path']).resolve()) or declared.get('sha256')!=layer['candidates'][r]['sha256']:
     raise ValueError(f'lineage layer {li} {r} candidate does not match its pinned report output')
    parent_sha=str(LINEAGE_LAYERS[li+1]['candidates'][r]['sha256']) if li+1<len(LINEAGE_LAYERS) else TYPED[r][1]
    immediate=np.asarray(arrs[r].get('immediate_parent_candidate_sha256',np.empty(0))).reshape(-1)
    if len(immediate)!=1 or str(immediate[0])!=parent_sha:raise ValueError(f'{r} layer {li} immediate-parent SHA does not match next configured layer')
   layer_arrays.append(arrs);unions={r:{} for r in ROLE_ORDER};ancestries={r:{} for r in ROLE_ORDER}
   for op in rep.get('accepted_operations',[]):
    rr=op['role']
    for row in op.get('new_output_face_rows',[]):
     unions[rr][int(row)]=op['complete_typed_parent_union'];ancestries[rr][int(row)]=op['source_face_ancestry_union']
   layer_ops.append({'union':unions,'ancestry':ancestries})
  # Some earlier batches intentionally kept inherited DP-created faces at -1 in
  # face_output_to_parent_rows. Resolve only those inherited rows through exact
  # oriented triangle equality after applying the authenticated output->parent
  # point map. Current-operation rows remain represented solely by their full
  # operation-union records. This is an exact row crosswalk, not a nearest match.
  for li,arrs in enumerate(layer_arrays):
   role_maps={}
   for rr in ROLE_ORDER:
    F=np.asarray(arrs[rr]['faces'],np.int64);rawmap=np.asarray(arrs[rr]['face_output_to_parent_rows'],np.int64)
    if len(rawmap)!=len(F):raise ValueError(f'{rr} ancestry layer {li} face map length mismatch')
    op_rows=set(layer_ops[li]['union'][rr])
    if op_rows!=set(layer_ops[li]['ancestry'][rr]):raise ValueError(f'{rr} layer {li} operation union/ancestry row sets differ')
    if any(row<0 or row>=len(F) for row in op_rows):raise ValueError(f'{rr} layer {li} current operation row outside output face domain')
    parent_arr=layer_arrays[li+1][rr] if li+1<len(layer_arrays) else typed[rr]
    parentF=np.asarray(parent_arr['faces'],np.int64)
    pointmap=np.asarray(arrs[rr]['point_output_to_parent_rows'],np.int64)
    if pointmap.ndim!=1 or np.any(pointmap<0) or np.any(pointmap>=len(parent_arr['points_world_m_f32'])):raise ValueError(f'{rr} layer {li} point output-to-parent map is incomplete/out of range')
    effective=np.full(len(F),-1,np.int64);inherited=[int(row) for row in range(len(F)) if int(row) not in op_rows]
    parent_lookup={}
    for prow,tri in enumerate(parentF):parent_lookup.setdefault(tuple(map(int,tri)),[]).append(prow)
    for frow in inherited:
     ptri=tuple(map(int,pointmap[F[frow]]))
     hits=parent_lookup.get(ptri,())
     if len(hits)!=1:raise ValueError(f'{rr} layer {li} inherited face {frow} has {len(hits)} exact oriented parent-triangle matches')
     effective[frow]=int(hits[0])
    mapped=np.flatnonzero(effective>=0)
    if not np.array_equal(F[mapped],parentF[effective[mapped]]):raise ValueError(f'{rr} layer {li} effective parent face map is not exact oriented triangle equality')
    if len(np.unique(effective[mapped]))!=len(mapped):raise ValueError(f'{rr} layer {li} multiple output faces map to one parent face')
    raw_valid=(rawmap>=0)&(rawmap<len(parentF));raw_match=int(np.count_nonzero(np.all(F[raw_valid]==parentF[rawmap[raw_valid]],axis=1))) if np.any(raw_valid) else 0
    role_maps[rr]=effective
    crosswalk_arrays[f"layer{li}_{rr}_output_to_parent_face_rows"]=effective
    crosswalk_rows.append({'layer':li,'role':rr,'input_faces':len(F),'immediate_parent_faces':len(parentF),'current_operation_faces':len(op_rows),'exact_oriented_parent_triangle_rows':len(inherited),'effective_mapped_faces':len(mapped),'raw_face_map_negative_rows':int(np.count_nonzero(rawmap<0)),'raw_face_map_out_of_range_rows':int(np.count_nonzero(rawmap>=len(parentF))),'raw_face_map_in_range_exact_matches':raw_match,'raw_face_map_interpretation':'retained unchanged from producer; this historical field is not used as the immediate face crosswalk where its row domain is inconsistent; effective map derives from exact oriented triangles through authenticated point map','effective_parent_map_sha256':hashlib.sha256(np.ascontiguousarray(effective).tobytes()).hexdigest()})
   effective_parent_maps.append(role_maps)
  crosswalk_npz=out/'lineage-crosswalk.npz';np.savez_compressed(crosswalk_npz,**crosswalk_arrays)
  crosswalk_doc={'schema':'h706.fixed-boundary-short-edge-lineage-crosswalk.v1','status':'PASS_EXACT_ORIENTED_TRIANGLE_ROW_CROSSWALK','meaning':'current operation faces use complete typed/source ancestry unions; inherited unmapped rows are linked only by exact oriented triangle equality through authenticated output-to-parent point maps','row_maps_path':str(crosswalk_npz),'row_maps_sha256':sha(crosswalk_npz),'layers':crosswalk_rows}
  (out/'lineage-crosswalk.json').write_text(json.dumps(crosswalk_doc,indent=2,sort_keys=True)+'\n')
  def resolve_lineage(rr,row,which):
   row=int(row)
   for li,arrs in enumerate(layer_arrays):
    opmap=layer_ops[li][which][rr]
    if row in opmap:return [tuple(map(int,x)) for x in opmap[row]]
    if li==len(layer_arrays)-1:
     src=int(effective_parent_maps[li][rr][row])
     if src<0:raise ValueError(f'{rr} DP-derived row {row} lacks operation {which}')
     if which=='union':return base.union_tuples(typed[rr],[src])
     return [tuple(map(int,x)) for x in base.ancestry_union(typed[rr],[src])]
    row=int(effective_parent_maps[li][rr][row])
    if row<0:raise ValueError(f'{rr} layer {li} row unexpectedly lacks exact parent crosswalk {which}')
   raise ValueError(f'{rr} lineage resolution exhausted for face {row}')
  def face_union(rr,row):return resolve_lineage(rr,int(row),'union')
  def face_ancestry(rr,row):return resolve_lineage(rr,int(row),'ancestry')
  back_scope=json.loads(BACK_SCOPE_REPORT.read_text())
  if back_scope.get('status')!='READ_ONLY_BACK412_SCOPE_AABB_AUDIT':raise ValueError('back412 AABB audit status mismatch')
  back_boxes=back_scope['back412_world_aabbs']
  protected={r:set() for r in ROLE_ORDER};prot=json.loads(base.PROT.read_text())
  if sha(base.PROT)!=base.PROT_SHA:raise ValueError('protected route source changed')
  for rec in prot['surface_routes']:
   r=rec['role'];xyz=np.asarray(rec['target_world_m_f32'],np.float32);hit=np.flatnonzero(np.all(points[r]==xyz[None,:],axis=1))
   if len(hit)!=1:raise ValueError(f'{r} protected point maps to {len(hit)} current rows')
   protected[r].add(int(hit[0]))
  # Recompute current normalized-det necessary witnesses and match the pinned 005 census before attempting contractions.
  bad={r:np.flatnonzero(base.qbound(points[r],faces[r])<MIN_QBOUND).astype(np.int64) for r in ROLE_ORDER}
  expected=EXPECTED
  qpre=json.loads(QUALITY_PREPASS.read_text())
  if qpre.get('status')!='READ_ONLY_CURRENT_QBOUND_WITNESS_CENSUS' or qpre.get('witness_counts')!=expected:raise ValueError('pinned quality witness receipt disagrees with configured counts')
  for r in ROLE_ORDER:
   qb=qpre.get('candidate_faces',{}).get(r,{})
   if str(Path(qb.get('path','')).resolve())!=str(INPUT[r][0]) or qb.get('sha256')!=INPUT[r][1]:raise ValueError(f'{r} qbound receipt does not pin current input candidate')
  if {r:len(bad[r]) for r in ROLE_ORDER}!=expected:raise ValueError(f'current witness census differs: { {r:len(bad[r]) for r in ROLE_ORDER} }')
  phase_times['input_ancestry_preflight_s']=time.monotonic()-t0
  progress('INPUT_AND_ANCESTRY_PREFLIGHT',current_faces={r:len(faces[r]) for r in ROLE_ORDER},mapped_to_immediate_parent={r:int(np.count_nonzero(parent_face[r]>=0)) for r in ROLE_ORDER},prior_derived_faces={r:int(np.count_nonzero(parent_face[r]<0)) for r in ROLE_ORDER},bad_face_counts={r:len(bad[r]) for r in ROLE_ORDER},back412_scope={'report_sha256':BACK_SCOPE_REPORT_SHA,'aabb_overlap_is_informational':True},phase_times=dict(phase_times))
  # Exact owner and complete unchanged context: current surfaces, both actual paired walls, all185.
  helper=base.module('quality_warning_helpers_198',base.HELPER);pred=base.module('quality_owner_loader_198',base.OWNER).load_exact_owner()
  if sha(Path(pred.__file__))!=base.OWNER_IMPL_SHA:raise ValueError('exact predicate implementation SHA changed')
  bone_mod=base.module('bone_barrier_198',base.BONE);filter_mod=base.module('filtered_exact_owner_198',base.FILTER)
  pbase={};cursor=0;combined=[];facebase=base.FACE_BASE.copy()
  for r in ('D','O','I'):pbase[r]=cursor;combined.append(points[r]);cursor+=len(points[r])
  combined_points=np.concatenate(combined,axis=0);surf=[];wall_receipts=[]
  for r in ('D','O','I'):
   raw,_=pred.float32_surface_records(np.asarray(points[r],np.float64),faces[r],source_seam_groups=())
   if len(raw)!=len(faces[r]):raise ValueError(f'{r} exact surface record dropped')
   surf.extend((q[0],q[1],q[2],facebase[r]+i,tuple(pbase[r]+int(v) for v in q[4])) for i,q in enumerate(raw))
  pointlookup={}
  for r in ('D','O','I'):
   for i,x in enumerate(points[r]):
    k=np.ascontiguousarray(x).tobytes()
    if k in pointlookup:raise ValueError(f'cross-role duplicate F32 vertex {r}/{i}')
    pointlookup[k]=(r,i)
  for vol,marker,roles in (('Skin',3,('O','I')),('Fat',6,('I','D'))):
   spath=base.SHELL[vol][0];sh=load(spath);sp=np.asarray(sh['points_world_m'],np.float32);sf=np.asarray(sh['faces'],np.int64);mk=np.asarray(sh['face_marker'],np.int64);wall=np.flatnonzero(mk==marker);used=np.unique(sf[wall]);rm=np.full(len(sp),-1,np.int64)
   for sr in used:
    hit=pointlookup.get(np.ascontiguousarray(sp[int(sr)]).tobytes())
    if hit is None or hit[0] not in roles:raise ValueError(f'{vol} wall row lacks current role map')
    rm[int(sr)]=pbase[hit[0]]+int(hit[1])
   wf=rm[sf[wall]]
   if np.any(wf<0):raise ValueError(f'{vol} wall vertex mapping incomplete')
   raw,_=pred.float32_surface_records(np.asarray(combined_points,np.float64),wf,source_seam_groups=())
   if len(raw)!=90:raise ValueError(f'{vol} wall exact record census !=90')
   wallbase={'Skin':900000,'Fat':901000}[vol]
   surf.extend((q[0],q[1],q[2],wallbase+i,tuple(map(int,q[4]))) for i,q in enumerate(raw))
   wall_receipts.append({'role':vol,'path':str(spath),'sha256':base.SHELL[vol][1],'marker':marker,'wall_faces':len(raw)})
  bones,bone_receipt=bone_mod.load_bone_barrier_records(pred,root=base.ROOT,vertex_offset=len(combined_points),expected_native_sha256=base.BONE_NATIVE,expected_manifest_sha256=base.BONE_MANIFEST)
  if len(bones)!=938272:raise ValueError('all185 F32/F64 barrier record census mismatch')
  exact_index=pred._prepare_surface_aabb(surf+bones);check_budget()
  allowed={facebase[r]+i for r in ROLE_ORDER for i in range(len(faces[r]))};filtered=filter_mod.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=())
  # Build face-key and vertex-to-face indices once per role, then use only local-star lookups.
  candidate_preprocess_start=time.monotonic()
  candidates={r:[] for r in ROLE_ORDER};topo_data={}
  for r in ROLE_ORDER:
   F=faces[r];P32=points[r];edge_rows={};neighbors=[set() for _ in range(len(P32))]
   face_key_rows={}
   for fi,f in enumerate(F):
    a,b,c=map(int,f)
    face_key_rows.setdefault(tuple(sorted((a,b,c))),[]).append(fi)
    for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
     key=(u,v) if u<v else (v,u);edge_rows.setdefault(key,[]).append((fi,w));neighbors[u].add(v);neighbors[v].add(u)
   flat_vertices=F.reshape(-1)
   flat_face_rows=np.repeat(np.arange(len(F),dtype=np.int64),3)
   order=np.argsort(flat_vertices,kind='stable')
   counts=np.bincount(flat_vertices,minlength=len(P32))
   vertex_face_offsets=np.empty(len(P32)+1,np.int64);vertex_face_offsets[0]=0;np.cumsum(counts,out=vertex_face_offsets[1:])
   vertex_face_rows=flat_face_rows[order]
   def incident(v,offsets=vertex_face_offsets,rows=vertex_face_rows):
    return rows[int(offsets[v]):int(offsets[v+1])]
   seen=set()
   if authorized_edges_by_role.get(r):
    seed_iter=authorized_edges_by_role[r]
   else:
    seed_iter=[]
    for fi in bad[r]:
     tri=F[int(fi)]
     for u,v in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):
      e=(u,v) if u<v else (v,u)
      seed_iter.append((e,int(fi)))
   for e,fi in seed_iter:
    if e in seen:continue
    seen.add(e);length=float(np.linalg.norm(P32[e[0]].astype(np.float64)-P32[e[1]].astype(np.float64)))
    if length>MAX_EDGE_M:continue
    ef=edge_rows.get(e,[])
    if len(ef)!=2 or int(fi) not in {int(x[0]) for x in ef}:continue
    rows=sorted(set(map(int,incident(e[0])))|set(map(int,incident(e[1]))));verts=set(map(int,F[np.asarray(rows,np.int64)].reshape(-1)))
    if verts & protected[r]:continue
    opposites={int(w) for _,w in ef};common=(neighbors[e[0]]&neighbors[e[1]])-{e[0],e[1]}
    if common!=opposites:continue
    changed_u=sorted(int(x) for x in incident(e[0]) if e[1] not in F[int(x)])
    changed_v=sorted(int(x) for x in incident(e[1]) if e[0] not in F[int(x)])
    if not changed_u and not changed_v:continue
    candidates[r].append({'edge':e,'length_m':length,'seed_face':int(fi),'bound_ratio':float(base.qbound(P32,F[[int(fi)]])[0]),'edge_rows':[int(x[0]) for x in ef],'opposite_vertices':sorted(opposites),'affected_rows':rows,'changed_rows':changed_u,'reverse_changed_rows':changed_v,'removed_rows':sorted(int(x[0]) for x in ef),'star_vertices':verts})
   candidates[r].sort(key=lambda c:(c['length_m'],c['bound_ratio'],c['seed_face'],c['edge']))
   topo_data[r]={'edge_rows':edge_rows,'vertex_face_offsets':vertex_face_offsets,'vertex_face_rows':vertex_face_rows,'neighbors':neighbors,'face_key_rows':face_key_rows}
  phase_times['candidate_discovery_s']=time.monotonic()-candidate_preprocess_start
  status['candidate_edges']=sum(len(candidates[r]) for r in ROLE_ORDER)
  progress('CANDIDATE_EDGE_CENSUS',bad_witnesses={r:len(bad[r]) for r in ROLE_ORDER},short_interior_link_edges={r:len(candidates[r]) for r in ROLE_ORDER},max_edge_m=MAX_EDGE_M)
  phase_times['pre_trial_s']=time.monotonic()-t0
  progress('READY_FOR_CANDIDATE_TRIALS',phase_times=dict(phase_times))
  trial_start=time.monotonic()
  accepted=[];accepted_records=[];retired={r:set() for r in ROLE_ORDER};used_vertices={r:set() for r in ROLE_ORDER};collapsed={r:[] for r in ROLE_ORDER};next_id=5000000
  # Keep role coverage in the bounded batch; process shortest admissible witness edges first.
  for r in ROLE_ORDER:
   for c in candidates[r]:
    if time.monotonic()>=deadline or len(accepted)>=MAX_ACCEPTED or status['trials']>=MAX_TRIALS:break
    if c['star_vertices']&used_vertices[r]:reject('overlaps_accepted_star');continue
    status['trials']+=1;F=faces[r];P32=points[r];u,v=c['edge'];ok=False;reason=''
    # Preserve deterministic lower-row preference in strict mode. In the
    # explicitly relaxed construction mode, try orientations that avoid lowering
    # the local bad-patch minimum first, then minimize the number of new bad faces.
    orientation_specs=[]
    for rem,keep in ((u,v),(v,u)):
     changed=c['changed_rows'] if rem==u else c['reverse_changed_rows']
     removed=c['removed_rows'];retired_rows=sorted(set(changed)|set(removed))
     oriented=F[np.asarray(changed,np.int64)].copy();oriented[oriented==rem]=keep
     pref=(0,0,0.0,rem,keep)
     if QBOUND_POLICY=='strict_local_net_reduction' and len(oriented) and not np.any(oriented[:,0]==oriented[:,1]) and not np.any(oriented[:,1]==oriented[:,2]) and not np.any(oriented[:,2]==oriented[:,0]):
      oldq0=base.qbound(P32,F[np.asarray(retired_rows,np.int64)]);newq0=base.qbound(P32,oriented)
      oldmask=oldq0<MIN_QBOUND;newmask=newq0<MIN_QBOUND
      oldmin=float(oldq0[oldmask].min()) if np.any(oldmask) else float('inf')
      newmin=float(newq0[newmask].min()) if np.any(newmask) else float(newq0.min())
      nondecreasing=(not np.any(newmask)) or (np.any(oldmask) and newmin>=oldmin)
      pref=(0 if nondecreasing else 1,int(np.count_nonzero(newmask)),-newmin,rem,keep)
     orientation_specs.append((pref,rem,keep,changed,retired_rows,oriented))
    if authorized_orientations_by_role.get(r):
     orientation_specs=[x for x in orientation_specs if (int(x[1]),int(x[2])) in authorized_orientations_by_role[r]]
    if QBOUND_POLICY=='strict_local_net_reduction':orientation_specs.sort(key=lambda x:x[0])
    for _,rem,keep,changed,retired_rows,newf in orientation_specs:
     removed=c['removed_rows'];star_rows=c['affected_rows']
     if len(newf)==0:reason='no_replacement_faces';continue
     if np.any(newf[:,0]==newf[:,1]) or np.any(newf[:,1]==newf[:,2]) or np.any(newf[:,0]==newf[:,2]):reason='degenerate_replacement';continue
     sorted_new=np.sort(newf,axis=1);newkeys=[tuple(map(int,x)) for x in sorted_new]
     retired_set=set(changed);retired_set.update(removed)
     if len(set(newkeys))!=len(newf) or any(any(int(fr) not in retired_set for fr in topo_data[r]['face_key_rows'].get(key,())) for key in newkeys):reason='duplicate_face';continue
     normals0=np.cross(P32[F[np.asarray(changed),1]].astype(np.float64)-P32[F[np.asarray(changed),0]].astype(np.float64),P32[F[np.asarray(changed),2]].astype(np.float64)-P32[F[np.asarray(changed),0]].astype(np.float64))
     normals1=np.cross(P32[newf[:,1]].astype(np.float64)-P32[newf[:,0]].astype(np.float64),P32[newf[:,2]].astype(np.float64)-P32[newf[:,0]].astype(np.float64))
     if not np.all(np.isfinite(normals0)) or not np.all(np.isfinite(normals1)) or np.any(np.linalg.norm(normals1,axis=1)<=0):reason='nonfinite_or_zero_normal';continue
     if np.any(np.einsum('ij,ij->i',normals0,normals1)<=0):reason='winding_or_zero_area';continue
     qa,qg=helper.stableq(P32,newf);qb=base.qbound(P32,newf)
     oldq=base.qbound(P32,F[np.asarray(retired_rows,np.int64)])
     oldbad=oldq<MIN_QBOUND;newbad=qb<MIN_QBOUND
     oldbad_count=int(np.count_nonzero(oldbad));newbad_count=int(np.count_nonzero(newbad))
     oldbad_min=float(oldq[oldbad].min()) if oldbad_count else None
     newbad_min=float(qb[newbad].min()) if newbad_count else None
     if not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qg)) or not np.all(np.isfinite(qb)):reason='nonfinite_quality_metric';continue
     if QBOUND_POLICY=='all_new_faces_pass' and newbad_count:reason='replacement_normalized_det_bound';continue
     if QBOUND_POLICY=='strict_local_net_reduction' and newbad_count>=oldbad_count:reason='replacement_qbound_not_local_net_reduction';continue
     if np.any(qa<MIN_ALTITUDE_M):reason='replacement_altitude';continue
     if np.any(qg<=0.0):reason='replacement_nonpositive_angle';continue
     # Full operation ancestry is the union of every old face incident to either endpoint.
     union=sorted(set(t for row in star_rows for t in face_union(r,int(row))))
     if not union:reason='missing_typed_union';continue
     ancestry=sorted(set(tuple(map(int,t)) for row in star_rows for t in face_ancestry(r,int(row))))
     raw,_=pred.float32_surface_records(np.asarray(P32,np.float64),newf,source_seam_groups=())
     if len(raw)!=len(newf):reason='exact_record_drop';continue
     recs=[(q[0],q[1],q[2],next_id+j,tuple(pbase[r]+int(x) for x in q[4])) for j,q in enumerate(raw)]
     ex={facebase[q]+int(x) for q in ROLE_ORDER for x in retired[q]};ex.update(facebase[r]+int(x) for x in retired_rows);filtered.set_excluded(ex)
     pair=filtered._audit_pair_prepared_first(exact_index,recs,same_surface=True)
     if pair['count']!=0:reason='retained_surface_wall_all185_intersection';continue
     internal=pred._audit_record_pairs(itertools.combinations(recs,2),same_surface=True)
     if internal['count']!=0:reason='replacement_self_intersection';continue
     if accepted_records:
      cross=pred._audit_record_pairs(((x,y) for x in recs for y in accepted_records),same_surface=True)
      if cross['count']!=0:reason='new_patch_intersection';continue
     patchid=f'{BATCH_ID}-{r}-short-edge-collapse-{len(accepted):04d}-edge{u}-{v}-keep{keep}'
     point_source=point015[r]
     new015=point_source[newf]
     if np.any(new015<0):reason='replacement_point_source_ancestry_missing';continue
     parent_source=source_face[r][np.asarray(retired_rows,np.int64)]
     patch_xyz=P32[np.asarray(sorted(c['star_vertices']),np.int64)].astype(np.float64);patch_lo=patch_xyz.min(axis=0);patch_hi=patch_xyz.max(axis=0)
     back_overlap=[sid for sid,bx in back_boxes.items() if np.all(patch_hi>=np.asarray(bx['min_world_m'])) and np.all(np.asarray(bx['max_world_m'])>=patch_lo)]
     union_array=np.asarray(union,dtype=np.int64).reshape((-1,7))
     report={'patch_id':patchid,'role':r,'operation':'fixed_coordinate_interior_surface_edge_contraction','collapsed_input_point_row':int(rem),'retained_input_point_row':int(keep),'edge_rows':list(map(int,c['edge_rows'])),'edge_length_m':c['length_m'],'seed_bad_face_row':int(c['seed_face']),'seed_bound_ratio':c['bound_ratio'],'link_condition':'link(u) intersect link(v) equals link(edge); two incident faces','link_common_vertices':c['opposite_vertices'],'patch_bbox_min_world_m':patch_lo.tolist(),'patch_bbox_max_world_m':patch_hi.tolist(),'back412_aabb_overlap_surface_ids':back_overlap,'back412_overlap_interpretation':'informational conservative AABB overlap only; not an exact muscle-surface intersection or a claim that back tissue changed','source_layer_ids':sorted(set(map(int,union_array[:,0]))),'source_body_ids':sorted(set(map(int,union_array[:,1]))),'source_bone_ids':sorted(set(map(int,union_array[:,2]))),'source_original_ids':sorted(set(map(int,union_array[:,4]))),'affected_parent_face_rows':star_rows,'retired_parent_face_rows':retired_rows,'retired_source015_face_rows':sorted(set(map(int,parent_source[parent_source>=0]))),'changed_parent_face_rows':changed,'removed_edge_face_rows':removed,'new_faces_current_point_rows':newf.astype(int).tolist(),'new_faces_source015_point_rows':new015.astype(int).tolist(),'complete_typed_parent_union':union,'typed_parent_union_sha256':base.union_digest(union),'source_face_ancestry_union':[list(x) for x in ancestry],'new_face_minimum_bound_ratio':float(qb.min()),'local_retired_bad_face_count':oldbad_count,'local_replacement_bad_face_count':newbad_count,'local_retired_bad_face_minimum_bound_ratio':oldbad_min,'local_replacement_bad_face_minimum_bound_ratio':newbad_min,'qbound_policy':QBOUND_POLICY,'replacement_minq_nondecreasing_vs_old_bad_patch':bool((newbad_count==0) or (oldbad_count>0 and newbad_min>=oldbad_min)),'new_face_minimum_altitude_m':float(qa.min()),'new_face_minimum_angle_degrees':float(qg.min()),'exact_changed_vs_complete_surface_walls_all185_pairs':int(pair['count']),'exact_aabb_candidates':int(pair.get('aabb_candidate_pairs',0)),'exact_replacement_self_pairs':int(internal['count']),'coordinates_changed':False,'native_steps':0,'physical_admission':False}
     accepted.append({'role':r,'candidate':c,'retired_rows':retired_rows,'newfaces':newf,'records':recs,'report':report,'union':union,'ancestry':ancestry,'retained_vertex':keep,'removed_vertex':rem});retired[r].update(retired_rows);used_vertices[r].update(c['star_vertices']);collapsed[r].append((rem,keep));accepted_records.extend(recs);next_id+=len(recs);status['accepted']=len(accepted);ok=True;reason='accepted'
     with (out/'accepted-operations.jsonl').open('a') as ledger:
      ledger.write(json.dumps(report,sort_keys=True)+'\n');ledger.flush()
     if len(accepted)%10==0:progress('COLLAPSE_ACCEPTED',last=report,accepted_by_role={q:sum(x['role']==q for x in accepted) for q in ROLE_ORDER})
     break
    if not ok:reject(reason or 'no_orientation_passed')
   if time.monotonic()>=deadline or len(accepted)>=MAX_ACCEPTED or status['trials']>=MAX_TRIALS:break
  if not accepted:raise RuntimeError(f'no bounded short-edge collapse passed exact gates; tried={status["trials"]}, rejects={failures}')
  ex={facebase[q]+int(x) for q in ROLE_ORDER for x in retired[q]};filtered.set_excluded(ex)
  joint=filtered._audit_pair_prepared_first(exact_index,accepted_records,same_surface=True)
  jointself=pred._audit_record_pairs(itertools.combinations(accepted_records,2),same_surface=True)
  if joint['count'] or jointself['count']:raise ValueError('joint changed-vs-complete/all185 exact gate failed')
  outroles={};quality={};
  for r in ROLE_ORDER:
   F=faces[r];parent_ret=np.asarray(sorted(retired[r]),np.int64);keepfaces=np.flatnonzero(~np.isin(np.arange(len(F),dtype=np.int64),parent_ret));patches=[x for x in accepted if x['role']==r];newf=np.concatenate([x['newfaces'] for x in patches],axis=0) if patches else np.empty((0,3),np.int64);fo=np.concatenate((F[keepfaces],newf),axis=0)
   b0=np.sort(helper.boundary(F),axis=1);b1=np.sort(helper.boundary(fo),axis=1)
   if not np.array_equal(b0,b1):raise ValueError(f'{r} boundary edge set changed')
   _,top=base.topology(fo)
   if top['nonmanifold_edges'] or top['same_direction_edges']:raise ValueError(f'{r} topology invalid {top}')
   q0=base.qbound(points[r],F);q1=base.qbound(points[r],fo)
   quality[r]={'input_faces':len(F),'output_faces':len(fo),'input_impossible_fixed_face_count':int(np.count_nonzero(q0<MIN_QBOUND)),'output_impossible_fixed_face_count':int(np.count_nonzero(q1<MIN_QBOUND)),'boundary_edges':top['boundary_edges'],'nonmanifold_edges':top['nonmanifold_edges'],'same_direction_edges':top['same_direction_edges'],'minimum_input_bound_ratio':float(q0.min()),'minimum_output_bound_ratio':float(q1.min())}
   if patches and quality[r]['output_impossible_fixed_face_count']>quality[r]['input_impossible_fixed_face_count']:raise ValueError(f'{r} accepted contractions regressed bad face count')
   if patches and QBOUND_POLICY!='all_new_faces_pass' and quality[r]['output_impossible_fixed_face_count']==quality[r]['input_impossible_fixed_face_count']:raise ValueError(f'{r} accepted contractions did not reduce bad face count under strict mode')
   typed_source_face=np.asarray(outarr[r]['face_output_to_input_rows'],np.int64);new015=point015[r][newf] if len(newf) else np.empty((0,3),np.int64)
   if len(newf) and np.any(new015<0):raise ValueError(f'{r} new face source points missing')
   unionrows=[];unionoff=[0];patchids=[];seedrows=[];patchmaps=[];cursor=len(keepfaces)
   for x in patches:
    n=len(x['newfaces']);rowsout=list(range(cursor,cursor+n));cursor+=n
    for _ in range(n):unionrows.extend(x['union']);unionoff.append(len(unionrows))
    patchids.extend([x['report']['patch_id']]*n);seedrows.extend([x['report']['seed_bad_face_row']]*n);patchmaps.append({'patch_id':x['report']['patch_id'],'output_face_rows':rowsout,'retired_parent_face_rows':x['retired_rows']});x['report']['new_output_face_rows']=rowsout
   unionarr=np.asarray(unionrows,np.int64).reshape((-1,7)) if unionrows else np.empty((0,7),np.int64)
   sourceface_keep=source_face[r][keepfaces];parent_keep=keepfaces.astype(np.int64)
   path=out/f'candidate-{r}-short-edge-collapse-raw.npz'
   np.savez_compressed(path,points_world_m=p64[r],points_world_m_f32=points[r],faces=fo,source_candidate_sha256=np.asarray([base.SOURCE[r][1]]),immediate_parent_candidate_sha256=np.asarray([INPUT[r][1]]),typed_parent_candidate_sha256=np.asarray([TYPED[r][1]]),point_output_to_parent_rows=np.arange(len(points[r]),dtype=np.int64),point_output_to_input_rows=point015[r],face_output_to_parent_rows=np.concatenate((parent_keep,np.full(len(newf),-1,np.int64))),face_output_to_input_rows=np.concatenate((sourceface_keep,np.full(len(newf),-1,np.int64))),retired_parent_face_rows=parent_ret,retired_input_face_rows=np.unique(source_face[r][parent_ret][source_face[r][parent_ret]>=0]).astype(np.int64),new_faces_parent_point_rows=newf,new_faces_input_point_rows=new015,new_face_patch_union_offsets=np.asarray(unionoff,np.int64),**{f'new_face_patch_union_{k}':unionarr[:,i] for i,k in enumerate(UNION_LANES)},new_output_face_patch_id=np.asarray(patchids,dtype='U100'),new_output_face_seed_parent_row=np.asarray(seedrows,np.int64),collapsed_vertex_input_rows=np.asarray([a for a,b in collapsed[r]],np.int64),collapsed_vertex_retained_rows=np.asarray([b for a,b in collapsed[r]],np.int64))
   outroles[r]={'path':str(path),'sha256':sha(path),'point_count':len(points[r]),'unused_collapsed_point_count':len(collapsed[r]),'input_faces':len(F),'output_faces':len(fo),'retired_faces':len(parent_ret),'new_faces':len(newf),'patches':patchmaps,'quality':quality[r],'coordinates_bitwise_unchanged':True}
  # Freeze per-operation ancestry documents only after output row IDs are known.
  for x in accepted:
   op=x['report'];op['schema']='h706.fixed-boundary-short-edge-collapse-operation.v2';op['lineage_semantics']='complete deduplicated source typed and face ancestry union over the affected edge star; not per-triangle overlap';op['input_path']=str(INPUT[x['role']][0]);op['input_sha256']=INPUT[x['role']][1];op['physical_admission']=False
   opath=out/(op['patch_id']+'.json');opath.write_text(json.dumps(op,indent=2,sort_keys=True)+'\n');op['operation_json_path']=str(opath);op['operation_json_sha256']=sha(opath)
  final={'schema':'h706.fixed-boundary-short-edge-collapse-generic.v1','batch_id':BATCH_ID,'run_config':{'path':str(CONFIG_PATH),'sha256':CONFIG_SHA},'authorized_edge_map':({'path':str(AUTHORIZED_EDGE_PATH),'sha256':AUTHORIZED_EDGE_SHA,'meaning':'only preflighted D edges and orientations are proposed'} if AUTHORIZED_EDGE_PATH is not None else None),'status':'PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL','physical_admission':False,'native_steps':0,'coordinates_changed':False,'max_edge_length_m':MAX_EDGE_M,'lineage_layers':[{'layer_id':x.get('layer_id'),'report':x['report'],'candidates':x['candidates']} for x in LINEAGE_LAYERS],'lineage_crosswalk':{'path':str(out/'lineage-crosswalk.json'),'sha256':sha(out/'lineage-crosswalk.json')},'quality_prepass_report':{'path':str(QUALITY_PREPASS),'sha256':QUALITY_PREPASS_SHA},'back_scope_report':{'path':str(BACK_SCOPE_REPORT),'sha256':BACK_SCOPE_REPORT_SHA},'input_candidates':{r:{'path':str(INPUT[r][0]),'sha256':INPUT[r][1]} for r in ROLE_ORDER},'typed_ancestry_parents':{r:{'path':str(TYPED[r][0]),'sha256':TYPED[r][1]} for r in ROLE_ORDER},'input_bad_face_count':expected,'input_all185_bone_records':len(bones),'wall_receipts':wall_receipts,'selected_candidate_edge_count':status['candidate_edges'],'trials':status['trials'],'accepted_collapse_count':len(accepted),'rejected_candidate_count':status['rejected'],'rejection_counts':failures,'accepted_by_role':{r:sum(x['role']==r for x in accepted) for r in ROLE_ORDER},'accepted_operations':[x['report'] for x in accepted],'joint_changed_vs_complete_OID_walls_all185':{'pair_count':int(joint['count']),'aabb_candidate_pairs':int(joint.get('aabb_candidate_pairs',0)),'query_records':int(joint.get('complete_query_records',0)),'new_face_self_pairs':int(jointself['count'])},'quality_after':quality,'quality_policy':{'normalized_det_minimum':MIN_QBOUND,'replacement_qbound_policy':QBOUND_POLICY,'altitude_minimum_m':MIN_ALTITUDE_M,'angle_minimum_degrees':MIN_ANGLE_DEG,'angle_floor_policy':'construction-only minimum angle floor is an explicit run parameter; strict positive angle/area, positive winding, altitude and normalized-det bound retained'},'timings':{**phase_times,'candidate_trials_s':time.monotonic()-trial_start},'back412_patch_aabb_overlaps':{'overlap_operation_count':sum(bool(x['report']['back412_aabb_overlap_surface_ids']) for x in accepted),'overlap_by_role':{q:sum(x['role']==q and bool(x['report']['back412_aabb_overlap_surface_ids']) for x in accepted) for q in ROLE_ORDER},'overlap_operation_ids':[x['report']['patch_id'] for x in accepted if x['report']['back412_aabb_overlap_surface_ids']],'surface_aabbs_world_m':back_boxes,'interpretation':'informational conservative AABB overlap only; actual back412 source assets/face rows are read-only and are not inputs to this contraction batch'},'outputs':outroles,'scope':'Local zero-coordinate-change surface edge contractions only. Vertex retirement changes topology and can change the represented surface; new matching volume meshing, full exact gates and mass/native checks are required. No physical admission claim.','elapsed_seconds':time.monotonic()-t0,'wall_budget_seconds':MAX_SECONDS}
  (out/'report.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n');progress('COMPLETE',status=final['status'],accepted=len(accepted),trials=status['trials'],quality_after=quality,joint_pairs=joint['count'])
  print(json.dumps({'status':final['status'],'out':str(out),'accepted':len(accepted),'trials':status['trials'],'quality_after':quality,'joint_pairs':joint['count'],'elapsed':final['elapsed_seconds']},indent=2),flush=True)
 except BaseException as exc:
  (out/'failure.json').write_text(json.dumps({'schema':'h706.fixed-boundary-short-edge-collapse-generic.failure.v1','batch_id':BATCH_ID,'run_config_sha256':CONFIG_SHA,'status':'BOUNDED_PROPOSAL_FAILED_OR_INCOMPLETE','error':repr(exc),'traceback':traceback.format_exc(),'counters':status,'rejection_counts':failures,'accepted_partial_count':len(locals().get('accepted',[])),'accepted_partial_operations_path':str(out/'accepted-operations.jsonl'),'elapsed_s':time.monotonic()-t0,'physical_admission':False,'native_steps':0},indent=2,sort_keys=True)+'\n');raise
if __name__=='__main__':main()
