from __future__ import annotations
import collections,csv,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np

BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207')
PT=BASE/'physical-transfer-209'
WORK=PT/'bank32-f32-conditioning-001'
INPUT=WORK/'input/D-conditioner-resume-f64.npz'
F32INPUT=WORK/'input/D-conditioner-resume-f32-cast-unqualified.npz'
EDGECSV=WORK/'input/collapsed-edge-map.csv'
MANIFEST=PT/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json'
D043=PT/'geometry-worker/matched-nesting-390/run-037-component21-f32-flips/candidate-D-component21-hull-f32-conditioned.npz'
OUT=WORK/'output'
OUT.mkdir(exist_ok=False)

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def face_edges(tr):return [tuple(sorted((int(tr[a]),int(tr[b])))) for a,b in ((0,1),(1,2),(2,0))]
def normals(P,F):
 T=P[F];return np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0])
def canonical_face(tr):return tuple(sorted(map(int,tr)))
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def build_inc(F,alive,nv):
 inc=[set() for _ in range(nv)];edge=collections.defaultdict(set);facekeys=collections.defaultdict(set)
 for fi in np.flatnonzero(alive):
  tri=F[fi];facekeys[canonical_face(tri)].add(int(fi))
  for v in tri:inc[int(v)].add(int(fi))
  for e in face_edges(tri):edge[e].add(int(fi))
 return inc,edge,facekeys

def edge_summary(F,alive,nv):
 tri=F[alive];E=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
 ue,cnt=np.unique(E,axis=0,return_counts=True)
 return ue,cnt

def boundary_loop_lengths(ue,cnt):
 bedges=ue[cnt==1];adj=collections.defaultdict(set)
 for a,b in bedges:adj[int(a)].add(int(b));adj[int(b)].add(int(a))
 if any(len(v)!=2 for v in adj.values()):return {'edge_count':int(len(bedges)),'degrees':collections.Counter(map(len,adj.values())), 'loops':None}
 unseen=set(adj);lengths=[]
 while unseen:
  root=min(unseen);cur=root;prev=-1;n=0
  while True:
   unseen.discard(cur);n+=1
   opts=adj[cur]-({prev} if prev>=0 else set())
   nxt=min(opts)
   prev,cur=cur,nxt
   if cur==root:break
   if n>len(adj)+1:return {'edge_count':int(len(bedges)),'loops':'cycle_error'}
  lengths.append(n)
 return {'edge_count':int(len(bedges)),'loops':sorted(lengths)}

def topology(P,F,alive):
 tri=F[alive];used=np.unique(tri);ue,cnt=edge_summary(F,alive,len(P));
 # Face-connected components by two-face edge adjacency.
 rows_by_edge=collections.defaultdict(list)
 for fi in np.flatnonzero(alive):
  for e in face_edges(F[fi]):rows_by_edge[e].append(int(fi))
 parent=np.arange(len(F),dtype=np.int64);rank=np.zeros(len(F),np.uint8)
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=int(parent[x])
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if rank[a]<rank[b]:a,b=b,a
  parent[b]=a
  if rank[a]==rank[b]:rank[a]+=1
 for rr in rows_by_edge.values():
  if len(rr)==2:union(rr[0],rr[1])
 comps=len({find(int(i)) for i in np.flatnonzero(alive)})
 return {'active_vertices':int(len(used)),'faces':int(len(tri)),'unique_edges':int(len(ue)),'euler_characteristic':int(len(used)-len(ue)+len(tri)),'face_components':int(comps),'edge_incidence':{str(int(k)):int(np.sum(cnt==k)) for k in np.unique(cnt)},'boundary':boundary_loop_lengths(ue,cnt)}

def signed_surface_volume(P,F,alive):
 tri=P[F[alive]]
 return float(np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])))/6.0)

def make_source_points():
 arrays=[]
 # Pins are exact source-order Float32 coordinates from the D037 base and the
 # verified 185-bank native-order replay; replay is emulation, not a fresh GPU capture.
 arrays.append(np.asarray(np.load(D043,allow_pickle=False)["points_world_m"],dtype=np.float64))
 replay_path=PT/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
 with np.load(replay_path,allow_pickle=False) as rz:
  arrays.append(np.asarray(rz["points_world_f32"],dtype=np.float64))
 manifest=json.loads(MANIFEST.read_text())
 if manifest.get("errors") or not manifest.get("all_operands_sha_checked"):
  raise RuntimeError("all-185 operand manifest is not verified")
 if len(manifest.get("operands",[]))!=185:raise RuntimeError("source manifest must contain 185 banks")
 return np.concatenate(arrays,axis=0),manifest

def main():
 t0=time.monotonic();input_sha=sha(INPUT);f32_sha=sha(F32INPUT)
 z=np.load(INPUT,allow_pickle=False);d={k:z[k] for k in z.files}
 P=np.asarray(d['points_world_m'],dtype=np.float64);F=np.asarray(d['faces'],dtype=np.int64).copy()
 if P.ndim!=2 or P.shape[1]!=3 or F.shape!=(int(d['expected_face_count'][0]),3):raise ValueError(f'unexpected replay-envelope candidate shape {P.shape}/{F.shape}')
 if not np.isfinite(P).all() or np.any(F<0) or np.any(F>=len(P)):raise ValueError('nonfinite or invalid candidate')
 Q=P.astype(np.float32).astype(np.float64);N64=normals(P,F);N32=normals(Q,F)
 initial_zero=np.flatnonzero(np.all(N32==0,axis=1));initial_bad=np.flatnonzero(np.einsum('ij,ij->i',N64,N32)<=0)
 topo0=topology(P,F,np.ones(len(F),bool));vol0=signed_surface_volume(P,F,np.ones(len(F),bool))
 source_points,manifest=make_source_points();source_keys={tuple(x) for x in source_points}
 pin=np.array([tuple(x) in source_keys for x in P],dtype=bool)
 inc,edge_faces,face_keys=build_inc(F,np.ones(len(F),bool),len(P));alive=np.ones(len(F),bool)
 # Preserve all actual open head-ring vertices and every original D/bank source point.
 edge0,cnt0=edge_summary(F,alive,len(P));boundary_vertices=set(map(int,np.unique(edge0[cnt0==1])))
 pin[list(boundary_vertices)]=True
 pin_count=int(pin.sum())
 face_orig=np.asarray(d.get('parent_face_source_original_id',d['face_source_original_id']),dtype=np.int64).copy()
 face_id=np.asarray(d.get('parent_face_source_face_id',d['face_source_face_id']),dtype=np.int64).copy()
 face_layer=np.asarray(d['face_source_layer'],dtype=np.int64).copy()
 face_body=np.asarray(d['face_source_body'],dtype=np.int64).copy()
 face_bone=np.asarray(d['face_source_bone'],dtype=np.int64).copy()
 face_i0=np.asarray(d['face_source_identity0'],dtype=np.int64).copy()
 face_i1=np.asarray(d['face_source_identity1'],dtype=np.int64).copy()
 union_run=np.asarray(d['union_input_run_original_id'],dtype=np.int64).copy()
 union_id=np.asarray(d['union_input_face_id'],dtype=np.int64).copy()
 union_row=np.asarray(d['union_input_face_rows'],dtype=np.int64).copy()
 d289_off=np.asarray(d['d289_parent_row_indptr'],dtype=np.int64);d289_rows=np.asarray(d['d289_parent_rows'],dtype=np.int64)
 d289_anc={};face_anc={};union_anc={};
 if 'resume_face_ancestry_offsets' in d:
  oa=np.asarray(d['resume_face_ancestry_offsets'],dtype=np.int64); ar=np.asarray(d['resume_face_ancestry_input_rows'],dtype=np.int64)
  face_anc={i:set(map(int,ar[oa[i]:oa[i+1]])) for i in range(len(F))}
  uo=np.asarray(d['resume_union_ancestry_offsets'],dtype=np.int64); ur=np.asarray(d['resume_union_ancestry_run_ids'],dtype=np.int64); uf=np.asarray(d['resume_union_ancestry_face_ids'],dtype=np.int64); uw=np.asarray(d['resume_union_ancestry_input_rows'],dtype=np.int64)
  union_anc={i:set((int(ur[j]),int(uf[j]),int(uw[j])) for j in range(uo[i],uo[i+1])) for i in range(len(F))}
  changed=np.asarray(d['resume_changed_face_mask'],dtype=bool).copy()
 else:
  changed=np.zeros(len(F),dtype=bool)
 vertex_rep=np.arange(len(P),dtype=np.int64)
 def ancestry(fi):return face_anc.get(int(fi),{int(fi)})
 def dparents(fi):
  if int(fi) in d289_anc:return d289_anc[int(fi)]
  return set(map(int,d289_rows[d289_off[fi]:d289_off[fi+1]]))
 def unionparents(fi):
  if int(fi) in union_anc:return union_anc[int(fi)]
  if union_row[fi]<0:return set()
  return {(int(union_run[fi]),int(union_id[fi]),int(union_row[fi]))}
 reject=collections.Counter();events=[];max_gap=0.0;max_surface_delta=0.0
 def contract(a,b,phase):
  nonlocal max_gap
  if not alive[a] or not alive[b] or not np.array_equal(Q[a],Q[b]):return False,'not_current_F32_equal_edge'
  if pin[a] and pin[b]:return False,'both_source_or_head_pinned'
  e=tuple(sorted((a,b)));ef=set(edge_faces.get(e,set()))
  if len(ef)!=2:return False,'not_interior_edge_incidence_two'
  region=set(inc[a])|set(inc[b])
  if any(boundary_vertices.intersection(F[fi]) for fi in region):return False,'boundary_one_ring_pinned'
  na=set(int(x) for fi in inc[a] for x in F[fi] if x!=a)
  nb=set(int(x) for fi in inc[b] for x in F[fi] if x!=b)
  opposite=set(int(x) for fi in ef for x in F[fi] if x not in (a,b))
  if na&nb!=opposite:return False,'link_condition'
  # Keep source-pinned rows; otherwise retain the endpoint incident to more D-layer faces.
  if pin[a]:keep,drop=a,b
  elif pin[b]:keep,drop=b,a
  else:
   score_a=sum(face_layer[fi]==2 for fi in inc[a]);score_b=sum(face_layer[fi]==2 for fi in inc[b])
   keep,drop=(a,b) if score_a>score_b or (score_a==score_b and a<b) else (b,a)
  gone=sorted(ef);rem=sorted(set(inc[drop])-ef)
  if not rem:return False,'no_surviving_drop_star'
  old=F[rem].copy();new=old.copy();new[new==drop]=keep
  if np.any((new[:,0]==new[:,1])|(new[:,1]==new[:,2])|(new[:,0]==new[:,2])):return False,'index_degenerate_survivor'
  nold64=normals(P,old);nnew64=normals(P,new)
  if np.any(np.all(nnew64==0,axis=1)):return False,'F64_degenerate_survivor'
  dot64=np.einsum('ij,ij->i',nold64,nnew64)
  f64_orientation_reversed=np.flatnonzero((np.linalg.norm(nold64,axis=1)>0)&(dot64<=0))
  if len(f64_orientation_reversed):
   old32_check=normals(Q,old);new32_check=normals(Q,new)
   if not np.array_equal(old32_check,new32_check):return False,'F32_orientation_reversal'
  nold32=normals(Q,old);nnew32=normals(Q,new)
  oldgood=np.any(nold32!=0,axis=1);newzero=np.all(nnew32==0,axis=1)
  if np.any(oldgood&newzero):return False,'new_F32_zero_face'
  dot32=np.einsum('ij,ij->i',nold32,nnew32)
  if np.any(oldgood&(dot32<=0)):return False,'F32_orientation_reversal'
  old_keys=[canonical_face(F[fi]) for fi in gone+rem]
  new_keys=[canonical_face(tr) for tr in new]
  if len(set(new_keys))!=len(new_keys):return False,'duplicate_new_faces'
  participating=set(gone+rem)
  for fi,key in zip(rem,new_keys):
   if key in face_keys and (face_keys[key]-participating):return False,'duplicate_output_face'
  for fi,key in zip(gone+rem,old_keys):
   face_keys[key].discard(fi)
   if not face_keys[key]:del face_keys[key]
  removed_ancestry=set().union(*(ancestry(fi) for fi in gone))
  removed_d289=set().union(*(dparents(fi) for fi in gone))
  removed_union=set().union(*(unionparents(fi) for fi in gone))
  source_union=sorted(set(int(face_orig[r]) for r in removed_ancestry))
  for fi in gone:
   for v in F[fi]:inc[int(v)].discard(fi)
   for ee in face_edges(F[fi]):
    edge_faces[ee].discard(fi)
    if not edge_faces[ee]:del edge_faces[ee]
   alive[fi]=False
  for fi,tr in zip(rem,new):
   for v in F[fi]:inc[int(v)].discard(fi)
   for ee in face_edges(F[fi]):
    edge_faces[ee].discard(fi)
    if not edge_faces[ee]:del edge_faces[ee]
   F[fi]=tr
   for v in tr:inc[int(v)].add(fi)
   for ee in face_edges(tr):edge_faces[ee].add(fi)
   face_keys[canonical_face(tr)].add(fi)
   face_anc[fi]=ancestry(fi)|removed_ancestry
   d289_anc[fi]=dparents(fi)|removed_d289
   union_anc[fi]=unionparents(fi)|removed_union
   changed[fi]=True
  vertex_rep[vertex_rep==drop]=keep
  pin[keep]=pin[keep] or pin[drop]
  gap=float(np.linalg.norm(P[keep]-P[drop]));max_gap=max(max_gap,gap)
  events.append({'phase':phase,'kept_input_vertex':int(keep),'removed_input_vertex':int(drop),'f32_equal_bits':Q[keep].astype(np.float32).view(np.uint32).tolist(),'f64_gap_m':gap,'removed_input_face_rows':gone,'changed_input_face_rows':rem,'removed_face_origins':source_union,'f64_orientation_sign_change_surviving_face_rows':[int(rem[j]) for j in f64_orientation_reversed]})
  return True,None

 # Recompute actual coincident edges after each full pass; 54 initial edges are a bounded witness set.
 initial_edges=[]
 for ee,rr in edge_faces.items():
  a,b=ee
  if np.array_equal(Q[a],Q[b]):initial_edges.append((float(np.linalg.norm(P[a]-P[b])),a,b))
 initial_edges.sort()
 max_contracts=2048
 for sweep in range(4):
  accepted_sweep=0
  current=[]
  for ee,rr in edge_faces.items():
   a,b=ee
   if np.array_equal(Q[a],Q[b]):current.append((float(np.linalg.norm(P[a]-P[b])),a,b))
  current.sort()
  for gap,a,b in current:
   if len(events)>=max_contracts:break
   if not alive[a] or not alive[b] or tuple(sorted((a,b))) not in edge_faces:continue
   success,why=contract(a,b,f'coincident_edge_sweep_{sweep}_forward')
   if not success:success,why=contract(b,a,f'coincident_edge_sweep_{sweep}_reverse')
   if success:accepted_sweep+=1
   else:reject[why]+=1
  if accepted_sweep==0:break

 # Bounded alternate-diagonal repair for nonzero F32 orientation failures.
 max_flips=512;flips=[];flip_reject=collections.Counter()
 def_flip_start=time.monotonic()
 # The helper is intentionally performed after all collapse sweeps so its edge keys reflect the final local topology.
 Q=P.astype(np.float32).astype(np.float64)
 for pass_id in range(2):
  badrows=[]
  for fi in np.flatnonzero(alive):
   n64=normals(P,F[fi:fi+1])[0];n32=normals(Q,F[fi:fi+1])[0]
   if float(n64@n32)<=0:badrows.append(int(fi))
  if not badrows:break
  progress=0
  for fi in badrows:
   if len(flips)>=max_flips:break
   if not alive[fi]:continue
   tr=F[fi].copy();done=False
   for side in range(3):
    a,b,c=map(int,(tr[side],tr[(side+1)%3],tr[(side+2)%3]));key=tuple(sorted((a,b)));adj=set(edge_faces.get(key,set()))-{fi}
    if len(adj)!=1:flip_reject['not_two_face_edge']+=1;continue
    fj=next(iter(adj));other=F[fj].copy();rr=[tuple(map(int,np.roll(other,-k))) for k in range(3)]
    ds=[v for v in other if v not in (a,b)]
    if len(ds)!=1:flip_reject['bad_neighbor']+=1;continue
    d0=int(ds[0])
    if (b,a,d0) not in rr:flip_reject['same_or_bad_winding']+=1;continue
    if any(int(v) in boundary_vertices for v in (a,b,c,d0)):flip_reject['head_boundary_one_ring']+=1;continue
    diag=tuple(sorted((c,d0)))
    if diag in edge_faces:flip_reject['alternate_diagonal_exists']+=1;continue
    e=P[b]-P[a];den=float(e@e)
    if den<=0:continue
    def altitude(x):
     beta=float((P[x]-P[a])@e/den)
     return float(np.linalg.norm(P[x]-(P[a]+np.clip(beta,0,1)*e))),beta
    hc,bc=altitude(c);hd,bd=altitude(d0)
    if bc<0 or bc>1 or hc>1e-6:flip_reject['bad_triangle_not_submicron_flat']+=1;continue
    nfa=np.asarray([[c,d0,b],[d0,c,a]],dtype=np.int64)
    old=F[[fi,fj]].copy();nold64=normals(P,old);guide=nold64.sum(axis=0)
    if np.linalg.norm(guide)==0:flip_reject['canceling_old_normals']+=1;continue
    nnew64=normals(P,nfa);nnew32=normals(Q,nfa)
    if np.any(np.all(nnew64==0,axis=1)) or np.any(np.all(nnew32==0,axis=1)):flip_reject['new_degenerate']+=1;continue
    if np.any(nnew64@guide<=0) or np.any(nnew32@guide<=0) or np.any(np.einsum('ij,ij->i',nnew64,nnew32)<=0):flip_reject['new_orientation']+=1;continue
    k0,k1=canonical_face(nfa[0]),canonical_face(nfa[1])
    # Exclude the two faces being replaced from the occupancy test.
    if (k0 in face_keys and face_keys[k0]-{fi,fj}) or (k1 in face_keys and face_keys[k1]-{fi,fj}):flip_reject['duplicate_face']+=1;continue
    old_anc=ancestry(fi)|ancestry(fj);old_d=set(dparents(fi))|set(dparents(fj));old_u=unionparents(fi)|unionparents(fj)
    for row,oldtr in zip((fi,fj),old):
     face_keys[canonical_face(oldtr)].discard(row)
     if not face_keys[canonical_face(oldtr)]:del face_keys[canonical_face(oldtr)]
     for ee in face_edges(oldtr):
      edge_faces[ee].discard(row)
      if not edge_faces[ee]:del edge_faces[ee]
    F[fi]=nfa[0];F[fj]=nfa[1]
    for row,ntr in zip((fi,fj),nfa):
     face_keys[canonical_face(ntr)].add(row)
     for ee in face_edges(ntr):edge_faces[ee].add(row)
     face_anc[row]=set(old_anc);d289_anc[row]=set(old_d);union_anc[row]=set(old_u);changed[row]=True
    flips.append({'face_rows':[int(fi),int(fj)],'old_faces':old.tolist(),'new_faces':nfa.tolist(),'quad_altitudes_m':[hc,hd],'diagonal_flip_error_bound_m':2*hc})
    done=True;progress+=1;break
   if not done:flip_reject['no_safe_alternate']+=1
  if progress==0:break

 # Preserve any remaining F32-coincident edge whose post-flip contraction would alter global topology.
 topo_after_flips=topology(P,F,alive)
 # Emit a compact F64 cache and one exact F32 cast through the existing deterministic emitter.
 rows=np.flatnonzero(alive);used,inv=np.unique(F[rows].reshape(-1),return_inverse=True);Pcompact=P[used];Fcompact=inv.reshape(-1,3);changed_rows=changed[rows]
 # Save the complete surgical F64 geometry before any F32 emission or nudge.
 np.savez_compressed(OUT/'pre-emission-f64-geometry.npz',points_world_m_f64=Pcompact,faces_f64=Fcompact,face_input_rows=rows,changed_face_mask=changed_rows,vertex_input_rows=used)
 # Keep operation and source ancestry before trying the F32 emitter.
 ck_anc=[sorted(ancestry(int(fi))) for fi in rows]; ck_aoff=[0]; ck_arows=[]
 ck_uoff=[0]; ck_ur=[]; ck_uf=[]; ck_urow=[]
 for fi,aa in zip(rows,ck_anc):
  ck_arows.extend(aa); ck_aoff.append(len(ck_arows))
  for rr,ff,fr in sorted(unionparents(int(fi))): ck_ur.append(rr); ck_uf.append(ff); ck_urow.append(fr)
  ck_uoff.append(len(ck_urow))
 np.savez_compressed(OUT/'pre-emission-lineage-checkpoint.npz',face_input_rows=rows,changed_face_mask=changed_rows,vertex_input_rows=used,face_ancestry_offsets=np.asarray(ck_aoff,np.int64),face_ancestry_input_rows=np.asarray(ck_arows,np.int64),union_ancestry_offsets=np.asarray(ck_uoff,np.int64),union_ancestry_run_ids=np.asarray(ck_ur,np.int64),union_ancestry_face_ids=np.asarray(ck_uf,np.int64),union_ancestry_input_rows=np.asarray(ck_urow,np.int64))
 (OUT/'conditioner-events-before-emission.json').write_text(json.dumps({'input_sha256':input_sha,'accepted_contractions':events,'contraction_rejections':dict(reject),'accepted_flips':flips,'flip_rejections':dict(flip_reject),'max_f64_endpoint_contraction_m':max_gap,'pre_emission_face_count':len(Fcompact),'pre_emission_vertex_count':len(Pcompact)},indent=2)+'\n')
 Qcompact=Pcompact.astype(np.float32).astype(np.float64);preN64=normals(Pcompact,Fcompact);preN32=normals(Qcompact,Fcompact)
 pre_zero=np.flatnonzero(np.all(preN32==0,axis=1));pre_orient=np.flatnonzero(np.einsum('ij,ij->i',preN64,preN32)<=0);pre_duplicate_extra=int(len(Qcompact)-len(np.unique(Qcompact,axis=0)))
 source32=source_points.astype(np.float32)
 sp=importlib.util.spec_from_file_location('emitter',BASE/'emit_f32.py');em=importlib.util.module_from_spec(sp);sp.loader.exec_module(em)
 Y,Femit,emit_report=em.emit(Pcompact,Fcompact,source32)
 nudge_distances=[float(x['distance_m']) for x in emit_report['nudged_vertices']]
 max_emitter_nudge=max(nudge_distances,default=0.0)
 if max_emitter_nudge>1e-7:
  raise ValueError(f'emitter nudge exceeded the bounded 0.1um limit: {max_emitter_nudge}')
 if not np.array_equal(Y.astype(np.float64),Y.astype(np.float32).astype(np.float64)):
  raise ValueError('emitted point table is not exactly Float32')
 # Map input candidate point ancestry through contraction representatives and emitter compaction.
 compact_input_rows=used
 input_to_compact=np.full(len(P),-1,np.int64);input_to_compact[used]=np.arange(len(used))
 compact_to_emit=np.asarray(emit_report['input_vertex_old_to_emitted'],np.int64)
 emit_to_compact=np.full(len(Y),-1,np.int64);emit_to_compact[compact_to_emit[compact_to_emit>=0]]=np.flatnonzero(compact_to_emit>=0)
 emit_input_rows=used[emit_to_compact]
 # Map every original input point to its surviving emitted representative.
 root_of=vertex_rep.copy()
 for i in range(len(root_of)):
  r=int(root_of[i])
  while int(root_of[r])!=r:r=int(root_of[r])
  root_of[i]=r
 root_emit={int(used[j]):int(compact_to_emit[j]) for j in range(len(used)) if compact_to_emit[j]>=0}
 vertex_groups=collections.defaultdict(list)
 for i,r in enumerate(root_of):
  eidx=root_emit.get(int(r))
  if eidx is not None:vertex_groups[eidx].append(i)
 voff=[0];vrows=[]
 for i in range(len(Y)):
  vrows.extend(vertex_groups.get(i,[]));voff.append(len(vrows))

 # Construct per-face ancestry CSR and preserve union and parent-tet references through surgery.
 input_rows=rows.copy();out_face_anc=[sorted(ancestry(int(fi))) for fi in rows]
 face_anc_off=[0];face_anc_rows=[];d289_off_out=[0];d289_rows_out=[];union_off=[0];union_r=[];union_f=[];union_row_out=[]
 for fi,aa in zip(rows,out_face_anc):
  face_anc_rows.extend(aa);face_anc_off.append(len(face_anc_rows))
  dp=dparents(int(fi));d289_rows_out.extend(sorted(dp));d289_off_out.append(len(d289_rows_out))
  uu=unionparents(int(fi));
  for rid,fid,rr in sorted(uu):union_r.append(rid);union_f.append(fid);union_row_out.append(rr)
  union_off.append(len(union_row_out))
 attrs={k:np.asarray(d[k])[rows].copy() for k in ('face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','face_source_identity0','face_source_identity1')}
 union_run_out=np.asarray(d['union_input_run_original_id'])[rows].copy();union_id_out=np.asarray(d['union_input_face_id'])[rows].copy();union_rows_out=np.asarray(d['union_input_face_rows'])[rows].copy()
 existing_ids=[int(k) for k in json.loads(str(d['face_source_identity_schema_by_original_id_json'][0])).keys() if k.isdigit()]
 newid=126000032
 changed_indices=np.flatnonzero(changed_rows);newfids=np.arange(len(changed_indices),dtype=np.int64)
 schema=json.loads(str(d['face_source_identity_schema_by_original_id_json'][0]))
 parents=sorted(set(int(face_orig[r]) for outrow in changed_indices for r in out_face_anc[outrow]))
 attrs['face_source_layer'][changed_rows]=2;attrs['face_source_body'][changed_rows]=23;attrs['face_source_bone'][changed_rows]=32
 attrs['face_source_original_id'][changed_rows]=newid;attrs['face_source_face_id'][changed_rows]=newfids;attrs['face_source_identity0'][changed_rows]=newfids;attrs['face_source_identity1'][changed_rows]=-1
 union_run_out[changed_rows]=-1;union_id_out[changed_rows]=-1;union_rows_out[changed_rows]=-1
 # Derive new source-view faces after Float32 emission; the view contains only referenced patch points.
 patch_emit_rows=changed_indices
 patch_faces=Femit[patch_emit_rows]
 patch_used,patch_inv=np.unique(patch_faces.reshape(-1),return_inverse=True);patch_faces_local=patch_inv.reshape(-1,3)
 patch_points=Y[patch_used]
 sourceview=OUT/f'BONE-derived-{newid}-source.npz'
 source_parent_rows=np.asarray([r for j in patch_emit_rows for r in out_face_anc[j]],dtype=np.int64)
 source_parent_off=[0]
 for j in patch_emit_rows:source_parent_off.append(source_parent_off[-1]+len(out_face_anc[j]))
 view_parent_orig=np.asarray(face_orig[source_parent_rows],dtype=np.int64);view_parent_fid=np.asarray(face_id[source_parent_rows],dtype=np.int64)
 view_parent_i0=np.asarray(d.get('parent_face_source_identity0',d['face_source_identity0'])[source_parent_rows],dtype=np.int64);view_parent_i1=np.asarray(d.get('parent_face_source_identity1',d['face_source_identity1'])[source_parent_rows],dtype=np.int64)
 view_attrs={k:attrs[k][patch_emit_rows].copy() for k in attrs}
 view_attrs['face_source_original_id'][:]=newid
 view_attrs['face_source_face_id']=newfids.copy();view_attrs['face_source_identity0']=newfids.copy();view_attrs['face_source_identity1'][:]=-1
 parent_assets=[{'asset_role':'pre_emission_all185_union_F64_cache','path':str(INPUT),'sha256':input_sha},{'asset_role':'direct_cast_unqualified_F32_diagnostic','path':str(F32INPUT),'sha256':f32_sha},{'asset_role':'source_order_native_F32_replay_archive','path':str(PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'),'sha256':sha(PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz')},{'asset_role':'all185_operand_manifest','path':str(MANIFEST),'sha256':sha(MANIFEST)}]
 np.savez_compressed(sourceview,points_world_m=patch_points,faces=patch_faces_local,source_candidate_face_rows=np.full(len(patch_emit_rows),-1,np.int64),source_candidate_vertex_rows=used[emit_to_compact[patch_used]],**view_attrs,
  face_ancestry_offsets=np.asarray(source_parent_off,np.int64),face_ancestry_input_cache_rows=source_parent_rows,face_ancestry_original_ids=view_parent_orig,face_ancestry_face_ids=view_parent_fid,face_ancestry_identity0=view_parent_i0,face_ancestry_identity1=view_parent_i1,
  parent_source_original_ids=np.asarray(parents,np.int64),parent_assets_json=np.asarray([json.dumps(parent_assets,sort_keys=True)]),source_operation=np.asarray(['bank23_bone32_cube2um_envelope_link_safe_f32_conditioning']),source_candidate_sha256=np.asarray([input_sha]),source_f64_candidate_sha256=np.asarray([input_sha]),source_cache_sha256=np.asarray([str(d['input_union_sha256'][0])]))
 schema[str(newid)]={'role':'BONE','body':23,'bone':32,'layer':2,'path':str(sourceview),'sha256':sha(sourceview),'identity_keys':['face_source_identity0','face_source_identity1'],'identity_face_id_key':'face_source_face_id','face_count':len(patch_emit_rows),'derived_operation':'float32_coincident_seam_link_contraction_and_bounded_dualprecision_diagonal_flip','derived_parent_source_ids':parents,'derived_parent_assets':parent_assets,'source_geometry_scope':'complete_reconstructed_patch_faces','maximum_vertex_displacement_m':float(max_gap),'maximum_diagonal_flip_error_bound_m':max([e['diagonal_flip_error_bound_m'] for e in flips] if flips else [0.0])}

 # Recompute global postconditions on emitted Float32 geometry.
 N64compact=normals(Pcompact,Fcompact);N32emit=normals(Y,Femit)
 zero=np.flatnonzero(np.all(N32emit==0,axis=1));orient=np.flatnonzero(np.einsum('ij,ij->i',N64compact,N32emit)<=0)
 ue,cnt=edge_summary(Femit,np.ones(len(Femit),bool),len(Y));topo1=topology(Y.astype(np.float64),Femit,np.ones(len(Femit),bool));vol1=signed_surface_volume(Y.astype(np.float64),Femit,np.ones(len(Femit),bool))
 unique_f32=len(np.unique(Y,axis=0));duplicate_classes=len(Y)-unique_f32
 if len(zero) or len(orient):
  structural='FAIL_F32_FACE_GEOMETRY'
 elif np.any(cnt>2) or topo1['boundary']!=topo0['boundary'] or topo1['euler_characteristic']!=topo0['euler_characteristic'] or topo1['face_components']!=topo0['face_components']:
  structural='FAIL_TOPOLOGY'
 else:structural='PASS_STRUCTURAL_F32'
 # Dump candidate metadata in full before source producer preflight.
 mainattrs={k:attrs[k] for k in attrs}
 # Native face row/order is unchanged except the 106 degenerate contraction rows and further locally edited rows.
 base_meta={k:np.asarray(d[k])[rows] for k in ()}
 report={'schema':'numi.bank32-bounded-f32-topology-conditioning.v1','input_f64_path':str(INPUT),'input_f64_sha256':input_sha,'input_f32_path':str(F32INPUT),'input_f32_sha256':f32_sha,'initial_vertices':len(P),'initial_faces':len(F),'initial_zero_f32_faces':len(initial_zero),'initial_f32_orientation_invalid_faces':len(initial_bad),'initial_topology':topo0,'topology_after_link_contractions_and_flips':topo_after_flips,'initial_oriented_surface_volume_sum_m3':vol0,'source_and_head_pinned_vertex_count':pin_count,'initial_duplicate_f32_coordinate_classes':int(len(P)-len(np.unique(Q,axis=0))),'collapsed_edge_candidates_initial':len(initial_edges),'accepted_contractions':events,'contraction_rejections':dict(reject),'accepted_diagonal_flips':flips,'flip_rejections':dict(flip_reject),'max_f64_endpoint_contraction_m':max_gap,'source_exact_points_loaded':int(len(source_points)),'source_manifest_sha256':sha(MANIFEST),'pre_emission_zero_faces':pre_zero.tolist(),'pre_emission_orientation_invalid_faces':pre_orient.tolist(),'pre_emission_duplicate_coordinate_extra_rows':pre_duplicate_extra,'emitter_nudges':emit_report['nudged_vertices'],'max_emitter_nudge_m':max_emitter_nudge,'emitter_report':emit_report,'post_f32_zero_faces':zero.tolist(),'post_f32_orientation_invalid_faces':orient.tolist(),'post_f32_duplicate_coordinate_classes':int(duplicate_classes),'post_topology':topo1,'post_oriented_surface_volume_sum_m3':vol1,'oriented_surface_volume_sum_delta_m3':vol1-vol0,'face_ancestry_derived_id':int(newid),'derived_face_count':int(len(patch_emit_rows)),'derived_parent_source_ids':parents,'candidate_structural_status':structural,'exact_self_audit':'NOT_RUN','original_f64_bank_containment':'NOT_RUN','native_f32_bank_containment':'NOT_RUN','admission':False,'physical_steps':0,'elapsed_s':time.monotonic()-t0}
 # Save the F64 stage before any downstream audit.
 np.savez_compressed(OUT/'candidate-f64.npz',points_world_m_f64=Pcompact,faces_f64=Fcompact,output_face_input_candidate_rows=rows,output_vertex_input_candidate_rows=emit_input_rows,
  output_vertex_input_parent_offsets=np.asarray(voff,np.int64),output_vertex_input_parent_rows=np.asarray(vrows,np.int64),
  **mainattrs,union_input_run_original_id=union_run_out,union_input_face_id=union_id_out,union_input_face_rows=union_rows_out,
  union_face_ancestry_offsets=np.asarray(union_off,np.int64),union_face_ancestry_run_original_ids=np.asarray(union_r,np.int64),union_face_ancestry_face_ids=np.asarray(union_f,np.int64),union_face_ancestry_rows=np.asarray(union_row_out,np.int64),
  d289_parent_row_indptr=np.asarray(d289_off_out,np.int64),d289_parent_rows=np.asarray(d289_rows_out,np.int64),
  candidate_face_ancestry_offsets=np.asarray(face_anc_off,np.int64),candidate_face_ancestry_rows=np.asarray(face_anc_rows,np.int64),
  source_cache_sha256=np.asarray([input_sha]),face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True)]))
 # F32 authoritative candidate retains all required typed arrays and source lineage CSRs.
 np.savez_compressed(OUT/'candidate-f32.npz',points_world_m=Y.astype(np.float32),faces=Femit.astype(np.int64),**mainattrs,
  union_input_run_original_id=union_run_out,union_input_face_id=union_id_out,union_input_face_rows=union_rows_out,
  union_face_ancestry_offsets=np.asarray(union_off,np.int64),union_face_ancestry_run_original_ids=np.asarray(union_r,np.int64),union_face_ancestry_face_ids=np.asarray(union_f,np.int64),union_face_ancestry_rows=np.asarray(union_row_out,np.int64),
  d289_parent_row_indptr=np.asarray(d289_off_out,np.int64),d289_parent_rows=np.asarray(d289_rows_out,np.int64),
  candidate_face_ancestry_offsets=np.asarray(face_anc_off,np.int64),candidate_face_ancestry_rows=np.asarray(face_anc_rows,np.int64),
  output_vertex_input_parent_offsets=np.asarray(voff,np.int64),output_vertex_input_parent_rows=np.asarray(vrows,np.int64),output_vertex_input_rows=emit_input_rows,
  face_source_identity_schema_by_original_id_json=np.asarray([json.dumps(schema,sort_keys=True)]),float32_emission_json=np.asarray([json.dumps(emit_report,sort_keys=True)]),
  input_candidate_sha256=np.asarray([input_sha]),input_candidate_f32_sha256=np.asarray([f32_sha]))
 report['candidate_f64_sha256']=sha(OUT/'candidate-f64.npz');report['candidate_f32_sha256']=sha(OUT/'candidate-f32.npz');report['sourceview_sha256']=sha(sourceview)
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 # Bank-envelope structural source-preparation result; native self/containment gates run separately.
 bank_structural={"status":"PASS" if structural=="PASS_STRUCTURAL_F32" else "FAIL","scope":"Float32 nondegeneracy, orientation, indexed closed topology; no geometric self/containment/admission claim"}
 (OUT/"bank-structural-result.json").write_text(json.dumps(bank_structural,indent=2)+"\n")
 print(json.dumps({"output":str(OUT),"status":structural,"bank_structural":bank_structural,"input_zero_faces":len(initial_zero),"input_orientation_bad":len(initial_bad),"contractions":len(events),"flips":len(flips),"remaining_zero":len(zero),"remaining_orientation_bad":len(orient),"remaining_duplicate_classes":int(duplicate_classes),"boundary":topo1["boundary"],"euler":topo1["euler_characteristic"],"face_components":topo1["face_components"],"max_gap_m":max_gap,"elapsed_s":time.monotonic()-t0},sort_keys=True),flush=True)
 if structural!="PASS_STRUCTURAL_F32":raise SystemExit(2)

if __name__ == "__main__":
 main()
