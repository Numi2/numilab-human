#!/Users/numi/numi-human-prep-venv-20261005/bin/python
import collections,hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
ROOT=P/'paired-head-reference-fit-684'
IN=ROOT/'run-006-i-sliver-local-flips/candidate.npz'
IN_REPORT=ROOT/'run-006-i-sliver-local-flips/receipt.json'
OP=P/'surface-transfer-checkpoint-392/payload/O/candidate.npz'
IP=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
FIELD=ROOT/'run-005-normalized-grid12mm/shared-world-field.npz'
SKIN=P/'paired-fine-headwall-rim-635/run-013/output/fine-skin-closed-boundary.npz'
FAT=P/'paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz'
HELPER=P/'root-edge-flips/local_float32_edge_flip.py'
OUT=ROOT/'run-010-two-link-valid-collapses'
TARGET_FAT=160e-6
SKIN_BASE_REGION_ML=244.2586928786206
SKIN_CAP_REGION_ML=264.5427051829506
FAT_BASE_REGION_ML=562.5438644482234
FAT_MIN_REGION_ML=718.311
COLLAPSES=[(52064,52065),(59209,59208)] # keep, remove

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def volume(Pts,F):
 q=np.asarray(Pts,np.float64);f=np.asarray(F,np.int64);o=q.mean(0);t=q[f]-o
 return float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2]),optimize=True).sum()/6.)
def edge_data(F):
 inc=collections.defaultdict(list);direct=collections.defaultdict(list)
 for r,t in enumerate(F):
  for k in range(3):
   a,b=map(int,(t[k],t[(k+1)%3]));inc[(min(a,b),max(a,b))].append(r);direct[(min(a,b),max(a,b))].append((a,b))
 return inc,direct
def boundary_edges(F):
 inc,_=edge_data(F);return sorted(e for e,rows in inc.items() if len(rows)==1)
def topology(Pts,F,expected_boundary=None):
 F=np.asarray(F,np.int64);Pts=np.asarray(Pts)
 inc,direct=edge_data(F);bad=[]
 for e,rows in inc.items():
  if len(rows)>2:bad.append((e,'incidence',len(rows)))
  elif len(rows)==2 and direct[e][0]==direct[e][1]:bad.append((e,'same_direction',direct[e]))
  elif len(rows)<1:bad.append((e,'empty',0))
 dups=collections.Counter(tuple(sorted(map(int,t))) for t in F)
 duplicate_faces=[k for k,v in dups.items() if v>1]
 used=np.unique(F);boundary=set(e for e,r in inc.items() if len(r)==1)
 if expected_boundary is not None and boundary!=set(expected_boundary):bad.append(('boundary_set',len(boundary),len(expected_boundary),len(boundary^set(expected_boundary))))
 links=[{} for _ in range(len(Pts))]
 for t in F:
  a,b,c=map(int,t)
  for v,x,y in ((a,b,c),(b,c,a),(c,a,b)):
   links[v].setdefault(x,set()).add(y);links[v].setdefault(y,set()).add(x)
 bvertices={v for e in boundary for v in e};badlinks=[]
 for v in used:
  adj=links[v]
  if not adj:badlinks.append((int(v),'empty'));continue
  deg=[len(q) for q in adj.values()]
  expected_end=2 if int(v) in bvertices else 0
  if (expected_end and deg.count(1)!=2) or (not expected_end and any(d!=2 for d in deg)) or any(d>2 or d==0 for d in deg):badlinks.append((int(v),'degree',deg.count(1),max(deg)))
  seen=set();stack=[next(iter(adj))]
  while stack:
   u=stack.pop()
   if u in seen:continue
   seen.add(u);stack.extend(adj[u]-seen)
  if len(seen)!=len(adj):badlinks.append((int(v),'disconnected',len(seen),len(adj)))
 # Connected components via vertex union-find.
 parent={int(v):int(v) for v in used}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for t in F:
  a=int(t[0])
  for b in map(int,t[1:]):parent[find(a)]=find(b)
 comps=len({find(v) for v in parent})
 return {'vertices_used':int(len(used)),'edges':int(len(inc)),'faces':int(len(F)),'euler':int(len(used)-len(inc)+len(F)),'components':comps,'boundary_edges':len(boundary),'nonmanifold_or_orientation_edges':bad[:20],'nonmanifold_or_orientation_count':len(bad),'duplicate_face_count':len(duplicate_faces),'bad_vertex_link_count':len(badlinks),'bad_vertex_link_examples':badlinks[:20]}
def tri_quality(points,faces,rows=None):
 f=np.asarray(faces,np.int64)
 if rows is not None:f=f[np.asarray(rows,np.int64)]
 q=np.asarray(points,np.float64)[f];a=np.linalg.norm(q[:,1]-q[:,0],axis=1);b=np.linalg.norm(q[:,2]-q[:,1],axis=1);c=np.linalg.norm(q[:,0]-q[:,2],axis=1);short=np.minimum(np.minimum(a,b),c);long=np.maximum(np.maximum(a,b),c);area2=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1);alt=area2/np.maximum(long,1e-300)
 def stats(x):return {'min_m':float(x.min()) if len(x) else None,'p1_m':float(np.quantile(x,.01)) if len(x) else None,'p5_m':float(np.quantile(x,.05)) if len(x) else None,'median_m':float(np.median(x)) if len(x) else None,'count_below_5um':int(np.count_nonzero(x<5e-6))}
 return {'faces':len(f),'min_edge':stats(short),'min_altitude':stats(alt),'zero_area_faces':int(np.count_nonzero(area2==0)),'nonfinite_faces':int(np.count_nonzero(~np.isfinite(area2)))}
def cross(Pts,F):
 q=Pts[F];return np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
def gate(base,now,F):
 b=cross(base,F);n=cross(now,F);b2=np.einsum('ij,ij->i',b,b);n2=np.einsum('ij,ij->i',n,n);dots=np.einsum('ij,ij->i',b,n)
 return {'faces':len(F),'candidate_zero':int(np.count_nonzero(n2==0)),'candidate_nonfinite':int(np.count_nonzero(~np.isfinite(n2))),'reversed':int(np.count_nonzero(dots<=0)),'reversed_face_rows':np.flatnonzero(dots<=0).astype(int).tolist(),'zero_face_rows':np.flatnonzero(n2==0).astype(int).tolist(),'min_area_ratio':float(np.min(np.sqrt(n2/np.maximum(b2,1e-300)))),'quality':tri_quality(now,F)}
def orientation(a,b):
 a=tuple(map(int,a));b=tuple(map(int,b))
 for k in range(3):
  if a[k:]+a[:k]==b:return 1
  if (a[k],a[(k-1)%3],a[(k-2)%3])==b:return -1
 return 0
def open_closed_map(openF,closedF,offset,name):
 table=collections.defaultdict(list)
 for r,t in enumerate(closedF):table[tuple(sorted(map(int,t)))].append(r)
 rows=np.full(len(openF),-1,np.int64);rel=np.zeros(len(openF),np.int8)
 for r,t in enumerate(openF):
  q=np.asarray(t,np.int64)+offset;hits=table.get(tuple(sorted(map(int,q))),[])
  if len(hits)!=1:raise ValueError(f'{name}: open face {r} maps to {len(hits)} closed faces')
  rows[r]=hits[0];rel[r]=orientation(q,closedF[hits[0]])
  if rel[r]==0:raise ValueError(f'{name}: invalid orientation row {hits[0]}')
 if len(np.unique(rows))!=len(rows):raise ValueError(f'{name}: mapping is not bijective')
 if len(set(rel.tolist()))!=1:raise ValueError(f'{name}: inconsistent global orientation relation {set(rel.tolist())}')
 return rows,rel
if OUT.exists():raise FileExistsError(OUT)
for p,h in [(IN,'1726565b280a94f08e542d1e335b61b496fdc6bfdec78b57d161435452a35fa0'),(OP,'2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a'),(IP,'5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410')]:
 if sha(p)!=h:raise ValueError(f'input SHA mismatch: {p}')
D=load(IN);O=load(OP);I0=load(IP);Fld=load(FIELD);S=load(SKIN);Fat=load(FAT)
P0=np.asarray(D['points_world_m'],np.float64);F0=np.asarray(D['faces'],np.int64);nface=len(F0);nvert=len(P0)
if not np.array_equal(P0,np.asarray(I0['points_world_m'],np.float64)):raise ValueError('run006 and pinned I points differ')
# Decode existing operation ancestry from the first 15 accepted flips.
off=np.asarray(D['edge_flip_parent_offsets'],np.int64);par=np.asarray(D['edge_flip_parent_face_rows'],np.int64)
if len(off)!=nface+1 or off[0]!=0 or off[-1]!=len(par) or np.any(np.diff(off)<0):raise ValueError('malformed initial face-parent CSR')
oldparents=[set(map(int,par[off[r]:off[r+1]])) for r in range(nface)]
if any(not q for q in oldparents):raise ValueError('empty initial face ancestry')
inc0,_=edge_data(F0);boundary0=set(e for e,rs in inc0.items() if len(rs)==1)
neighbor=[set() for _ in range(nvert)]
for t in F0:
 a,b,c=map(int,t);neighbor[a]|={b,c};neighbor[b]|={a,c};neighbor[c]|={a,b}
boundv={v for e in boundary0 for v in e}
vertex_map=np.arange(nvert,dtype=np.int64)
collapse_info=[];affected_by_op=[]
for keep,remove in COLLAPSES:
 e=(min(keep,remove),max(keep,remove));rows=inc0.get(e,[])
 if len(rows)!=2:raise ValueError(f'collapse edge {e} incidence={len(rows)}')
 common=neighbor[keep]&neighbor[remove];opposite={int(v) for r in rows for v in F0[r] if int(v) not in e}
 if common!=opposite:raise ValueError(f'link condition fail {e}: common={common} opposite={opposite}')
 if keep in boundv or remove in boundv:raise ValueError(f'collapse touches open boundary {e}')
 if vertex_map[remove]!=remove or vertex_map[keep]!=keep:raise ValueError('collapse edges overlap')
 d=float(np.linalg.norm(P0[keep].astype(np.float32).astype(np.float64)-P0[remove].astype(np.float32).astype(np.float64)))
 vertex_map[remove]=keep
 star=set(np.flatnonzero(np.any(F0==remove,axis=1)).tolist())
 affected_by_op.append(star)
 collapse_info.append({'keep_input_vertex_row':keep,'retire_input_vertex_row':remove,'edge_length_m':d,'link_common_vertices':sorted(common),'link_opposite_vertices':sorted(opposite),'incident_face_rows':sorted(rows),'star_face_rows':sorted(star),'max_retired_point_to_retained_coordinate_m':d,'boundary_vertices_in_edge':False})
if affected_by_op[0]&affected_by_op[1]:raise ValueError('two collapse stars overlap; simultaneous contraction not independent')
remapped=vertex_map[F0]
keep_face=(remapped[:,0]!=remapped[:,1])&(remapped[:,1]!=remapped[:,2])&(remapped[:,2]!=remapped[:,0])
kept_old_face_rows=np.flatnonzero(keep_face);dropped_old_face_rows=np.flatnonzero(~keep_face)
if len(dropped_old_face_rows)!=4:raise ValueError(f'expected two target triangles per collapse, got {len(dropped_old_face_rows)}')
used=np.unique(remapped[keep_face]);old_to_new=np.full(nvert,-1,np.int64);old_to_new[used]=np.arange(len(used),dtype=np.int64)
Fnew=old_to_new[remapped[keep_face]]
face_old_to_new=np.full(nface,-1,np.int64);face_old_to_new[kept_old_face_rows]=np.arange(len(kept_old_face_rows),dtype=np.int64)
# Compose ancestry: untouched faces keep their existing CSR; each rewritten star face
# records the full original input/source CSR covered by its local collapse operation.
source_union_by_op=[set().union(*(oldparents[r] for r in star)) for star in affected_by_op]
parent_new=[];changed_old=[]
for oldrow in kept_old_face_rows:
 opids=[j for j,star in enumerate(affected_by_op) if int(oldrow) in star]
 if opids:
  changed_old.append(int(oldrow));p=set().union(*(source_union_by_op[j] for j in opids))
 else:p=oldparents[int(oldrow)]
 parent_new.append(p)
OUT.mkdir(parents=True)
# Cache compacted topology first, before downstream volume or packaging gates.
np.savez_compressed(OUT/'baseline-collapsed-I.npz',points_world_m=P0[used],faces=Fnew,old_input_vertex_to_output_row=old_to_new,old_input_face_to_output_row=face_old_to_new,kept_old_face_rows=kept_old_face_rows,dropped_old_face_rows=dropped_old_face_rows,face_parent_offsets=np.r_[0,np.cumsum([len(q) for q in parent_new])].astype(np.int64),face_parent_rows=np.asarray([r for q in parent_new for r in sorted(q)],np.int64))
# Run topology and exact local changed-face gates.
# Expected boundary remapped through collapse and compaction.
expected_boundary=set()
for a,b in boundary0:
 a,b=int(old_to_new[vertex_map[a]]),int(old_to_new[vertex_map[b]])
 if a<0 or b<0:raise ValueError('boundary vertex retired')
 expected_boundary.add(tuple(sorted((a,b))))
topo_before=topology(P0,F0,boundary0);topo_after=topology(P0[used],Fnew,expected_boundary)
if topo_after['nonmanifold_or_orientation_count'] or topo_after['duplicate_face_count'] or topo_after['bad_vertex_link_count'] or topo_after['euler']!=topo_before['euler'] or topo_after['components']!=topo_before['components']:
 raise ValueError(f'cheap topology failed: {topo_after}')
spc=importlib.util.spec_from_file_location('local_float32_edge_flip',HELPER);mod=importlib.util.module_from_spec(spc);spc.loader.exec_module(mod)
fl=mod.Flips(P0[used],Fnew,parent_new)
changed_new=np.asarray([face_old_to_new[r] for r in changed_old if face_old_to_new[r]>=0],np.int64)
if any(r is None for r in fl.recs):raise ValueError('candidate has F32-degenerate face')
local_pairs=fl.pairs(changed_new.tolist()) if len(changed_new) else 0
# Build the matching closed Skin/Fat topology by replacing precisely affected open-I face rows.
Os=np.asarray(O['points_world_m'],np.float64);Of=np.asarray(O['faces'],np.int64)
Is=np.asarray(I0['points_world_m'],np.float64);If=np.asarray(I0['faces'],np.int64)
Sp=np.asarray(S['points_world_m_f64'],np.float64);Sf=np.asarray(S['faces'],np.int64)
Fp=np.asarray(Fat['points_world_m_f64'],np.float64);Ff=np.asarray(Fat['faces'],np.int64)
if not np.array_equal(Sp[:len(Os)],Os) or not np.array_equal(Sp[len(Os):len(Os)+len(Is)],Is):raise ValueError('closed Skin O/I base mismatch')
if not np.array_equal(Fp[:len(Is)],Is):raise ValueError('closed Fat I base mismatch')
skin_map,skin_rel=open_closed_map(If,Sf,len(Os),'skin');fat_map,fat_rel=open_closed_map(If,Ff,0,'fat')
affected_old=set(changed_old)|set(map(int,dropped_old_face_rows))
remove_skin={int(skin_map[r]) for r in affected_old};remove_fat={int(fat_map[r]) for r in affected_old}
# Rows that survived collapse and were changed by a retired endpoint.
changed_kept_old=[r for r in changed_old if face_old_to_new[r]>=0]
# Baseline F32-widened vertices with the same source identity rows, except the two collapsed interior samples.
Os32=Os.astype(np.float32).astype(np.float64);Is32=Is.astype(np.float32).astype(np.float64);Sp32=Sp.astype(np.float32).astype(np.float64);Fp32=Fp.astype(np.float32).astype(np.float64)
Inbase=P0[used].astype(np.float32).astype(np.float64)
# Ensure run006 coordinates are exactly the retained run010 F32 coordinates.
if not np.array_equal(Inbase,Is32[used]):raise ValueError('collapsed I baseline differs from original F32 point rows')
extraS=Sp32[len(Os)+len(Is):];extraF=Fp32[len(Is):]
Spbase=np.vstack([Os32,Inbase,extraS]);Fpbase=np.vstack([Inbase,extraF])
# Compact old closed surface faces after deleting affected old I faces; remap vertex indices.
Skeep=np.asarray([r for r in range(len(Sf)) if r not in remove_skin],np.int64);Fkeep=np.asarray([r for r in range(len(Ff)) if r not in remove_fat],np.int64)
SF= Sf[Skeep].copy();FF=Ff[Fkeep].copy()
# Remap closed-Skin vertex indices through O/I compaction, preserving any tail vertices.
def map_skin_vertex(v):
 v=int(v)
 if v<len(Os):return v
 if v<len(Os)+len(Is):
  q=int(old_to_new[v-len(Os)])
  if q<0:raise ValueError('retired I vertex remains on retained closed Skin face')
  return len(Os)+q
 return len(Os)+len(used)+(v-len(Os)-len(Is))
def map_fat_vertex(v):
 v=int(v)
 if v<len(Is):
  q=int(old_to_new[v])
  if q<0:raise ValueError('retired I vertex remains on retained closed Fat face')
  return q
 return len(used)+(v-len(Is))
SF=np.asarray([[map_skin_vertex(v) for v in t] for t in SF],np.int64)
FF=np.asarray([[map_fat_vertex(v) for v in t] for t in FF],np.int64)
# Append surviving remapped faces in the original orientation of each closed surface.
addS=[];addF=[]
for oldrow in changed_kept_old:
 newrow=int(face_old_to_new[oldrow]);tri=Fnew[newrow].copy()
 addS.append((tri[::-1] if skin_rel[oldrow]<0 else tri)+len(Os))
 addF.append(tri[::-1] if fat_rel[oldrow]<0 else tri)
SF=np.vstack([SF,np.asarray(addS,np.int64)]) if addS else SF
FF=np.vstack([FF,np.asarray(addF,np.int64)]) if addF else FF
# Closed baseline volumes and topology delta before the shared field.
old_skin_v=volume(Sp32,Sf);old_fat_v=volume(Fp32,Ff);new_skin_v=volume(Spbase,SF);new_fat_v=volume(Fpbase,FF)
# Decode the corrected run-005 shared world displacement field.
UO=np.zeros_like(Os);UI=np.zeros_like(Is)
UO[np.asarray(Fld['O_rows'],np.int64)]=Fld['O_displacement_per_unit'];UI[np.asarray(Fld['I_rows'],np.int64)]=Fld['I_displacement_per_unit']
UIn=UI[used]
def emit(alpha):
 o=(Os+alpha*UO).astype(np.float32);i=(Inbase+alpha*UIn).astype(np.float32)
 sp=Spbase.copy();sp[:len(Os)]=o.astype(np.float64);sp[len(Os):len(Os)+len(used)]=i.astype(np.float64)
 fp=Fpbase.copy();fp[:len(used)]=i.astype(np.float64)
 return o,i,sp,fp
def delta_ml(alpha):
 _,_,sp,fp=emit(alpha);return (volume(sp,SF)-new_skin_v)*1e6,(volume(fp,FF)-new_fat_v)*1e6
lo=0.;hi=.004;target_reached=delta_ml(hi)[1]>=160.
if target_reached:
 for _ in range(32):
  mid=(lo+hi)/2
  if delta_ml(mid)[1]<160.:lo=mid
  else:hi=mid
alpha=hi if target_reached else .004
skinmove,fatmove=delta_ml(alpha);O32,I32,Spout,Fpout=emit(alpha)
# 24-step native-F32 cheap gate, with all failing face rows and quality distribution retained.
steps=[];failed=[]
for k in range(1,25):
 a=alpha*k/24;o,i,sp,fp=emit(a);go=gate(Os.astype(np.float32),o,Of);gi=gate(Inbase.astype(np.float32),i,Fnew)
 r={'step':k,'amplitude_mm':a*1000,'O':go,'I':gi};steps.append(r)
 if any(r[q][x] for q in ('O','I') for x in ('candidate_zero','candidate_nonfinite','reversed')):
  failed.append({'step':k,'amplitude_mm':a*1000,'O_reversed_face_rows':go['reversed_face_rows'],'O_zero_face_rows':go['zero_face_rows'],'I_reversed_face_rows':gi['reversed_face_rows'],'I_zero_face_rows':gi['zero_face_rows']})
closed_skin=gate(Spbase,Spout,SF);closed_fat=gate(Fpbase,Fpout,FF)
# Compose retained face metadata and full source CSR into a compact baseline I candidate.
Iout={}
for key,val in D.items():
 if key in ('faces','edge_flip_parent_offsets','edge_flip_parent_face_rows'):continue
 arr=np.asarray(val)
 if arr.ndim and arr.shape[0]==nface:Iout[key]=arr[kept_old_face_rows]
 else:Iout[key]=val
Iout['points_world_m']=I32.astype(np.float64);Iout['faces']=Fnew
Iout['edge_flip_parent_offsets']=np.r_[0,np.cumsum([len(q) for q in parent_new])].astype(np.int64)
Iout['edge_flip_parent_face_rows']=np.asarray([r for q in parent_new for r in sorted(q)],np.int64)
# Vertex retirement map is explicit; unrelated rows are identity or compacted, no face/tet relabeling.
np.savez_compressed(OUT/'candidate-O-f32.npz',**{**O,'points_world_m':O32.astype(np.float64)})
np.savez_compressed(OUT/'candidate-I-f32-collapsed-ancestry.npz',**Iout)
np.savez_compressed(OUT/'closed-emitted-surfaces.npz',skin_points_world_m_f64=Spout,skin_faces=SF,fat_points_world_m_f64=Fpout,fat_faces=FF)
np.savez_compressed(OUT/'vertex-face-remap.npz',input_vertex_to_output_row=old_to_new,input_face_to_output_row=face_old_to_new,kept_input_face_rows=kept_old_face_rows,dropped_input_face_rows=dropped_old_face_rows,changed_output_face_rows=changed_new)
receipt={'schema':'numi.paired-head-reference-fit-684.link-collapse-field.v1','physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'inputs':{str(x):sha(x) for x in [IN,OP,IP,FIELD,SKIN,FAT,HELPER]},'collapse_operations':collapse_info,'changed_old_face_rows':sorted(changed_old),'dropped_old_face_rows':dropped_old_face_rows.tolist(),'changed_output_face_rows':changed_new.tolist(),'parent_ancestry':'face_parent_offsets/rows covers all current source ancestry; rewritten faces in each local star carry union of that collapse star CSR, including dropped source rows','topology_before':topo_before,'topology_after':topo_after,'boundary_edges_unchanged':True,'coordinates_of_retained_vertices_unchanged':True,'exact_local_self_pairs_touching_changed_faces':local_pairs,'exact_full_self_not_run':True,'volume':{'target_fat_recovery_mL':160.,'amplitude_mm':alpha*1000,'target_reached':target_reached,'skin_delta_movement_mL':skinmove,'fat_delta_movement_mL':fatmove,'skin_delta_topology_mL':(new_skin_v-old_skin_v)*1e6,'fat_delta_topology_mL':(new_fat_v-old_fat_v)*1e6,'regional_skin_estimate_mL':SKIN_BASE_REGION_ML+skinmove+(new_skin_v-old_skin_v)*1e6,'regional_skin_cap_mL':SKIN_CAP_REGION_ML,'regional_fat_estimate_mL':FAT_BASE_REGION_ML+fatmove+(new_fat_v-old_fat_v)*1e6,'regional_fat_min_mL':FAT_MIN_REGION_ML,'regionalization_caveat':'global closed-surface deltas assigned to the targeted region as an estimate; no mass/admission claim'},'f32_substeps':{'count':24,'failed_steps':failed,'distinct_O_reversed_face_rows':sorted({r for x in steps for r in x['O']['reversed_face_rows']}),'distinct_I_reversed_face_rows':sorted({r for x in steps for r in x['I']['reversed_face_rows']}),'distinct_I_zero_face_rows':sorted({r for x in steps for r in x['I']['zero_face_rows']}),'O_final':steps[-1]['O'],'I_final':steps[-1]['I'],'closed_skin_final':closed_skin,'closed_fat_final':closed_fat},'quality_baseline':{'O':tri_quality(Os.astype(np.float32),Of),'I':tri_quality(Inbase.astype(np.float32),Fnew),'closed_skin':tri_quality(Spbase,SF),'closed_fat':tri_quality(Fpbase,FF)},'quality_final':{'O':tri_quality(O32,Of),'I':tri_quality(I32,Fnew),'closed_skin':tri_quality(Spout,SF),'closed_fat':tri_quality(Fpout,FF)},'limitations':['No full exact self, O-I cross, nesting, or all-185-bone query has run.','All gates here are bounded local/cheap F32/topology checks; no physical admission.','Historical single-face source identity tags on changed rows remain historical; multi-parent CSR is authoritative and requires typed consumer review.','No TetGen or native runtime was run.'],'outputs':{}}
# Store exact generating source alongside the output for reproducibility.
(OUT/'source').mkdir();shutil.copy2(Path(__file__),OUT/'source'/Path(__file__).name)
for q in OUT.iterdir():
 if q.is_file():receipt['outputs'][q.name]=sha(q)
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'status':'PASS_CHEAP' if not failed and closed_skin['candidate_zero']==0 and closed_fat['candidate_zero']==0 and target_reached and receipt['volume']['regional_skin_estimate_mL']<=SKIN_CAP_REGION_ML and receipt['volume']['regional_fat_estimate_mL']>=FAT_MIN_REGION_ML else 'FAIL_CHEAP','collapse_count':len(COLLAPSES),'dropped_faces':len(dropped_old_face_rows),'changed_faces':len(changed_new),'local_exact_pairs':local_pairs,'topology':topo_after,'alpha_mm':alpha*1000,'fat_movement_mL':fatmove,'skin_movement_mL':skinmove,'regional_skin_mL':receipt['volume']['regional_skin_estimate_mL'],'regional_fat_mL':receipt['volume']['regional_fat_estimate_mL'],'f32_failed_substeps':failed,'closed_skin_zero':closed_skin['candidate_zero'],'closed_fat_zero':closed_fat['candidate_zero']},indent=2))
