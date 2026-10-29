#!/usr/bin/env python3
from pathlib import Path
import collections, hashlib, importlib.util, json, math, time
import numpy as np
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001')
OUT=B/'run-042-abstract-cycle-coupled-remesh'
INPUT=B/'run-022-guide-plane-star-remesh/candidate-f64.npz'
PAIR_REPORT=B/'run-027-exact-self-witness-classification/report.json'
HELPER=B/'remesh_d_witness_stars_rerun022.py'
TOPO=B/'contract_d_connected_f32_classes_rerun013.py'
EMITTER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/emit_f32.py')
BUFFER_RUNS={704434,706662,709339,715855}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def edge_map(F):
 em=collections.defaultdict(set)
 for r,(a,b,c) in enumerate(F):
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):em[tuple(sorted((u,v)))].add(r)
 return em
def connected_components(rows,em):
 rows=set(rows); adj=collections.defaultdict(set)
 for rs in em.values():
  local=rs&rows
  if len(local)>1:
   for r in local:adj[r].update(local-{r})
 out=[]
 while rows:
  start=min(rows);stack=[start];comp=set()
  while stack:
   r=stack.pop()
   if r in comp:continue
   comp.add(r);rows.discard(r);stack.extend(adj[r]-comp)
  out.append(comp)
 return out
def constrained_ear_clip(loop,P,guide,forbidden_edges):
 xyz=P[np.asarray(loop,np.int64)];centered=xyz-xyz.mean(0);gn=np.asarray(guide,np.float64)/np.linalg.norm(guide)
 u=centered[1].copy();u-=float(u@gn)*gn
 if np.linalg.norm(u)<=1e-15:
  axis=np.eye(3)[int(np.argmin(np.abs(gn)))];u=np.cross(axis,gn)
 u/=np.linalg.norm(u);v=np.cross(gn,u);uv=centered@np.stack((u,v),axis=1)
 scale=max(float(np.linalg.norm(uv.max(0)-uv.min(0))),1e-30);eps=np.finfo(float).eps*scale*scale*32
 signed=float(sum(uv[i,0]*uv[(i+1)%len(loop),1]-uv[(i+1)%len(loop),0]*uv[i,1] for i in range(len(loop)))*0.5)
 if abs(signed)<=eps:raise ValueError('projected boundary loop has zero area')
 orient=1.0 if signed>0 else -1.0
 boundary={tuple(sorted((int(loop[i]),int(loop[(i+1)%len(loop)])))) for i in range(len(loop))}
 forbidden={tuple(sorted(map(int,e))) for e in forbidden_edges}-boundary
 def cr(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
 def inside(p,a,b,c):
  x=orient*cr(a,b,p);y=orient*cr(b,c,p);z=orient*cr(c,a,p);return x>=-eps and y>=-eps and z>=-eps
 def qscore(a,b,c):
  area2=orient*cr(uv[a],uv[b],uv[c]);lens=[np.linalg.norm(uv[a]-uv[b]),np.linalg.norm(uv[b]-uv[c]),np.linalg.norm(uv[c]-uv[a])]
  return float((2*math.sqrt(3.0)*area2)/(sum(x*x for x in lens)+1e-300))
 def edgeok(a,b):return tuple(sorted((int(loop[a]),int(loop[b])))) not in forbidden
 def search(ids):
  if len(ids)==3:
   a,b,c=ids
   if orient*cr(uv[a],uv[b],uv[c])<=eps:return None
   if not all(edgeok(x,y) for x,y in ((a,b),(b,c),(c,a))):return None
   n=np.cross(P[loop[b]]-P[loop[a]],P[loop[c]]-P[loop[a]])
   if float(n@gn)<=0:return None
   return [(loop[a],loop[b],loop[c])],[qscore(a,b,c)]
  ears=[]
  for j,cur in enumerate(ids):
   prev=ids[(j-1)%len(ids)];nxt=ids[(j+1)%len(ids)];a,b,c=uv[prev],uv[cur],uv[nxt]
   if orient*cr(a,b,c)<=eps or not all(edgeok(x,y) for x,y in ((prev,cur),(cur,nxt),(nxt,prev))):continue
   n=np.cross(P[loop[cur]]-P[loop[prev]],P[loop[nxt]]-P[loop[prev]])
   if float(n@gn)<=0:continue
   if any(inside(uv[k],a,b,c) for k in ids if k not in (prev,cur,nxt)):continue
   ears.append((qscore(prev,cur,nxt),cur,j,prev,nxt))
  for q,cur,j,prev,nxt in sorted(ears,key=lambda x:(-x[0],x[1])):
   result=search(ids[:j]+ids[j+1:])
   if result is not None:
    faces,scores=result;return [(loop[prev],loop[cur],loop[nxt])]+faces,[q]+scores
  return None
 result=search(tuple(range(len(loop))))
 if result is None:raise ValueError('no outward-wound boundary triangulation avoids external occupied edges')
 out,scores=result
 return out,{'projected_signed_area_m2':signed,'projection_max_residual_m':float(np.max(np.abs(centered@gn))),'min_projected_triangle_quality':float(min(scores)),'outward_projection_normal':gn.tolist(),'external_forbidden_edge_count':len(forbidden)}
def load_exact_owner():
 import sys,types
 base=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001')
 sys.path.insert(0,str(base/'tools'))
 pkg=types.ModuleType('numi_tools');pkg.__path__=[str(base/'tools')];sys.modules['numi_tools']=pkg
 for name,attrs in [('cardiac_cavity_geometry',{'analyze_topology':lambda *a,**k:None}),('model',{'ImportError':RuntimeError}),('physiology',{'canonical':lambda x:x})]:
  m=types.ModuleType('numi_tools.'+name);m.__dict__.update(attrs);sys.modules[m.__name__]=m
 import numi_tools.cardiac_cavity_intersections as pred
 return pred

def collision_aware_ear_clip(loop,P,P32,external_edges,base_index,pred,lattice,previous_records,first_candidate_id,max_nodes=25000,max_seconds=10.0):
 # The boundary is an abstract oriented cycle; no physical plane is assumed.
 n=len(loop);ang=2.0*math.pi*np.arange(n,dtype=np.float64)/n;uv=np.stack((np.cos(ang),np.sin(ang)),axis=1)
 eps=np.finfo(float).eps*128;boundary={tuple(sorted((int(loop[i]),int(loop[(i+1)%n])))) for i in range(n)}
 forbidden={tuple(sorted(map(int,e))) for e in external_edges}-boundary
 def cr(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
 def inside(p,a,b,c):
  x=cr(a,b,p);y=cr(b,c,p);z=cr(c,a,p);return x>=-eps and y>=-eps and z>=-eps
 def qscore(a,b,c):
  area=cr(uv[a],uv[b],uv[c]);lens=[np.linalg.norm(uv[a]-uv[b]),np.linalg.norm(uv[b]-uv[c]),np.linalg.norm(uv[c]-uv[a])]
  return float((2*math.sqrt(3.0)*area)/(sum(x*x for x in lens)+1e-300))
 def edgeok(a,b):return tuple(sorted((int(loop[a]),int(loop[b])))) not in forbidden
 stats={'nodes':0,'triangle_candidates':0,'exact_external_calls':0,'exact_external_pairs':0,'exact_internal_calls':0,'exact_internal_pairs':0,'external_cache_hits':0,'internal_cache_hits':0,'backtracks':0,'max_depth':0}
 start=time.monotonic();nextid=[int(first_candidate_id)];record_cache={};external_cache={};pair_cache={}
 def tri_key(tri):return tuple(sorted(map(int,tri)))
 def make_rec(tri):
  key=tri_key(tri)
  if key not in record_cache:
   raw=pred._records(lattice,[tuple(map(int,tri))])[0];record_cache[key]=(raw[0],raw[1],raw[2],nextid[0],raw[4]);nextid[0]+=1
  return record_cache[key]
 def clean_external(rec,tri):
  key=tri_key(tri)
  if key in external_cache:stats['external_cache_hits']+=1;return external_cache[key]
  stats['exact_external_calls']+=1;res=pred._audit_pair_prepared_first(base_index,(rec,),same_surface=True);ok=int(res['count'])==0
  stats['exact_external_pairs']+=int(res['count']);external_cache[key]=ok;return ok
 def clean_chosen(rec,tri,chosen):
  for other,otri in chosen:
   key=tuple(sorted((tri_key(tri),tri_key(otri))))
   if key in pair_cache:stats['internal_cache_hits']+=1;ok=pair_cache[key]
   else:
    stats['exact_internal_calls']+=1;res=pred._audit_record_pairs(((rec,other),),same_surface=True);ok=int(res['count'])==0
    stats['exact_internal_pairs']+=int(res['count']);pair_cache[key]=ok
   if not ok:return False
  return True
 def valid_triangle(a,b,c,ids):
  if cr(uv[a],uv[b],uv[c])<=eps:return None
  if not all(edgeok(x,y) for x,y in ((a,b),(b,c),(c,a))):return None
  tri=(int(loop[a]),int(loop[b]),int(loop[c]))
  xyz=P[np.asarray(tri,np.int64)];xyz32=P32[np.asarray(tri,np.int64)].astype(np.float64)
  n64=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]);n32=np.cross(xyz32[1]-xyz32[0],xyz32[2]-xyz32[0])
  if not np.all(np.isfinite(n64)) or not np.any(n64) or not np.all(np.isfinite(n32)) or not np.any(n32):return None
  if any(inside(uv[k],uv[a],uv[b],uv[c]) for k in ids if k not in (a,b,c)):return None
  return qscore(a,b,c)
 def search(ids,chosen_faces,chosen_pairs,depth):
  stats['nodes']+=1;stats['max_depth']=max(stats['max_depth'],depth)
  if stats['nodes']>max_nodes or time.monotonic()-start>max_seconds:raise TimeoutError('bounded abstract-cycle ear search limit reached')
  if len(ids)==3:
   a,b,c=ids;q=valid_triangle(a,b,c,ids)
   if q is None:return None
   tri=(int(loop[a]),int(loop[b]),int(loop[c]));rec=make_rec(tri);stats['triangle_candidates']+=1
   if not clean_external(rec,tri) or not clean_chosen(rec,tri,chosen_pairs):return None
   return chosen_faces+[tri],chosen_pairs+[(rec,tri)],[q]
  ears=[]
  for j,cur in enumerate(ids):
   prev=ids[(j-1)%len(ids)];nxt=ids[(j+1)%len(ids)];q=valid_triangle(prev,cur,nxt,ids)
   if q is not None:ears.append((q,cur,j,prev,nxt))
  for q,cur,j,prev,nxt in sorted(ears,key=lambda x:(-x[0],x[1])):
   tri=(int(loop[prev]),int(loop[cur]),int(loop[nxt]));rec=make_rec(tri);stats['triangle_candidates']+=1
   if not clean_external(rec,tri) or not clean_chosen(rec,tri,chosen_pairs):continue
   result=search(ids[:j]+ids[j+1:],chosen_faces+[tri],chosen_pairs+[(rec,tri)],depth+1)
   if result is not None:
    faces,recs,scores=result;return faces,recs,[q]+scores
   stats['backtracks']+=1
  return None
 result=search(tuple(range(n)),[],list(previous_records),0)
 if result is None:raise ValueError('no exact-clean abstract-cycle triangulation: '+json.dumps(stats,separators=(',',':')))
 faces,recs,scores=result
 return faces,[r for r,t in recs],{'basis':'abstract_unit_circle_in_boundary_order','boundary_vertex_count':n,'min_unit_circle_triangle_quality':float(min(scores)),'occupied_external_diagonal_count':len(forbidden),'exact_search':stats,'elapsed_s':time.monotonic()-start}
def main():
 t0=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
 with np.load(INPUT,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 P=np.asarray(d['points_world_m_f64'],np.float64);F=np.asarray(d['faces_f64'],np.int64)
 R=json.load(open(PAIR_REPORT));pairhash=sha(PAIR_REPORT)
 attrs={k:np.asarray(d[k]) for k in ('face_cache_row','face_source_layer','face_source_body','face_source_bone','face_run_original_id','face_id','face_source_identity0','face_source_identity1')}
 run=attrs['face_run_original_id'].astype(np.int64);layer=attrs['face_source_layer'].astype(np.int64)
 # Build coupled source+buffer witness patches from all 51 exact pairs.
 # Original source rows are only edited inside these face-adjacency patches.
 seed_all=set();pair_classes=collections.Counter();witness_records=[]
 for pi,pair in enumerate(R['pairs']):
  faces=pair['faces'];rows=[int(q['candidate_face_row']) for q in faces]
  cls=['source' if int(q['source_layer'])==2 else 'buffer' if int(q['source_layer'])==3 else 'other' for q in faces]
  if any(int(attrs['face_cache_row'][r])!=int(q['source_cache_row']) or int(run[r])!=int(q['source_run_original_id']) or int(layer[r])!=int(q['source_layer']) for r,q in zip(rows,faces)):
   raise ValueError(f'exact witness row/source map mismatch at pair {pi}')
  if any(c=='other' for c in cls):raise ValueError(f'unhandled exact pair classes {cls}')
  pair_classes['+'.join(sorted(cls))]+=1;seed_all.update(rows)
  witness_records.append({'pair_index':pi,'face_rows':rows,'classes':cls,'source_cache_rows':[int(q['source_cache_row']) for q in faces]})
 em=edge_map(F);allowed=set(np.flatnonzero(np.isin(layer,[2,3])).tolist())
 selected=set(seed_all)
 for r in seed_all:
  a,b,c=map(int,F[r])
  for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):selected |= em[e]&allowed
 patch_specs=[]
 for comp in connected_components(selected,em):patch_specs.append({'role':'coupled_source_buffer','seed_rows':sorted(comp&seed_all),'rows':comp})
 # Expand only the failing component containing exact-source cache row 15423 by
 # one additional 2/3-layer edge ring, then recompute connected components.
 anchor_cache_row=15423
 anchor=next((q for q in patch_specs if anchor_cache_row in set(map(int,attrs['face_cache_row'][sorted(q['rows'])]))),None)
 if anchor is None:raise ValueError('witness-directed coupled expansion anchor is absent')
 anchor_rows=set(anchor['rows']);ring=set()
 for r in anchor_rows:
  a,b,c=map(int,F[r])
  for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):ring |= em[e]&allowed
 ring -= selected;selected |= ring
 patch_specs=[]
 for comp in connected_components(selected,em):patch_specs.append({'role':'coupled_source_buffer','seed_rows':sorted(comp&seed_all),'rows':comp})
 patch_specs.sort(key=lambda x:min(x['rows']))
 expansion={'anchor_cache_row':anchor_cache_row,'added_face_rows':sorted(ring),'added_face_count':len(ring),'expanded_component_faces_before':len(anchor_rows),'expanded_component_faces_after':sum(len(q['rows']) for q in patch_specs if anchor_cache_row in set(map(int,attrs['face_cache_row'][sorted(q['rows'])])))}
 helper=loadmod('run022_helpers',HELPER);topo=loadmod('topology_tools',TOPO);pred=load_exact_owner()
 off=np.asarray(d['face_ancestry_offsets'],np.int64);ar=np.asarray(d['face_ancestry_input_cache_rows'],np.int64)
 ances=[set(map(int,ar[off[i]:off[i+1]])) for i in range(len(F))]
 patch_reports=[];all_remove=set().union(*(p['rows'] for p in patch_specs));remove=set();new_faces=[];new_anc=[];new_attrs={k:[] for k in attrs}; volume_delta=0.;area_delta=0.
 keepbase=np.ones(len(F),bool);keepbase[list(all_remove)]=False;baseF=F[keepbase]
 P32=P.astype(np.float32);lattice=tuple(pred.float32_point_lattice_key(p) for p in P32)
 base_records,_=pred.float32_surface_records(P32,baseF,source_seam_groups=());base_index=pred._prepare_surface_aabb(base_records)
 base_edge_set=set()
 for aa,bb,cc in baseF:
  base_edge_set.update((tuple(sorted((int(aa),int(bb)))),tuple(sorted((int(bb),int(cc)))),tuple(sorted((int(cc),int(aa))))))
 previous_records=[];next_candidate_id=len(baseF)+1
 for pi,patch in enumerate(patch_specs):
  rows=sorted(map(int,patch['rows']))
  if remove.intersection(rows):raise ValueError('patch face sets overlap across role domains')
  remove.update(rows)
  loops,bd=helper.boundary_cycles(F,rows)
  if len(loops)!=1:raise ValueError(f"{patch['role']} patch {pi} has {len(loops)} boundary loops")
  loop=loops[0]
  guide=np.sum(np.cross(P[F[rows]][:,1]-P[F[rows]][:,0],P[F[rows]][:,2]-P[F[rows]][:,0]),axis=0)
  external_edges=set(base_edge_set)
  for pf in new_faces:
   aa,bb,cc=map(int,pf);external_edges.update((tuple(sorted((aa,bb))),tuple(sorted((bb,cc))),tuple(sorted((cc,aa)))))
  try:
   tri,newrecs,planar=collision_aware_ear_clip(loop,P,P32,external_edges,base_index,pred,lattice,previous_records,next_candidate_id,max_nodes=25000,max_seconds=10.0)
  except Exception as exc:
   failure={'patch_index':pi,'role':patch['role'],'seed_face_rows':patch['seed_rows'],'removed_face_rows':rows,'removed_cache_rows':sorted(set(map(int,attrs['face_cache_row'][rows]))),'boundary_loop_vertex_rows':loop,'guide_vector':guide.tolist(),'error':repr(exc),'candidate_input_sha256':sha(INPUT),'exact_pair_report_sha256':pairhash,'witness_directed_expansion':expansion,'prior_successful_patch_reports':patch_reports,'admission':False}
   (OUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
   raise
  print(json.dumps({'patch_index':pi,'role':patch['role'],'faces':len(rows),'loop':len(loop),'new_faces':len(tri),'search':planar['exact_search']}),flush=True)
  next_candidate_id += len(tri)+100
  previous_records.extend(newrecs);NF=np.asarray(tri,np.int64)
  # Assert oriented patch boundary is unchanged exactly.
  oldbd=collections.Counter(tuple(x) for x in bd)
  new_edges=collections.defaultdict(list)
  for a,b,c in NF:
   for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):new_edges[tuple(sorted((u,v)))].append((u,v))
  newbd=collections.Counter(vals[0] for vals in new_edges.values() if len(vals)==1)
  if oldbd!=newbd:raise ValueError(f'patch {pi} altered directed boundary')
  if any(len(vals)>2 for vals in new_edges.values()):raise ValueError(f'patch {pi} has nonmanifold internal edges')
  for vals in new_edges.values():
   if len(vals)==2 and vals[0]!=(vals[1][1],vals[1][0]):raise ValueError(f'patch {pi} has inconsistent internal edge winding')
  source_union=set().union(*(ances[r] for r in rows))
  origins=sorted(set(map(int,run[rows])))
  cats=sorted(set((int(attrs['face_source_layer'][r]),int(attrs['face_source_body'][r]),int(attrs['face_source_bone'][r])) for r in rows))
  for nf in NF:
   new_faces.append(nf);new_anc.append(set(source_union))
   for k,v in (('face_cache_row',-1),('face_source_layer',cats[0][0] if len({c[0] for c in cats})==1 else -1),('face_source_body',cats[0][1] if len({c[1] for c in cats})==1 else -1),('face_source_bone',cats[0][2] if len({c[2] for c in cats})==1 else -1),('face_run_original_id',origins[0] if len(origins)==1 else -1),('face_id',-1),('face_source_identity0',-1),('face_source_identity1',-1)):
    new_attrs[k].append(v)
  oldtri=P[F[np.asarray(rows,np.int64)]];newtri=P[NF]
  v0=float(np.einsum('ij,ij->i',oldtri[:,0],np.cross(oldtri[:,1],oldtri[:,2])).sum()/6.0);v1=float(np.einsum('ij,ij->i',newtri[:,0],np.cross(newtri[:,1],newtri[:,2])).sum()/6.0)
  a0=float(np.linalg.norm(np.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0]),axis=1).sum()/2);a1=float(np.linalg.norm(np.cross(newtri[:,1]-newtri[:,0],newtri[:,2]-newtri[:,0]),axis=1).sum()/2)
  volume_delta+=v1-v0;area_delta+=a1-a0
  patch_reports.append({'patch':pi,'role':patch['role'],'seed_face_rows':patch['seed_rows'],'removed_face_rows':rows,'removed_cache_rows':sorted(set(map(int,attrs['face_cache_row'][rows]))),'removed_faces':len(rows),'boundary_vertices':loop,'boundary_count':len(loop),'new_faces':int(len(NF)),'source_run_ids':origins,'source_categories':[{'layer':x[0],'body':x[1],'bone':x[2]} for x in cats],'ancestry_input_cache_row_count':len(source_union),'triangulation':planar,'volume_delta_m3':v1-v0,'area_delta_m2':a1-a0})
 keep=np.ones(len(F),bool);keep[list(remove)]=False
 Fout=np.vstack((F[keep],np.asarray(new_faces,np.int64)));outrow=np.r_[np.flatnonzero(keep).astype(np.int64),np.full(len(new_faces),-1,np.int64)]
 outattrs={k:np.concatenate((np.asarray(v)[keep],np.asarray(new_attrs[k],dtype=np.asarray(v).dtype))) for k,v in attrs.items()}
 ancesout=[ances[r] for r in np.flatnonzero(keep)]+new_anc
 offsets=[0];arows=[]
 for s0 in ancesout:arows.extend(sorted(s0));offsets.append(len(arows))
 offsets=np.asarray(offsets,np.int64);arows=np.asarray(arows,np.int64)
 meta={}
 for row,oid,fid,i0,i1 in zip(np.asarray(d['face_ancestry_input_cache_rows'],np.int64),np.asarray(d['face_ancestry_original_ids'],np.int64),np.asarray(d['face_ancestry_face_ids'],np.int64),np.asarray(d['face_ancestry_identity0'],np.int64),np.asarray(d['face_ancestry_identity1'],np.int64)):meta[int(row)]=(int(oid),int(fid),int(i0),int(i1))
 if any(int(r) not in meta for r in arows):raise ValueError('ancestry metadata missing')
 aoid=np.asarray([meta[int(r)][0] for r in arows],np.int64);afid=np.asarray([meta[int(r)][1] for r in arows],np.int64);ai0=np.asarray([meta[int(r)][2] for r in arows],np.int64);ai1=np.asarray([meta[int(r)][3] for r in arows],np.int64)
 topo64=topo.topology(P,Fout);loops64=topo.loops(Fout)
 srcQ=np.asarray(d['source_points_f32'],np.float32);Q=P.astype(np.float32);T=Q[Fout].astype(np.float64);cr=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);zero_pre=np.flatnonzero(np.all(cr==0.,axis=1))
 patchmeta={'operation':'exact_intersection_constrained_connected_star_remesh','pair_report_sha256':pairhash,'pair_classes':dict(pair_classes),'patches':patch_reports}
 packed=dict(points_world_m_f64=P,faces_f64=Fout,vertex_input_to_representative=d['vertex_input_to_representative'],face_cache_row=outattrs['face_cache_row'],face_source_layer=outattrs['face_source_layer'],face_source_body=outattrs['face_source_body'],face_source_bone=outattrs['face_source_bone'],face_run_original_id=outattrs['face_run_original_id'],face_id=outattrs['face_id'],face_source_identity0=outattrs['face_source_identity0'],face_source_identity1=outattrs['face_source_identity1'],face_ancestry_offsets=offsets,face_ancestry_input_cache_rows=arows,face_ancestry_original_ids=aoid,face_ancestry_face_ids=afid,face_ancestry_identity0=ai0,face_ancestry_identity1=ai1,source_points_f32=srcQ,removed_exact_cap_cache_rows=d['removed_exact_cap_cache_rows'],source_cache_sha256=d['source_cache_sha256'],source_geometry_sha256=np.asarray([hashlib.sha256(np.ascontiguousarray(P).view(np.uint8).tobytes()+np.ascontiguousarray(Fout).view(np.uint8).tobytes()).hexdigest()]),inherited_prior_edge_collapse_json=d.get('inherited_prior_edge_collapse_json',np.asarray(['[]'])),prior_edge_collapse_json=d.get('prior_edge_collapse_json',d.get('edge_collapse_json',np.asarray(['[]']))),edge_patch_remesh_json=np.asarray([json.dumps(patchmeta,separators=(',',':'))]))
 out64=OUT/'candidate-f64.npz';np.savez_compressed(out64,**packed)
 structure64=(topo64['component_count']==1 and topo64['boundary_edge_count']==45 and topo64['nonmanifold_edge_count']==0 and topo64['components'][0]['euler']==0 and loops64==[21,24])
 emission={'status':'NOT_RUN'};topo32=None;loops32=None;zero32=[];out32=None
 if structure64 and len(zero_pre)==0:
  emod=loadmod('numi_emit_f32',EMITTER);Y,F32,erep=emod.emit(P,Fout,srcQ)
  cr32=np.cross(Y[F32][:,1].astype(np.float64)-Y[F32][:,0].astype(np.float64),Y[F32][:,2].astype(np.float64)-Y[F32][:,0].astype(np.float64));zero32=np.flatnonzero(np.all(cr32==0.,axis=1))
  if len(zero32)==0:
   topo32=topo.topology(Y.astype(np.float64),F32);loops32=topo.loops(F32)
   out32=OUT/'candidate-f32.npz';np.savez_compressed(out32,points_world_m=Y,faces=F32,face_cache_row=packed['face_cache_row'],face_source_layer=packed['face_source_layer'],face_source_body=packed['face_source_body'],face_source_bone=packed['face_source_bone'],face_run_original_id=packed['face_run_original_id'],face_id=packed['face_id'],face_source_identity0=packed['face_source_identity0'],face_source_identity1=packed['face_source_identity1'],face_ancestry_offsets=offsets,face_ancestry_input_cache_rows=arows,face_ancestry_original_ids=aoid,face_ancestry_face_ids=afid,face_ancestry_identity0=ai0,face_ancestry_identity1=ai1,vertex_input_old_to_emitted=np.asarray(erep['input_vertex_old_to_emitted'],np.int64),vertex_emitted_input_parent_offsets=np.asarray(erep['emitted_vertex_input_parent_offsets'],np.int64),vertex_emitted_input_parent_rows=np.asarray(erep['emitted_vertex_input_parent_rows'],np.int64),vertex_emitted_source_point_offsets=np.asarray(erep['emitted_vertex_source_point_offsets'],np.int64),vertex_emitted_source_point_rows=np.asarray(erep['emitted_vertex_source_point_rows'],np.int64),float32_emission_json=np.asarray([json.dumps(erep,separators=(',',':'))]))
   emission={'status':'PASS','path':str(out32),'sha256':sha(out32),'zero_area_faces':0}
  else:emission={'status':'FAIL','zero_area_F32_rows':list(map(int,zero32))[:100],'zero_area_count':int(len(zero32))}
 else:emission={'status':'FAIL_PREEMISSION_GATE','F64_structure':bool(structure64),'preemission_zero_area_F32_count':int(len(zero_pre))}
 report={'schema':'D-selfpair-connected-star-remesh.v1','input_f64':str(INPUT),'input_f64_sha256':sha(INPUT),'input_f32':R['candidate'],'input_f32_sha256':R['candidate_sha256'],'exact_pair_report':str(PAIR_REPORT),'exact_pair_report_sha256':pairhash,'witness_directed_expansion':expansion,'exact_pair_count':int(R['pair_count']),'pair_classes':dict(pair_classes),'selected_coupled_witness_seed_faces':len(seed_all),'selected_coupled_patch_faces':len(all_remove),'output_f64':str(out64),'output_f64_sha256':sha(out64),'output_f32':str(out32) if out32 else None,'output_f32_sha256':sha(out32) if out32 else None,'input_faces':int(len(F)),'output_faces':int(len(Fout)),'changed_rows_removed':sorted(remove),'changed_cache_rows_removed':sorted(set(map(int,attrs['face_cache_row'][sorted(remove)]))),'patches':patch_reports,'F64_topology':topo64,'F64_boundary_loops':loops64,'F64_structure_pass':bool(structure64),'pre_emission_zero_area_F32_count':int(len(zero_pre)),'F32_emission':emission,'F32_topology':topo32,'F32_boundary_loops':loops32,'full_exact_self_gate':'NOT_RUN','D_inside_I_gate':'NOT_RUN','source_points_unchanged_bitwise':True,'source_layer2_faces_modified_only_inside_connected_exact_witness_patches':True,'signed_volume_delta_total_m3':volume_delta,'surface_area_delta_total_m2':area_delta,'elapsed_s':time.monotonic()-t0,'admission':False}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='patches'},indent=2))
if __name__=='__main__':main()
