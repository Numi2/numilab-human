#!/usr/bin/env python3
import argparse, collections, gc, hashlib, heapq, json, math, os, pickle, resource, signal, struct, subprocess, sys, threading, time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
ROOT=Path('/Users/numi/numi-human-fds132-m6-1350')
A=Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276')
R=Path('/Users/numi/numi-human-retained-delivery-20261009')
MAP=ROOT/'attempt-061-step0-capture-remesh-diagnostic/candidate060-adapted-source-parent-map-v2.pkl'
FIELD=ROOT/'attempt-073-shared-route-anchor-field-001/result-001/route-anchor-reference-field-001.npz'
ROUTE=ROOT/'attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json'
C=ROOT/'attempt-060-paired-cut-sliver-contraction-001/candidate.pkl'
CK=C.parent/'exact-check-001/report.json'
CI=Path('/Users/n/numi-human-anatomy-completion-1276/src/numilab_human/cardiac_cavity_intersections.py')
RIG=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
ARCH=R/'skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz'
CAP=R/'skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011/baseline/native-run/accepted-geometry'
LIMIT=int(2.5*1024*1024); START=time.monotonic(); OUT=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
def save(p,x):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z))+'\n');q.replace(p)
def watchdog():
 while True:
  try:
   rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip() or '0')
   if rss>LIMIT:
    print(json.dumps({'fatal':'RSS watchdog exceeded 2.5 GiB','rss_kib':rss}),flush=True);os.kill(os.getpid(),signal.SIGTERM)
  except Exception:pass
  time.sleep(2)
threading.Thread(target=watchdog,daemon=True).start()
def f32(x):return np.float32(x)
def cross32(a,b):
 q=np.float32
 return np.asarray([q(q(a[1]*b[2])-q(a[2]*b[1])),q(q(a[2]*b[0])-q(a[0]*b[2])),q(q(a[0]*b[1])-q(a[1]*b[0]))],dtype=np.float32)
def rotate_cxx(q,p):
 q=np.asarray(q,dtype=np.float32);p=np.asarray(p,dtype=np.float32);a=q[:3];cr=cross32(a,p);tw=np.asarray([f32(f32(2)*v) for v in cr],dtype=np.float32)
 corr=np.asarray([f32(f32(f32(f32(q[3]*tw[0])+f32(a[1]*tw[2]))-f32(a[2]*tw[1]))),f32(f32(f32(f32(q[3]*tw[1])+f32(a[2]*tw[0]))-f32(a[0]*tw[2]))),f32(f32(f32(f32(q[3]*tw[2])+f32(a[0]*tw[1]))-f32(a[1]*tw[0])))],dtype=np.float32)
 return np.asarray([f32(p[i]+corr[i]) for i in range(3)],dtype=np.float32)
def rotate_metal(q,p):
 q=np.asarray(q,dtype=np.float32);p=np.asarray(p,dtype=np.float32);a=q[:3];cr=cross32(a,p)
 inn=np.asarray([f32(cr[i]+f32(q[3]*p[i])) for i in range(3)],dtype=np.float32);out=cross32(a,inn)
 return np.asarray([f32(p[i]+f32(f32(2)*out[i])) for i in range(3)],dtype=np.float32)
def parse_nhrigid(path,archive):
 raw=Path(path).read_bytes();magic,pabi,eabi,sc,bc,jc,nq,nv,root,virt,reserved,srcsha=struct.unpack_from('<8s10I32s',raw,0)
 req(magic==b'NHRIGID2' and pabi==1 and nq==nv+1,'NHRIGID header invalid');req(srcsha.hex()==sha(archive),'NHRIGID embedded source hash mismatch')
 bo=224;jo=bo+160*bc;do=jo+144*jc;qo=do+64*nv;q=np.frombuffer(raw,dtype='<f4',count=nq,offset=qo).copy();vo=qo+4*nq;mo=vo+4*nv;stc=np.frombuffer(raw,dtype='<u4',count=sc,offset=mo).copy();po=mo+4*sc;poses={}
 for i,c in enumerate(stc):
  x,y,z,qx,qy,qz,qw=struct.unpack_from('<7f',raw,po+28*i);poses[int(c)]=(np.asarray([x,y,z],dtype=np.float32),np.asarray([qx,qy,qz,qw],dtype=np.float32))
 req(len(poses)==sc and po+28*sc==len(raw),'NHRIGID layout mismatch');return poses
def load_pose(step):
 pack=CAP/f'step-{step}.mrvpack';rec=CAP/f'step-{step}.receipt.json';req(pack.is_file() and rec.is_file(),f'missing P011 baseline accepted capture {step}')
 d=json.loads(rec.read_text());req(int(d.get('accepted_step',-1))==step and d.get('physical_endpoint')=='accepted','pose not accepted')
 ph=sha(pack);req(d.get('pack_file_sha256')==ph,'pack hash mismatch')
 poses={int(x['body_index']):(np.asarray(x['position_m'],dtype=np.float32),np.asarray(x['quaternion_xyzw'],dtype=np.float32)) for x in d['accepted_registered_body_poses']};req(len(poses)==86,'registered pose count mismatch')
 return {'step':step,'pack':str(pack),'pack_sha256':ph,'receipt':str(rec),'receipt_sha256':sha(rec),'body_state_sha256':d.get('accepted_body_state_sha256'),'poses':poses}
def f32sum_points(points,weights,core_ids,default,capture):
 out=np.zeros((len(points),3),dtype=np.float32)
 for j,core in enumerate(core_ids):
  dp,dq=default[int(core)];inv=np.asarray([-dq[0],-dq[1],-dq[2],dq[3]],dtype=np.float32);bt=np.asarray([-x for x in rotate_cxx(inv,dp)],dtype=np.float32)
  local=np.asarray([[f32(bt[k]+x[k]) for k in range(3)] for x in rotate_cxx(inv,points)],dtype=np.float32)
  cp,cq=capture[int(core)];trans=np.asarray([[f32(cp[k]+x[k]) for k,x in enumerate(v)] for v in rotate_metal(cq,local)],dtype=np.float32)
  w32=weights[:,j].astype(np.float32)
  for k in range(3):out[:,k]=np.asarray([f32(float(w32[i])*float(trans[i,k])+float(out[i,k])) for i in range(len(points))],dtype=np.float32)
 return out
def field_for_points(points,control_xyz,control_core,core_ids):
 points=np.asarray(points,dtype=np.float64);controls=np.asarray(control_xyz,dtype=np.float64);out=np.zeros((len(points),len(core_ids)),dtype=np.float64);ci={int(k):i for i,k in enumerate(core_ids)};cc=np.asarray([ci[int(k)] for k in control_core],dtype=np.int64)
 for i,p in enumerate(points):
  d2=np.sum((controls-p[None,:])**2,axis=1);zero=np.flatnonzero(d2==0.)
  if len(zero):nw=np.zeros(len(controls),dtype=np.float64);nw[zero]=1./len(zero)
  else:
   md=float(np.min(d2));raw=md/d2;den=math.fsum(float(x) for x in raw);req(math.isfinite(den) and den>0,'invalid Shepard denominator');nw=raw/den
  cw=np.zeros(len(core_ids),dtype=np.float64)
  for j,w in enumerate(nw):cw[cc[j]]+=float(w)
  total=math.fsum(float(x) for x in cw);cw/=total
  if len(cw)>1:cw[-1]=1.-sum(float(x) for x in cw[:-1])
  req(np.isfinite(cw).all() and np.all(cw>0.) and np.all(cw.astype(np.float32)>0.) and abs(sum(float(x) for x in cw)-1.)<=1e-12,'invalid / nonpositive route field')
  out[i]=cw
 return out
def d2(a,b):return sum(((a[k]-b[k])**2 for k in range(3)),F(0))
def max_edge(face,coords):
 es=[tuple(sorted((face[i],face[(i+1)%3]))) for i in range(3)]
 # max exact squared length; deterministic lexicographic tie break.
 return max(((d2(coords[e[0]],coords[e[1]]),e) for e in es),key=lambda z:(z[0],tuple(-x for x in z[1])))
def fraction_digest(x):
 h=hashlib.sha256()
 for n in (x.numerator,x.denominator):
  sign=b'-' if n<0 else b'+'
  mag=abs(n);data=mag.to_bytes(max(1,(mag.bit_length()+7)//8),'big')
  h.update(sign);h.update(len(data).to_bytes(8,'big'));h.update(data)
 return h.hexdigest()
def topo(face_list,coords):
 edge=collections.defaultdict(list);adj=collections.defaultdict(set);used=set();vol=F(0)
 for fi,t in enumerate(face_list):
  a,b,c=t;used.update(t);vol+=(coords[a][0]*(coords[b][1]*coords[c][2]-coords[b][2]*coords[c][1])+coords[a][1]*(coords[b][2]*coords[c][0]-coords[b][0]*coords[c][2])+coords[a][2]*(coords[b][0]*coords[c][1]-coords[b][1]*coords[c][0]))/6
  for x,y in ((a,b),(b,c),(c,a)):edge[tuple(sorted((x,y)))].append((fi,1 if x<y else -1))
 for e,inc in edge.items():
  for f,_ in inc:adj[f].update(g for g,_ in inc if g!=f)
 bad=[(list(e),[(f,s) for f,s in q]) for e,q in edge.items() if len(q)!=2 or sum(s for _,s in q)!=0]
 unseen=set(range(len(face_list)));components=[];comp_face_lists=[]
 while unseen:
  seed=unseen.pop();stack=[seed];comp=[]
  while stack:
   f=stack.pop();comp.append(f)
   for g in adj[f]:
    if g in unseen:unseen.remove(g);stack.append(g)
  vv={v for fi in comp for v in face_list[fi]};cv=F(0)
  for fi in comp:
   a,b,c=(coords[v] for v in face_list[fi]);cv+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6
  components.append({'faces':len(comp),'vertices':len(vv),'signed_volume_exact_sha256':fraction_digest(cv),'signed_volume_m3':float(cv),'euler_characteristic':len(vv)-sum(1 for e in edge if e[0] in vv and e[1] in vv)+len(comp)})
  comp_face_lists.append(comp)
 return {'vertices_used':len(used),'edges':len(edge),'faces':len(face_list),'component_count':len(components),'components':components,'edge_incidence_or_orientation_errors':len(bad),'bad_edge_examples':bad[:16],'oriented_closed':not bad,'signed_volume_exact_sha256':fraction_digest(vol),'signed_volume_m3':float(vol)},comp_face_lists
def union_faces(rows):
 d=collections.defaultdict(list)
 for sid,faces in rows.items():
  for fi,t in enumerate(faces):d[tuple(sorted(t))].append((sid,t,fi))
 out=[];meta=[];bad=[];cancelled=0
 for key,items in d.items():
  if len(items)==1:out.append(items[0][1]);meta.append({'stable_id':items[0][0],'row_face':items[0][2]})
  elif len(items)==2 and items[0][0]!=items[1][0]:
   a,b=items[0][1],items[1][1]
   if any(a==tuple(reversed(b[k:]+b[:k])) for k in range(3)):cancelled+=1
   else:bad.append({'key':key,'faces':[(z[0],z[2]) for z in items],'reason':'shared winding not opposite'})
  else:bad.append({'key':key,'faces':[(z[0],z[2]) for z in items],'reason':'invalid multiplicity'})
 return out,meta,{'input_faces':sum(len(v) for v in rows.values()),'cancelled_shared_faces':cancelled,'union_faces':len(out),'bad_shared_face_keys':bad[:32],'bad_shared_face_key_count':len(bad)}
def intersection_audit(points,faces,ci,mode):
 if mode=='exact':P=[tuple(F(x) for x in p) for p in points]
 else:P=[tuple(F.from_float(float(np.float32(x))) for x in p) for p in points]
 deg=[]
 for i,t in enumerate(faces):
  a,b,c=(P[int(v)] for v in t)
  if not any(ci._cross(ci._sub(b,a),ci._sub(c,a))):deg.append(i)
 if deg:return {'mode':mode,'degenerate_count':len(deg),'degenerate_faces':deg[:64],'self_count':None,'pairs':[],'predicate_complete':False}
 rec=ci._records(P,np.asarray(faces,dtype=np.int64).tolist());q=ci._audit_pair(rec,rec,same_surface=True)
 return {'mode':mode,'degenerate_count':0,'degenerate_faces':[],'self_count':int(q['count']),'pairs':[list(map(int,z)) for z in q.get('triangle_pairs',[])],'aabb_candidate_pairs':int(q.get('aabb_candidate_pairs',0)),'predicate_complete':True}
def crossing_metrics(points,faces,pairs):
 p=np.asarray(points,dtype=np.float64);out=[]
 for ia,ib in pairs:
  ta=p[np.asarray(faces[int(ia)],dtype=np.int64)];tb=p[np.asarray(faces[int(ib)],dtype=np.int64)];na=np.cross(ta[1]-ta[0],ta[2]-ta[0]);nb=np.cross(tb[1]-tb[0],tb[2]-tb[0]);la=float(np.linalg.norm(na));lb=float(np.linalg.norm(nb))
  if la==0 or lb==0:continue
  da=(tb-ta[0])@na/la;db=(ta-tb[0])@nb/lb
  xa=max(0.,min(float(da.max()),-float(da.min())));xb=max(0.,min(float(db.max()),-float(db.min())))
  out.append({'faces':[int(ia),int(ib)],'two_sided_plane_crossing_um':max(xa,xb)*1e6,'crossing_each_direction_um':[xa*1e6,xb*1e6],'normal_cosine':float(np.dot(na,nb)/(la*lb))})
 return {'pairs':out,'max_two_sided_plane_crossing_um':max((x['two_sided_plane_crossing_um'] for x in out),default=0.),'metric_definition':'Triangle-to-opposite-plane crossing measure, not solid penetration depth.'}
def split_face(face,edge,mid):
 for i in range(3):
  a,b,c=face[i],face[(i+1)%3],face[(i+2)%3]
  if tuple(sorted((a,b)))==edge:return (a,mid,c),(mid,b,c)
 raise RuntimeError('selected edge is not a face edge')
def build_field_controls(route_path):
 doc=json.loads(route_path.read_text());req(doc.get('schema')=='numi.human.fds-fdp-source-route-provenance-061.v1','route provenance schema mismatch')
 muscles=[m for m in doc['matching_muscles'] if str(m.get('actuator_name','')).endswith('_l')];expected={'FDS2_l','FDS3_l','FDS4_l','FDS5_l','FDP2_l','FDP3_l','FDP4_l','FDP5_l'};req({m['actuator_name'] for m in muscles}==expected,'left FDS/FDP route list mismatch')
 dedup={};occ=0
 for m in muscles:
  for n in m['route_nodes']:
   occ+=1;k=(int(n['source_id']),str(n['kind']),int(n['core_body_index']));ident={z:n[z] for z in ('source_id','name','kind','source_body_id','source_body_name','core_body_index','position_body_m','position_world_m')}
   if k in dedup:req(dedup[k]==ident,'duplicate route-control identity mismatch')
   else:dedup[k]=ident
 controls=[dedup[k] for k in sorted(dedup)];cores=sorted({int(x['core_body_index']) for x in controls});xyz=np.asarray([x['position_world_m'] for x in controls],dtype=np.float64);cc=np.asarray([int(x['core_body_index']) for x in controls],dtype=np.int64)
 req(len(controls)==74 and occ==81 and len(cores)==21,'route control inventory changed')
 return xyz,cc,cores,{'route_occurrences':occ,'unique_controls':len(controls),'core_body_indices':cores,'dedup_key':'(source_id,kind,core_body_index)','field_formula':'Normalized inverse-square Shepard mass over deduplicated route controls, summed by core; exact coincident anchors equally share mass.'}
def refine_one(threshold_mm, base, coord0, rows0, metadata0, field_old, control_xyz,control_core,core_ids):
 threshold=F(int(threshold_mm),1000);h2=threshold*threshold;coords=dict(coord0);coord_to_gid={v:g for g,v in coords.items()};next_gid=max(coords)+1
 faces={};active=set();edgefaces=collections.defaultdict(set);heap=[];next_fid=0;seam_keys=set()
 for a in rows0[132]:
  ka=tuple(sorted(a['tri']))
  if ka in rows0['_keys134']:seam_keys.add(ka)
 seam_vertices={v for k in seam_keys for v in k}
 for sid in (132,134):
  faces[sid]=[]
  for rec in rows0[sid]:
   tri=rec['tri'];is_target=tuple(sorted(tri)) in seam_keys or bool(set(tri)&seam_vertices)
   nr=dict(fid=next_fid,sid=sid,tri=tri,target=is_target,ancestry=rec['ancestry'],ref=rec['ref'])
   faces[sid].append(nr);active.add(next_fid);face_by_id[next_fid]=nr
   for e in (tuple(sorted((tri[0],tri[1]))),tuple(sorted((tri[1],tri[2]))),tuple(sorted((tri[2],tri[0])))):edgefaces[e].add(next_fid)
   if is_target:
    q,e=max_edge(tri,coords)
    if q>h2:heapq.heappush(heap,(-float(q),e,next_fid))
   next_fid+=1
 initial_counts={str(s):sum(1 for r in faces[s] if r['target']) for s in (132,134)}
 split_edges=[];split_vertices={};split_parent_ids={};split_incidence=collections.Counter();processed=0
 # Each longest-edge split is global: every face in either stable row incident to that exact edge is bisected together.
 while heap:
  _,edge,fid=heapq.heappop(heap)
  if fid not in active:continue
  parent=face_by_id.get(fid)
  if parent is None or not parent['target']:continue
  q,le=max_edge(parent['tri'],coords)
  if q<=h2:continue
  edge=le
  req(edge in edgefaces and fid in edgefaces[edge],'target longest edge missing from adjacency')
  a,b=edge;midp=tuple((coords[a][i]+coords[b][i])/2 for i in range(3))
  mid=coord_to_gid.get(midp)
  if mid is None:
   mid=next_gid;next_gid+=1;coords[mid]=midp;coord_to_gid[midp]=mid
   split_vertices[mid]=midp;split_parent_ids[mid]=(a,b)
  incident=sorted(edgefaces[edge]);req(fid in incident and len(incident)>=2,'subdivision edge lacks incident faces')
  split_edges.append((a,b,mid));split_incidence[len(incident)]+=1;processed+=1
  for oldid in incident:
   old=face_by_id.pop(oldid);active.remove(oldid)
   for e in (tuple(sorted((old['tri'][0],old['tri'][1]))),tuple(sorted((old['tri'][1],old['tri'][2]))),tuple(sorted((old['tri'][2],old['tri'][0])))):
    edgefaces[e].remove(oldid)
    if not edgefaces[e]:del edgefaces[e]
   children=split_face(old['tri'],edge,mid)
   for tri in children:
    child=dict(fid=next_fid,sid=old['sid'],tri=tri,target=old['target'],ancestry=old['ancestry'],ref=old['ref'])
    next_fid+=1;face_by_id[child['fid']]=child;active.add(child['fid']);faces[child['sid']].append(child)
    for e in (tuple(sorted((tri[0],tri[1]))),tuple(sorted((tri[1],tri[2]))),tuple(sorted((tri[2],tri[0])))):edgefaces[e].add(child['fid'])
    if child['target']:
     qq,ee=max_edge(tri,coords)
     if qq>h2:heapq.heappush(heap,(-float(qq),ee,child['fid']))
  if processed%1000==0:print(json.dumps({'event':'subdivision_progress','threshold_mm':threshold_mm,'split_edges':processed,'vertices':len(coords),'active_faces':len(active)}),flush=True)
 # Canonical geometry arrays only. New coordinates are exact edge midpoints; route weights are deferred.
 outrows={};allweights={}
 for sid in (132,134):
  active_rows=sorted((r for r in faces[sid] if r['fid'] in active),key=lambda r:r['fid']);used=sorted({v for r in active_rows for v in r['tri']});ix={g:i for i,g in enumerate(used)}
  arr=np.asarray([[np.float32(float(coords[g][k])) for k in range(3)] for g in used],dtype=np.float32)
  outrows[sid]={'global_vertex_ids':used,'vertices_exact_m':[coords[g] for g in used],'vertices_float32_m':arr,'core_weights_float64':None,'faces_global':[r['tri'] for r in active_rows],'faces':np.asarray([[ix[g] for g in r['tri']] for r in active_rows],dtype=np.int64),'parent_060_face_indices':[r['ancestry'] for r in active_rows],'parent_054_face_refs':[r['ref'] for r in active_rows],'target_refinement_band':[bool(r['target']) for r in active_rows]}
 shared_out=set(tuple(sorted(t)) for t in outrows[132]['faces_global']) & set(tuple(sorted(t)) for t in outrows[134]['faces_global'])
 req(len(shared_out)>0,'refinement lost every shared interface triangle')
 f_by={s:collections.defaultdict(list) for s in (132,134)}
 for s in (132,134):
  for i,t in enumerate(outrows[s]['faces_global']):f_by[s][tuple(sorted(t))].append((i,t))
 bad=[]
 for key in set(f_by[132])&set(f_by[134]):
  aa=f_by[132][key];bb=f_by[134][key]
  if len(aa)!=1 or len(bb)!=1 or not any(aa[0][1]==tuple(reversed(bb[0][1][k:]+bb[0][1][:k])) for k in range(3)):bad.append(key)
 req(not bad,'shared output face mismatch: '+str(bad[:2]))
 shared_gids=set(v for key in set(f_by[132])&set(f_by[134]) for v in key)
 target_max={}
 for sid in (132,134):
  lens=[math.sqrt(float(max_edge(r['tri'],coords)[0]))*1000 for r in faces[sid] if r['fid'] in active and r['target']]
  target_max[str(sid)]={'target_faces':len(lens),'max_edge_mm':max(lens,default=0.),'count_over_threshold':sum(x>threshold_mm+1e-9 for x in lens)}
 req(all(v['count_over_threshold']==0 for v in target_max.values()),'refinement band edge threshold not satisfied')
 meta={'threshold_mm':threshold_mm,'input_vertices':len(coord0),'output_global_vertices':len(coords),'new_vertices':len(split_vertices),'input_faces':{str(s):len(rows0[s]) for s in (132,134)},'output_faces':{str(s):sum(1 for r in faces[s] if r['fid'] in active) for s in (132,134)},'initial_target_face_counts':initial_counts,'final_target_edge_metrics':target_max,'split_edge_count':processed,'split_edge_incident_face_count_distribution':{str(k):v for k,v in sorted(split_incidence.items())},'shared_input_faces':len(seam_keys),'shared_output_faces':len(shared_out),'shared_output_vertices':len(shared_gids),'new_vertex_parent_edge_count':len(split_parent_ids),'split_edge_records':split_edges,'new_vertex_parent_global_ids':{str(k):v for k,v in split_parent_ids.items()},'new_vertices_exact_m':{str(k):v for k,v in split_vertices.items()},'geometry_change':'None: every new coordinate is the exact rational midpoint of an existing source edge; no existing coordinate moved. Child faces are exact bisections of parent faces; each child retains 060 and 054 ancestry. Route weights omitted pending registration-frame verification.'}
 return outrows,coords,allweights,meta

def run(threshold_mm,out_name):
 global OUT,face_by_id,weight_cache
 OUT=ROOT/out_name;OUT.mkdir(parents=False,exist_ok=False)
 (OUT/'result-001').mkdir();res=OUT/'result-001'
 for p in (MAP,C,CK,CI):req(Path(p).exists(),"missing input "+str(p))
 req(json.loads(CK.read_text()).get('complete') and json.loads(CK.read_text())['checks']['union']['self_count']==0,'candidate060 exact check no longer passes')
 mp=pickle.loads(MAP.read_bytes());req(mp['candidate_sha256']==sha(C),'source map candidate SHA mismatch')
 coords={};rows0={132:[],134:[]}
 for sid in (132,134):
  s=mp['surfaces'][str(sid)];gids=list(map(int,s['global_vertex_ids']));exact=[tuple(F(q) if isinstance(q,F) else F(int(q[0]),int(q[1])) for q in p) for p in s['rest_vertices_exact_m']]
  for g,p in zip(gids,exact):
   req(g not in coords or coords[g]==p,'global source point mismatch gid '+str(g));coords[g]=p
  for i,f in enumerate(s["faces"]):
   tri=tuple(gids[int(j)] for j in f);rows0[sid].append({'tri':tri,'ancestry':int(s['original_054_face_indices'][i]),'ref':s['original_054_face_parent_refs'][i]})
 keys132=collections.Counter(tuple(sorted(r['tri'])) for r in rows0[132]);keys134=collections.Counter(tuple(sorted(r['tri'])) for r in rows0[134]);req(all(v==1 for v in keys132.values()) and all(v==1 for v in keys134.values()),'duplicate triangle within input row')
 common=set(keys132)&set(keys134);req(len(common)==2146,'common interface triangle count changed');rows0['_keys134']=keys134
 # Run target-specific global edge bisection.
 face_by_id={};weight_cache={}
 outrows,newcoords,newweights,refine_meta=refine_one(threshold_mm,mp,coords,rows0,{},None,None,None,None)
 # Build exact topology/source-volume checks; common faces cancel to a closed material union.
 rowglobal={s:outrows[s]['faces_global'] for s in (132,134)};union,union_meta,union_stats=union_faces(rowglobal)
 base_global={s:[r['tri'] for r in rows0[s]] for s in (132,134)};base_union,base_union_meta,base_union_stats=union_faces(base_global)
 base_topology={str(s):topo(base_global[s],coords)[0] for s in (132,134)};base_topology['union']=topo(base_union,coords)[0]
 topology={}
 for s in (132,134):topology[str(s)]=topo(rowglobal[s],newcoords)[0]
 topology['union']=topo(union,newcoords)[0]
 req(all(topology[s]['oriented_closed'] for s in ('132','134','union')),'subdivision changed closed/oriented topology')
 for name in ('132','134','union'):
  b=base_topology[name];a=topology[name]
  req(a['component_count']==b['component_count'] and a['edge_incidence_or_orientation_errors']==0,'topology component/edge change '+name)
  req(sorted(x['signed_volume_exact_sha256'] for x in a['components'])==sorted(x['signed_volume_exact_sha256'] for x in b['components']),'exact signed component volume changed '+name)
 # Every component preserves exact signed volume because this is exact face bisection.
 sys.path.insert(0,str(CI.parent.parent));from numilab_human import cardiac_cavity_intersections as ci
 # Source-rest self-checks on exact geometry; parent candidate060's exact certificate is also pinned.
 rest_global_ids=sorted(newcoords);rest_arr=np.asarray([[float(newcoords[g][k]) for k in range(3)] for g in rest_global_ids],dtype=np.float64);globalix={g:i for i,g in enumerate(rest_global_ids)}
 rest_faces={s:np.asarray([[globalix[g] for g in t] for t in rowglobal[s]],dtype=np.int64) for s in (132,134)};rest_union=np.asarray([[globalix[g] for g in t] for t in union],dtype=np.int64)
 rest_audits={}
 for s in (132,134):
  rest_audits[str(s)]=intersection_audit([newcoords[g] for g in rest_global_ids],rest_faces[s],ci,'exact')
  print(json.dumps({'event':'source_exact_audit','threshold_mm':threshold_mm,'stable_id':s,'self_count':rest_audits[str(s)]['self_count'],'degenerate':rest_audits[str(s)]['degenerate_count']}),flush=True)
 rest_audits['union']=intersection_audit([newcoords[g] for g in rest_global_ids],rest_union,ci,'exact')
 print(json.dumps({'event':'source_exact_audit','threshold_mm':threshold_mm,'stable_id':'union','self_count':rest_audits['union']['self_count'],'degenerate':rest_audits['union']['degenerate_count']}),flush=True)
 candidate={'schema':'numi.human.left-fds132-fdp134-conforming-source-interface-geometry-083.v1','input_candidate_path':str(C),'input_candidate_sha256':sha(C),'input_source_parent_map_path':str(MAP),'input_source_parent_map_sha256':sha(MAP),'threshold_mm':threshold_mm,'shared_interface_definition':'Exact global triangle key occurs once in each stable row with opposite winding. Refinement target is every such interface triangle and every face in either row incident to any vertex of such a triangle.','surfaces':outrows,'global_vertices_exact_m':newcoords,'refinement':refine_meta,'topology':topology,'input_topology':base_topology,'source_rest_exact_self_audits':rest_audits,'source_rest_union_cancellation':union_stats,'source_self_inheritance':'Candidate060 exact self/union certificate is pinned. Every child triangle is an exact subset of one input face after midpoint bisection; exact component volumes and source-rest self predicates were rechecked. Route weights/poses are intentionally absent pending registration-frame verification.'}
 cp=res/'candidate.pkl';cp.write_bytes(pickle.dumps(candidate,protocol=5))
 save(res/'source-checks.json',{'candidate_sha256':sha(cp),'topology':topology,'input_topology':base_topology,'source_rest_exact_self_audits':rest_audits,'source_rest_union_cancellation':union_stats,'refinement':{k:v for k,v in refine_meta.items() if k not in ('split_edge_records','new_vertex_parent_global_ids','new_vertices_exact_m')},'source_c060_exact_check_path':str(CK),'source_c060_exact_check_sha256':sha(CK)})
 input_paths=[MAP,C,CK,CI];inputs={str(p):sha(p) for p in input_paths}
 report={'schema':'numi.human.left-fds132-fdp134-conforming-interface-subdivision-report-083.v1','status':'completed_bounded_source_geometry_refinement','threshold_mm':threshold_mm,'scope':'CPU-only exact source mesh refinement of candidate060 interface and one-face incidence band. Source geometry and lineage only; route field and pose work withheld until the source-to-registered-default transform is established. No source mutation, no GPU/native run.','prediction':'Reducing interface/incident source triangle edge length to the declared ceiling must leave exact geometry, closed topology, component count, orientation, and signed volumes unchanged; those are falsified by any geometric/topological change. Pose-level benefit remains untested until registered-frame weights are available.','source_and_candidate':{'path':str(C),'sha256':sha(C),'exact_check_path':str(CK),'exact_check_sha256':sha(CK),'source_parent_map_path':str(MAP),'source_parent_map_sha256':sha(MAP)},'refinement':{k:v for k,v in refine_meta.items() if k not in ('split_edge_records','new_vertex_parent_global_ids','new_vertices_exact_m')},'topology':topology,'input_topology':base_topology,'source_rest_exact_self_audits':rest_audits,'source_rest_union_cancellation':union_stats,'candidate_path':str(cp),'candidate_sha256':sha(cp),'qualification':{'cpu_only':True,'gpu_used':False,'route_field_weights_bound':False,'pose_audit_run':False,'native_admission':False,'reason':'Registration frame for force-route coordinates is unresolved; no spatial weights or pose results were emitted.'},'environment':{'python':sys.version,'numpy':np.__version__,'cpu_workers':1,'rss_watchdog_kib':LIMIT},'input_hashes':inputs,'input_hashes_stable_at_completion':all(sha(p)==v for p,v in inputs.items()),'elapsed_seconds':time.monotonic()-START}
 save(res/'report.json',report)
 print(json.dumps({'event':'geometry_report_complete','report':str(res/'report.json'),'report_sha256':sha(res/'report.json'),'candidate':str(cp),'candidate_sha256':sha(cp),'threshold_mm':threshold_mm,'stable_inputs':report['input_hashes_stable_at_completion'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--threshold-mm',type=int,choices=(1,2),required=True);ap.add_argument('--out-name',required=True);args=ap.parse_args();run(args.threshold_mm,args.out_name)
if __name__=='__main__':main()
