#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import hashlib,importlib.util,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');ROOT=P.parent
BASE035=P/'quality-star-dp-batch-035';BASE036=P/'quality-star-connected-union-036-run002';WARN=P/'volume-007-batch035-source004/warning-current-role-map-001/report.json'
R035_SHA='7ba174b74a280126e56357a9d6123379e9668b9fff7d54303cfbd017b3d1cb74';V035_SHA='32dc1ef7b093615bc1f65d92de959f7d693882118a9ea1addd1a34b6a455b047';R036_SHA='137f1f1ff571134d7ea290eca1dbf4cfc4907e4d57663205fa15696c1fec9873';WARN_SHA='af9e47f731bbb9e60958772e2241719d0f81c8c48791bea759fedc5eeec8bcc9'
SRC={
'O':(P/'microedge-csr-canonical-011/O/candidate-O-csr-canonical.npz','578cee97a3bbecb73c0e1a6d7c26f45ac6cc111a552996f3910866a4a3652a9b',300000),
'I':(P/'local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz','9721102247157eeb42c59633fda59d63fd5ba48ad11ba4e63987834136711e31',600000),
'D':(P/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz','15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f',0)}
CUR={
'O':(BASE036/'O-connected-union-proposal.npz','52be7aa80711cebbcc67fa078f9f71100835671919758b555d89cca5b02c8ded'),
'I':(BASE036/'I-connected-union-proposal.npz','b85ac6a926ed454ab920618e56a467082a0cf14ff45e3019456e9380b246fe38'),
'D':(BASE036/'D-connected-union-proposal.npz','7985a3c3b61b2b8480fbaee05ee3018c355373907fdbab1b661220c78caecbfb')}
TARGET=P/'microedge-collar-014-saved-op-replay/receipt.json';TARGET_SHA='a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01'
OWNER_LOADER=ROOT/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py';OWNER_LOADER_SHA='f1d697b42fc9d97472fbd70cec2d8adfe178bb591d3513680609396ec4c204ad';OWNER_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
BONE=ROOT/'bone-barrier-remeshing-609/bone_barrier.py';BONE_SHA='734527f060dcae186ae1b9609eca1ab95768d952aa0e9e8c7af76cae882b236b';BONE_NATIVE='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5';BONE_MANIFEST='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c'
DISK=P/'quality-disk-eligibility-031/constraint_disk_quality.py';DISK_SHA='38e7a1ab5caa9044b86de931555590061339ed11388a84bf302ee78eeb53e97f'
CHORD=ROOT/'exact-chord-pruning-618/chord_pruning.py';CHORD_SHA='c59f57487dc180db31f8820db23b7e355d69585de3b363eb39c95edaadedbebc'
FILTER=P/'quality-exact-filter-adapter-030/filtered_prepared_exact_owner.py';FILTER_SHA='c3f427432096719b9fe3b91cbd2fc1a6f779228c03456cef85dc8bbbae9da16d'
ALT=20e-9;ANGLE=1e-4;MAX_LOOP=22;MAX_FACES=240;BUDGET=165.;PER_PATCH=8.
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False)as z:return{k:np.array(z[k],copy=True)for k in z.files}
def stableq(p,f):
 x=np.asarray(p,np.float32)[np.asarray(f,np.int64)].astype(np.float64);a=x[:,1]-x[:,0];b=x[:,2]-x[:,0];c=b-a;ar=np.linalg.norm(np.cross(a,b),axis=1);long=np.maximum.reduce((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)));alt=np.divide(ar,long,out=np.zeros_like(ar),where=long>0);aa=np.einsum('ij,ij->i',a,a);bb=np.einsum('ij,ij->i',b,b);ab=np.einsum('ij,ij->i',a,b);dot=np.stack((ab,aa-ab,bb-ab),axis=1);ang=np.min(np.degrees(np.arctan2(ar[:,None],dot)),axis=1);return alt,ang
def oneq(p,t):
 a,b=stableq(p,np.asarray(t,np.int64).reshape((1,3)));return float(a[0]),float(b[0])
def edgeindex(f,n):
 f=np.asarray(f,np.int64);e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));e=np.sort(e,axis=1);k=e[:,0]*np.int64(n)+e[:,1];rows=np.tile(np.arange(len(f),dtype=np.int64),3);o=np.argsort(k,kind='mergesort');return k[o],rows[o],n
def boundary(f):
 e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]));s=np.sort(e,axis=1);u,inv,c=np.unique(s,axis=0,return_inverse=True,return_counts=True);return u[c==1]
def patchloop(f,rows):
 e=defaultdict(list)
 for r in map(int,rows):
  a,b,c=map(int,f[r])
  for u,v in((a,b),(b,c),(c,a)):e[tuple(sorted((u,v)))].append((u,v))
 if any(len(x)>2 for x in e.values()):return None
 bd=[x[0]for x in e.values()if len(x)==1];nxt={}
 for u,v in bd:
  if u in nxt:return None
  nxt[u]=v
 if not nxt:return None
 s=min(nxt);loop=[];u=s
 while u not in loop:
  if u not in nxt:return None
  loop.append(u);u=nxt[u]
 if u!=s or len(loop)!=len(bd):return None
 verts=set(map(int,np.asarray(f)[np.asarray(rows,np.int64)].reshape(-1)))
 if len(verts)-len(e)+len(rows)!=1:return None
 return loop
def external(loop,ei,current,retired):
 keys,fr,n=ei;curr=set(map(int,current));retired=retired;out=set()
 for i,u in enumerate(map(int,loop)):
  for v in map(int,loop[i+1:]):
   a,b=sorted((u,v));k=np.int64(a)*np.int64(n)+b;l=int(np.searchsorted(keys,k,'left'))
   if l>=len(keys)or keys[l]!=k:continue
   r=int(np.searchsorted(keys,k,'right'))
   if any(int(x)not in curr and int(x)not in retired for x in fr[l:r]):out.add((a,b))
 return out
def records(pred,p,f,fb,vb):
 r,_=pred.float32_surface_records(np.asarray(p,np.float64),np.asarray(f,np.int64),source_seam_groups=())
 if len(r)!=len(f):raise ValueError('surface record count')
 return[(x[0],x[1],x[2],fb+i,tuple(vb+int(v)for v in x[4]))for i,x in enumerate(r)]
class Proxy:
 def __init__(self,filtered,vb,n):self.filtered=filtered;self.vertex_offset=int(vb);self.physical=np.arange(int(n),dtype=np.int64)+self.vertex_offset
 def __getattr__(self,k):return getattr(self.filtered,k)
 def _records(self,lattice,faces):
  raw=self.filtered._owner._records(lattice,faces);return[(r[0],r[1],r[2],r[3],tuple(self.vertex_offset+int(v)for v in r[4]))for r in raw]
 def _query_surface_aabb(self,q,index):
  ex=self.filtered.excluded_face_ids
  for x,y in self.filtered._owner._query_surface_aabb(q,index):
   if int(y[3])not in ex:yield x,y
def disk_ok(p,loop):
 n=len(loop);q={}
 for i in range(n-2):
  for k in range(i+1,n-1):
   for j in range(k+1,n):
    a,b=oneq(p,(loop[i],loop[k],loop[j]));q[i,k,j]=(a>=ALT and b>=ANGLE)
 d=[[False]*n for _ in range(n)]
 for i in range(n-1):d[i][i+1]=True
 for span in range(2,n):
  for i in range(n-span):
   j=i+span;d[i][j]=any(q[i,k,j]and d[i][k]and d[k][j]for k in range(i+1,j))
 return d[0][n-1]
def source_lineage(role,A,row):
 if role=='O':
  o=A['face_ancestry_offsets'];ks=('face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows');return{'face_source_layer':[int(A['face_source_layer'][row])],'face_source_body':[int(A['face_source_body'][row])],'face_source_bone':[int(A['face_source_bone'][row])],'face_source_original_id':[int(A['face_source_original_id'][row])],'complete_face_ancestry_union':[list(map(int,(A[k][j]for k in ks)))for j in range(int(o[row]),int(o[row+1]))]}
 if role=='I':
  d={'face_source_layer':[int(A['face_source_layer'][row])],'face_source_body':[int(A['face_source_body'][row])],'face_source_bone':[int(A['face_source_bone'][row])],'face_source_original_id':[int(A['face_source_original_id'][row])]}
  for ok,vk in(('face_parent_offsets','face_parent_rows'),('edge_flip_parent_offsets','edge_flip_parent_face_rows'),('microedge_parent_input_face_offsets','microedge_parent_input_face_rows')):d[ok+'_union']=list(map(int,A[vk][int(A[ok][row]):int(A[ok][row+1])]))
  return d
 d={'face_source_layer':[int(A['face_source_layer'][row])],'face_source_body':[int(A['face_source_body'][row])],'face_source_bone':[int(A['face_source_bone'][row])],'face_source_original_id':[int(A['face_source_original_id'][row])]}
 for ok,vk,nm in(('candidate_face_ancestry_offsets','candidate_face_ancestry_rows','candidate_face_ancestry_union'),('operational_output_face_parent_offsets','operational_output_face_parent_rows','operational_output_face_parent_offsets_union'),('microedge_parent_input_face_offsets','microedge_parent_input_face_rows','microedge_parent_input_face_offsets_union')):d[nm]=list(map(int,A[vk][int(A[ok][row]):int(A[ok][row+1])]))
 a,b=map(int,(A['union_face_ancestry_offsets'][row],A['union_face_ancestry_offsets'][row+1]));d['union_face_ancestry_tuples']=[[int(A['union_face_ancestry_run_original_ids'][j]),int(A['union_face_ancestry_face_ids'][j]),int(A['union_face_ancestry_rows'][j])]for j in range(a,b)];return d
def merge_lineage(items):
 out=defaultdict(set)
 def cv(x):return tuple(cv(y)for y in x)if isinstance(x,(list,tuple))else int(x)
 for d in items:
  for k,v in d.items():
   if k=='patch_face_rows':continue
   for x in v:out[k].add(cv(x))
 return{k:[list(x)if isinstance(x,tuple)else x for x in sorted(v)]for k,v in sorted(out.items())}
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+BUDGET
 launch={'schema':'numi.warning-local-connected-union-launch.v1','status':'STARTED','script_path':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__)),'output_directory':str(out),'physical_admission':False,'runtime':0,'accepted_steps':0,'GPU':False,'started_unix_time':time.time()}
 (out/'launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
 if sha(BASE035/'report.json')!=R035_SHA or sha(BASE035/'independent-verification-001/report.json')!=V035_SHA or sha(BASE036/'report.json')!=R036_SHA or sha(WARN)!=WARN_SHA or sha(TARGET)!=TARGET_SHA:raise ValueError('warning or parent candidate proof pin')
 for p,d in((OWNER_LOADER,OWNER_LOADER_SHA),(BONE,BONE_SHA),(DISK,DISK_SHA),(CHORD,CHORD_SHA),(FILTER,FILTER_SHA)):
  if sha(p)!=d:raise ValueError('helper SHA mismatch '+str(p))
 lm=module('warning_union_owner_loader_037',OWNER_LOADER);pred=lm.load_exact_owner()
 if sha(Path(pred.__file__))!=OWNER_SHA:raise ValueError('exact owner SHA mismatch')
 dm=module('warning_union_disk_037',DISK);cm=module('warning_union_chord_037',CHORD);fm=module('warning_union_filter_037',FILTER);bm=module('warning_union_bone_037',BONE)
 rr035=json.loads((BASE035/'report.json').read_text());rr036=json.loads((BASE036/'report.json').read_text());wd=json.loads(WARN.read_text());targ=json.loads(TARGET.read_text())['protected149']['surface_routes']
 src={};carr={};parent035={};points={};faces={};facebase={};pointbase={};map36={};map35={};pmap36={};pmap35={};current_point_to_source={};patchid36={};ops36={};patchid35={};ops35={}
 for role in ('O','I','D'):
  sp,ss,fb=SRC[role];cp,cs=CUR[role]
  if sha(sp)!=ss or sha(cp)!=cs:raise ValueError(role+' candidate hash')
  src[role]=load(sp);carr[role]=load(cp);parent_path=Path(rr035['role_outputs'][role]['path'])
  if sha(parent_path)!=rr035['role_outputs'][role]['sha256']:raise ValueError(role+' 035 proposal hash')
  parent035[role]=load(parent_path)
  points[role]=np.asarray(carr[role]['points_world_m'],np.float32);faces[role]=np.asarray(carr[role]['faces'],np.int64);facebase[role]=fb
  map36[role]=np.asarray(carr[role]['face_output_to_input_rows'],np.int64)
  map35[role]=np.asarray(parent035[role]['face_output_to_input_rows'],np.int64)
  pmap36[role]=np.asarray(carr[role]['point_output_to_input_rows'],np.int64)
  pmap35[role]=np.asarray(parent035[role]['point_output_to_input_rows'],np.int64)
  patchid36[role]={}
  for pm in rr036['outputs'][role]['patch_face_map']:
   for row in pm['output_face_rows']:patchid36[role][int(row)]=pm['patch_id']
  for x in rr036['accepted_patches']:
   if x['role']==role:ops36.setdefault(role,{})[x['patch_id']]=json.loads(Path(x['source_operation_path']).read_text())
  patchid35[role]={}
  for pm in rr035['role_outputs'][role]['patch_output_face_row_map']:
   for row in pm['output_face_rows']:patchid35[role][int(row)]=pm['patch_id']
  for x in rr035['accepted_patches']:
   if x['role']==role:ops35.setdefault(role,{})[x['patch_id']]=json.loads(Path(x['source_operation_path']).read_text())
  if len(map36[role])!=len(faces[role]) or np.any(map36[role]>=len(parent035[role]['faces'])):raise ValueError(role+' 036→035 face map domain')
  if len(map35[role])!=len(parent035[role]['faces']) or np.any(map35[role]>=len(src[role]['faces'])):raise ValueError(role+' 035→015 face map domain')
  if len(pmap36[role])!=len(points[role]) or np.any(pmap36[role]<0)or np.any(pmap36[role]>=len(parent035[role]['points_world_m'])):raise ValueError(role+' 036→035 point map domain')
  if len(pmap35[role])!=len(parent035[role]['points_world_m']) or np.any(pmap35[role]<0)or np.any(pmap35[role]>=len(src[role]['points_world_m'])):raise ValueError(role+' 035→015 point map domain')
  current_point_to_source[role]=pmap35[role][pmap36[role]]
  if np.any(current_point_to_source[role]<0)or len(np.unique(current_point_to_source[role]))!=len(current_point_to_source[role]):raise ValueError(role+' non-injective 036→015 point ancestry')
  if not np.array_equal(np.asarray(src[role]['points_world_m'])[current_point_to_source[role]],np.asarray(carr[role]['points_world_m'])):raise ValueError(role+' 036→015 point coordinates are not exact')
 def resolve035(role,row):
  row=int(row);source=int(map35[role][row])
  if source>=0:return {source},[source_lineage(role,src[role],source)],set(),set()
  patch=patchid35[role].get(row)
  if patch is None:raise ValueError(f'{role} 035 derived face has no patch owner: {row}')
  op=ops35[role][patch]
  return set(map(int,op['retired_input_face_rows'])),[op['source_lineage']],{patch},set()
 def resolve036(role,row):
  row=int(row);source=int(map36[role][row])
  if source>=0:return resolve035(role,source)
  patch=patchid36[role].get(row)
  if patch is None:raise ValueError(f'{role} 036 derived face has no patch owner: {row}')
  op=ops36[role][patch];orig=set();lines=[];p35=set();p36={patch}
  for r35 in op['source_initial_face_rows']:
   o,l,a,b=resolve035(role,int(r35));orig.update(o);lines.extend(l);p35.update(a);p36.update(b)
  return orig,lines,p35,p36
 # Convert 035 TetGen warning endpoints through the exact 036 point maps, retain only segments still present.
 warning_sets={r:[]for r in faces};warning_resolution=[]
 inv36={r:np.asarray(carr[r]['point_input_to_output_row'],np.int64)for r in faces}
 for rb in wd['roles']:
  for ww in rb['warnings']:
   for si,seg in enumerate(ww['segments']):
    pairs=seg['role_rows'];r0,old0=pairs[0];r1,old1=pairs[1]
    if r0!=r1:warning_resolution.append({'warning_role':rb['role'],'warning_index':ww['warning_index'],'segment':si,'status':'CROSS_ROLE_SEGMENT_UNSUPPORTED','source_segment':pairs});continue
    role=r0;old0=int(old0);old1=int(old1);m=inv36[role]
    if old0>=len(m)or old1>=len(m)or m[old0]<0 or m[old1]<0:
     warning_resolution.append({'warning_role':rb['role'],'warning_index':ww['warning_index'],'segment':si,'status':'RESOLVED_ENDPOINT_REMOVED_BY_036','source_segment':pairs});continue
    u,v=int(m[old0]),int(m[old1]);
    if u==v:warning_resolution.append({'warning_role':rb['role'],'warning_index':ww['warning_index'],'segment':si,'status':'COLLAPSED_ENDPOINTS','source_segment':pairs});continue
    coords=np.asarray(seg['coordinates_f32_m'],np.float32)
    if not np.array_equal(points[role][u],coords[0])or not np.array_equal(points[role][v],coords[1]):raise ValueError('warning coordinate row mapping mismatch')
    if not np.any(np.all(np.sort(faces[role][:,[0,1]],axis=1)==np.sort((u,v)),axis=1)) and not np.any(np.all(np.sort(faces[role][:,[1,2]],axis=1)==np.sort((u,v)),axis=1)) and not np.any(np.all(np.sort(faces[role][:,[2,0]],axis=1)==np.sort((u,v)),axis=1)):
     warning_resolution.append({'warning_role':rb['role'],'warning_index':ww['warning_index'],'segment':si,'status':'EDGE_REMOVED_BY_036','source_segment':pairs,'current_rows':[u,v]});continue
    warning_sets[role].append({'warning_role':rb['role'],'warning_index':int(ww['warning_index']),'segment':si,'edge':(u,v),'source_edge':(old0,old1),'coords':coords.astype(float).tolist()})
 # Build connected components of unresolved warning edges per physical role.
 comps={}
 for role,edges in warning_sets.items():
  left=set(range(len(edges)));cc=[]
  while left:
   seed=left.pop();comp={seed};verts=set(edges[seed]['edge']);changed=True
   while changed:
    changed=False
    for j in list(left):
     if verts&set(edges[j]['edge']):left.remove(j);comp.add(j);verts.update(edges[j]['edge']);changed=True
   cc.append({'indices':sorted(comp),'vertices':verts,'edges':[edges[i]for i in sorted(comp)]})
  comps[role]=cc
 # Current protected target rows and open boundaries.
 prot={r:set(map(int,boundary(faces[r]).reshape(-1)))for r in faces};routes={r:[]for r in faces};cur_route={r:[]for r in faces}
 for x in targ:
  role=x['role'];row=int(rr036['outputs'][role]['protected_target_output_rows'][len(cur_route[role])]);cur_route[role].append(row)
  if row>=len(points[role])or not np.array_equal(points[role][row],np.asarray(x['target_world_m_f32'],np.float32)):raise ValueError('036 target rebase')
  routes[role].append({'canonical_protected_row':int(x['canonical_protected_row']),'row':row});prot[role].add(row)
 # Build one disk patch per connected warning component, including every current incident face around all warning endpoints.
 patch_candidates={r:[]for r in faces};candidate_rejects=[]
 for role in ('O','I','D'):
  F=faces[role];
  for ci,comp in enumerate(comps[role]):
   vs=sorted(comp['vertices']);inc=[]
   for v in vs:inc.append(np.flatnonzero(np.any(F==v,axis=1)))
   rows=np.unique(np.concatenate(inc)).astype(np.int64)
   record={'role':role,'component_index':ci,'warning_indices':sorted({(e['warning_role'],e['warning_index'])for e in comp['edges']}),'warning_edges':[{'source_edge':list(e['source_edge']),'current_edge':list(e['edge']),'warning_role':e['warning_role'],'warning_index':e['warning_index']}for e in comp['edges']],'vertices':vs,'rows':rows}
   if len(rows)>MAX_FACES:candidate_rejects.append({**{k:v for k,v in record.items()if k!='rows'},'status':'WARNING_PATCH_TOO_LARGE','face_count':len(rows)});continue
   if set(vs)&prot[role]:candidate_rejects.append({**{k:v for k,v in record.items()if k!='rows'},'status':'WARNING_PATCH_TOUCHES_PROTECTED_POINT','protected_rows':sorted(set(vs)&prot[role])});continue
   loop=patchloop(F,rows)
   if loop is None:candidate_rejects.append({**{k:v for k,v in record.items()if k!='rows'},'status':'WARNING_PATCH_NOT_DISK','face_count':len(rows)});continue
   if len(loop)>MAX_LOOP:candidate_rejects.append({**{k:v for k,v in record.items()if k!='rows'},'status':'WARNING_LOOP_TOO_LARGE','loop_vertices':len(loop)});continue
   if not disk_ok(points[role],loop):candidate_rejects.append({**{k:v for k,v in record.items()if k!='rows'},'status':'NO_QUALITY_FEASIBLE_WARNING_DISK','loop_vertices':len(loop)});continue
   badalt,badang=stableq(points[role],F[rows]);record.update(loop=list(map(int,loop)),bad_face_count=int(np.count_nonzero((badalt<ALT)|(badang<ANGLE))),old_min_altitude_m=float(badalt.min()),old_min_angle_degrees=float(badang.min()),face_count=len(rows));patch_candidates[role].append(record)
 # Any warning components that produce overlapping star patches are merged once to keep one flat source union.
 for role in ('O','I','D'):
  cs=patch_candidates[role];changed=True
  while changed:
   changed=False
   for i in range(len(cs)):
    if changed:break
    for j in range(i+1,len(cs)):
     if set(cs[i]['rows'])&set(cs[j]['rows']) or set(cs[i]['vertices'])&set(cs[j]['vertices']):
      merged_edges=cs[i]['warning_edges']+cs[j]['warning_edges'];vs=sorted(set(cs[i]['vertices'])|set(cs[j]['vertices']));F=faces[role];rows=np.unique(np.concatenate([np.flatnonzero(np.any(F==v,axis=1))for v in vs])).astype(np.int64);loop=patchloop(F,rows)
      if len(rows)<=MAX_FACES and loop is not None and len(loop)<=MAX_LOOP and not(set(vs)&prot[role])and disk_ok(points[role],loop):
       cs[i].update(vertices=vs,rows=rows,loop=list(map(int,loop)),warning_edges=merged_edges,warning_indices=sorted(set(tuple(x)for x in cs[i]['warning_indices']+cs[j]['warning_indices'])),face_count=len(rows));cs.pop(j);changed=True;break
      else:
       candidate_rejects.append({'role':role,'status':'OVERLAPPING_WARNING_PATCHES_NOT_UNIONABLE','component_indices':[cs[i]['component_index'],cs[j]['component_index']],'face_count':len(rows),'loop_vertices':None if loop is None else len(loop)});cs.pop(j);changed=True;break
 # Exact global owner/index and sequential collision-aware disk remeshes.
 surf=[];edgeidx={};lattice={};pbase={};vb=0
 for r in ('D','O','I'):
  pbase[r]=vb;vb+=len(points[r]);rr,_=pred.float32_surface_records(points[r].astype(np.float64),faces[r],source_seam_groups=());fb=SRC[r][2];surf.extend([(z[0],z[1],z[2],fb+i,tuple(pbase[r]+int(v)for v in z[4]))for i,z in enumerate(rr)]);edgeidx[r]=edgeindex(faces[r],len(points[r]));lattice[r]=tuple(pred.float32_point_lattice_key(x)for x in points[r])
 bones,breceipt=bm.load_bone_barrier_records(pred,root=ROOT,vertex_offset=vb,expected_native_sha256=BONE_NATIVE,expected_manifest_sha256=BONE_MANIFEST)
 if len(bones)!=938272:raise ValueError('all185 coverage')
 idx=pred._prepare_surface_aabb(surf+bones);allowed=set()
 for r in faces:allowed.update(int(x)+SRC[r][2]for x in range(len(faces[r])))
 filtered=fm.FilteredPreparedExactOwner(pred,allowed_retirable_face_ids=allowed,excluded_face_ids=());retired={r:set()for r in faces};acceptedrecs=[];newby={r:[]for r in faces};accepted=[];failed=[];processed=0
 for role in ('O','I','D'):
  F=faces[role];proxy=Proxy(filtered,pbase[role],len(points[role]))
  for c in patch_candidates[role]:
   if time.monotonic()>=deadline:failed.append({'role':role,'status':'BUDGET_EXPIRED','remaining':sum(len(patch_candidates[r])for r in ('O','I','D'))-processed});break
   rows=np.asarray(c['rows'],np.int64);ids={int(x)+SRC[role][2]for x in rows};ex=set(int(x)+SRC[r][2]for r in ('O','I','D')for x in retired[r]);ex.update(ids);filtered.set_excluded(ex);ext=external(c['loop'],edgeidx[role],rows,retired[role]);ch=None;cr=None
   try:
    ch,cr=cm.prune_chords(c['loop'],lattice[role],idx,proxy,max_seconds=4.,cache={})
    if cr.get('blocked_boundary_edges'):raise ValueError('warning patch boundary edge blocked')
    first=3000000+processed*1000
    tris,recs,dp=dm.collision_aware_dp_disk(c['loop'],points[role],points[role],ext|set(ch),idx,proxy,lattice[role],acceptedrecs,first,max_seconds=PER_PATCH,max_branches=512,chunk_size=64,minimum_altitude_m=ALT,minimum_angle_degrees=ANGLE)
    nf=np.asarray(tris,np.int64).reshape((-1,3));qa,qg=stableq(points[role],nf)
    if len(nf)!=len(c['loop'])-2 or np.any(qa<ALT)or np.any(qg<ANGLE):raise ValueError('warning disk postquality')
    prior=[];lins=[];origrows=set();midrows=set();midpatches=set();currentpatches=set()
    for x in rows:
     x=int(x);o,ll,p35,p36=resolve036(role,x);origrows.update(o);lins.extend(ll);midrows.add(int(map36[role][x]));midpatches.update(p35);currentpatches.update(p36);prior.extend(p35);prior.extend(p36)
    line=merge_lineage(lins);pid=f'037-{role}-warning-{processed:03d}'
    source_newfaces=current_point_to_source[role][nf]
    if np.any(source_newfaces<0):raise ValueError(role+' new patch point lacks 015 source row')
    opdoc={'schema':'numi.warning-local-connected-union-operation.v1','patch_id':pid,'role':role,'warning_components':c['warning_indices'],'warning_edges':c['warning_edges'],'retired_input_face_rows':sorted(origrows),'source_036_face_rows':rows.astype(int).tolist(),'source_035_face_rows':sorted(q for q in midrows if q>=0),'source_original_face_rows':sorted(origrows),'prior_patch_ids_consumed':sorted(set(prior)),'source_035_patch_ids_consumed':sorted(midpatches),'source_036_patch_ids_consumed':sorted(currentpatches),'boundary_loop_036_point_rows':list(map(int,c['loop'])),'new_faces_036_point_rows':nf.astype(int).tolist(),'new_faces_input_point_rows':source_newfaces.astype(int).tolist(),'source_lineage':line,'selected_quality':{'minimum_altitude_m':float(qa.min()),'minimum_angle_degrees':float(qg.min())},'lineage_semantics':'Each replacement triangle inherits the complete flattened original-015 source-face union of consumed 036 faces; patch-level lineage only, not per-triangle overlap.','chord_pruning':cr,'disk_dp':dp}
    opath=out/(pid+'.json');opath.write_text(json.dumps(opdoc,indent=2,sort_keys=True)+'\n');newby[role].append({'patch_id':pid,'faces':nf,'rows':rows,'op_path':str(opath)});retired[role].update(map(int,rows));acceptedrecs.extend(recs);accepted.append({'patch_id':pid,'role':role,'warning_components':c['warning_indices'],'warning_edges':c['warning_edges'],'retired_current_face_count':len(rows),'new_face_count':len(nf),'minimum_altitude_m':float(qa.min()),'minimum_angle_degrees':float(qg.min()),'exact_external_pairs':int(dp['statistics']['external_exact_pairs']),'exact_self_conflicts':int(dp['statistics']['self_pair_conflicts']),'source_operation_path':str(opath),'source_operation_sha256':sha(opath),'source_original_face_count':len(origrows)})
   except Exception as e:failed.append({'role':role,'warning_components':c['warning_indices'],'warning_edges':c['warning_edges'],'status':'WARNING_DISK_NOT_EXACTLY_TRIANGULABLE','error':repr(e),'chord_pruning':cr})
   processed+=1
 # Compose raw successor outputs and explicit source row maps.
 outputs={};quality={}
 for role in ('O','I','D'):
  F=faces[role];ret=np.asarray(sorted(retired[role]),np.int64);keep=np.flatnonzero(~np.isin(np.arange(len(F),dtype=np.int64),ret));pl=newby[role];allnew=np.concatenate([x['faces']for x in pl],axis=0)if pl else np.empty((0,3),np.int64);ff=np.concatenate((F[keep],allnew));parentfm=np.concatenate((keep,np.full(len(allnew),-1,np.int64)));used=np.unique(ff);inv=np.full(len(points[role]),-1,np.int64);inv[used]=np.arange(len(used));fc=inv[ff]
  if np.any(fc<0):raise ValueError(role+' point compaction')
  target_rows=np.asarray([x['row']for x in routes[role]],np.int64)
  if len(target_rows)and np.any(inv[target_rows]<0):raise ValueError(role+' removed a protected target')
  b0={tuple(sorted((int(inv[u]),int(inv[v]))))for u,v in boundary(faces[role])};b1={tuple(map(int,e))for e in boundary(fc)}
  if any(u<0 or v<0 for u,v in b0)or b0!=b1:raise ValueError(role+' oriented boundary remap')
  qa,qg=stableq(points[role][used],fc);quality[role]={'face_count':len(fc),'bad_union':int(np.count_nonzero((qa<ALT)|(qg<ANGLE))),'minimum_altitude_m':float(qa.min()),'minimum_angle_degrees':float(qg.min()),'boundary_edges':len(b1)}
  parent_to_source=np.full(len(F),-1,np.int64);valid36=np.flatnonzero(map36[role]>=0);parent_to_source[valid36]=map35[role][map36[role][valid36]]
  face_input=np.concatenate((parent_to_source[keep],np.full(len(allnew),-1,np.int64)))
  source_face_count=len(src[role]['faces']);direct=face_input[face_input>=0]
  if len(np.unique(direct))!=len(direct):raise ValueError(role+' duplicate retained original face map')
  retired_source=np.setdiff1d(np.arange(source_face_count,dtype=np.int64),direct,assume_unique=False)
  derived_rows=np.flatnonzero(face_input<0);current_to_source=current_point_to_source[role]
  newfaces_parent=ff[derived_rows];newfaces_source=current_to_source[newfaces_parent]
  if np.any(newfaces_source<0):raise ValueError(role+' derived output face point ancestry')
  input_face_to_output=np.full(source_face_count,-1,np.int64);input_face_to_output[direct]=np.flatnonzero(face_input>=0)
  source_point_map=current_to_source[used];source_point_count=len(src[role]['points_world_m']);source_point_to_output=np.full(source_point_count,-1,np.int64)
  if len(np.unique(source_point_map))!=len(source_point_map):raise ValueError(role+' duplicate original point map')
  source_point_to_output[source_point_map]=np.arange(len(used),dtype=np.int64)
  if not np.array_equal(np.asarray(src[role]['points_world_m'])[source_point_map],np.asarray(carr[role]['points_world_m'])[used]):raise ValueError(role+' output point source reconstruction')
  derived_provenance=[]
  patch_boundaries=np.cumsum([len(x['faces']) for x in pl],dtype=np.int64)
  for outrow, parentrow in enumerate(parentfm):
   if parentrow<0:
    j=outrow-len(keep);pi=int(np.searchsorted(patch_boundaries,j,side='right'));derived_provenance.append(('037',pl[pi]['patch_id']))
   elif int(map36[role][parentrow])<0:derived_provenance.append(('036',patchid36[role][int(parentrow)]))
   elif int(map35[role][int(map36[role][parentrow])])<0:derived_provenance.append(('035',patchid35[role][int(map36[role][parentrow])]))
   else:derived_provenance.append(('source',''))
  if len(np.unique(direct)) + len(retired_source) != source_face_count or not np.array_equal(np.sort(np.concatenate((direct,retired_source))),np.arange(source_face_count,dtype=np.int64)):raise ValueError(role+' original source-face partition')
  target_source_rows=current_to_source[target_rows]
  if np.any(target_source_rows<0)or np.any(source_point_to_output[target_source_rows]<0):raise ValueError(role+' protected target source map')
  cursor=len(keep);patchmap=[]
  for x in pl:
   n=len(x['faces']);ro=list(range(cursor,cursor+n));op=json.loads(Path(x['op_path']).read_text());op['new_output_face_rows']=ro;op['patches']=[{'patch_id':x['patch_id'],'retired_input_face_rows':op['retired_input_face_rows'],'source_original_face_rows':op['source_original_face_rows'],'new_output_face_rows':ro,'new_faces_input_point_rows':op['new_faces_input_point_rows'],'source_lineage':op['source_lineage']}];Path(x['op_path']).write_text(json.dumps(op,indent=2,sort_keys=True)+'\n');cursor+=n;patchmap.append({'patch_id':x['patch_id'],'output_face_rows':ro,'retired_current_face_rows':x['rows'].astype(int).tolist(),'retired_original_face_rows':op['retired_input_face_rows']})
  all_new_patch_ids=np.asarray([derived_provenance[q][1] for q in derived_rows],dtype='U64')
  all_new_patch_sources=np.asarray([derived_provenance[q][0] for q in derived_rows],dtype='U8')
  all_new_patch_indices=np.asarray([([x['patch_id'] for x in pl].index(pid) if pid in [x['patch_id'] for x in pl] else -1) for pid in all_new_patch_ids],np.int64)
  opath=out/(role+'-warning-local-connected-union-proposal.npz');np.savez_compressed(opath,points_world_m=carr[role]['points_world_m'][used],points_world_m_f32=np.asarray(carr[role]['points_world_m'][used],np.float32),faces=fc,source_candidate_sha256=np.array([SRC[role][1]]),immediate_parent_candidate_sha256=np.array([CUR[role][1]]),point_output_to_input_rows=source_point_map.astype(np.int64),point_input_to_output_row=source_point_to_output,point_output_to_parent_rows=used.astype(np.int64),point_parent_input_to_output_row=inv,face_output_to_input_rows=face_input,face_input_to_output_row=input_face_to_output,face_output_to_parent_rows=parentfm,retired_input_face_rows=retired_source,new_faces_input_point_rows=newfaces_source.astype(np.int64),new_faces_parent_point_rows=newfaces_parent.astype(np.int64),new_output_face_patch_ids=all_new_patch_indices,new_output_face_patch_source=all_new_patch_sources,new_output_face_patch_id=all_new_patch_ids,candidate_patch_operation_ids=np.asarray([x['patch_id']for x in pl]));outputs[role]={'path':str(opath),'sha256':sha(opath),'source_candidate_sha256':SRC[role][1],'immediate_parent_candidate_sha256':CUR[role][1],'point_count':len(used),'face_count':len(fc),'derived_face_count':len(derived_rows),'retired_original_face_count':len(retired_source),'new_037_face_count':len(allnew),'protected_target_output_rows':source_point_to_output[target_source_rows].astype(int).tolist(),'boundary_edges':len(b1),'patch_face_map':patchmap,'derived_face_kind_counts':{k:int(sum(1 for q in derived_provenance if q[0]==k))for k in ('035','036','037')},'source_map_semantics':'point and face maps are composed to immutable 015 input domains; parent maps separately retain 036 row identities'}
 report={'schema':'numi.warning-local-connected-union-proposal.v1','status':'WARNING_LOCAL_EXACT_UNION_PROPOSALS_READY'if accepted else'NO_WARNING_LOCAL_PATCH_ACCEPTED','inputs_035_report_sha256':R035_SHA,'inputs_035_verifier_sha256':V035_SHA,'inputs_036_report_sha256':R036_SHA,'warning_source':{'path':str(WARN),'sha256':WARN_SHA},'parent_035_proposals':{r:{'path':rr035['role_outputs'][r]['path'],'sha256':rr035['role_outputs'][r]['sha256']}for r in rr035['role_outputs']},'current_036':{r:{'path':str(CUR[r][0]),'sha256':CUR[r][1]}for r in CUR},'source_015':{r:{'path':str(SRC[r][0]),'sha256':SRC[r][1]}for r in SRC},'protected149':{'path':str(TARGET),'sha256':TARGET_SHA,'routes_by_role':{r:len(routes[r])for r in routes}},'warning_resolution_before':warning_resolution,'candidate_rejections':candidate_rejects,'candidate_components':{r:[{k:v for k,v in c.items()if k!='rows'}for c in comps[r]]for r in comps},'accepted_patch_count':len(accepted),'failed_patch_count':len(failed),'accepted_patches':accepted,'failed_patches':failed,'outputs':outputs,'quality_after':quality,'all185_records':len(bones),'exact_scope':'Each accepted warning-local union was checked by exact 616 DP against one complete current 036 O/I/D plus all185 F32/F64 index; only accepted-retired faces and the current warning patch were masked. The source warning segments were translated through pinned 035→036 point maps and exact F32 coordinate checks. Output point/face lineage maps are composed to the original 015 input rows and retain explicit immediate-parent 036 maps.','coordinates_changed':False,'physical_admission':False,'runtime':0,'accepted_steps':0,'GPU':False,'elapsed_seconds':time.monotonic()-started}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'out':str(out),'accepted':len(accepted),'failed':len(failed),'resolved_before':len(warning_resolution),'quality':quality,'elapsed_s':report['elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
