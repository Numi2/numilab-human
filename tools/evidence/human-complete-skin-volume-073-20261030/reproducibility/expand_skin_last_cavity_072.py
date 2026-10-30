#!/usr/bin/env python3
"""Restore all pending Skin PLC source facets in one local-cavity batch."""
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,json,sys,hashlib,subprocess,time,ast,itertools
from scipy.optimize import linprog
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
D=H/'skin010-restored-cavity-batch-072-expand1';D.mkdir(exist_ok=False)
S=H/'skin010-local-tetgen-070/skin-local-tetgen-proposal.npz'
with np.load(S)as z:a={k:z[k]for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'];B=a['boundary_faces'];original=202037;pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
with np.load(H/'skin010-inverse-boundary-splits-068/skin-boundary-restoration-proposal.npz')as z:map0=z['source065_node_to_candidate_node']
with np.load(H/'skin010-restored-cavity-batch-069/skin-restored-boundary-proposal.npz')as z:map1=z['source068_node_to_candidate_node']
map2=a['source069_node_to_candidate_node'];rawmap=np.full(len(map0),-1,np.int64)
valid=np.flatnonzero(map0>=0);rawmap[valid]=map1[map0[valid]];valid=np.flatnonzero(rawmap>=0);rawmap[valid]=map2[rawmap[valid]]
mat=a['material_indices'];assert np.all(mat==15)
with np.load(H/'verified-volume-preparation-010-merged038005/plcs/skin-shell.npz')as z:sf=z['faces'];markers=z['face_marker']
def key(q):return tuple(sorted(map(int,q)))
source={key(f):i for i,f in enumerate(sf)};globalB={key(f)for f in B};missing=[i for i,f in enumerate(sf)if key(f)not in globalB]
vp=defaultdict(set)
for i in missing:
 for v in sf[i]:vp[int(v)].add(i)
diag=json.loads((H/'tetgen-scale-relative-tolerance-059/boundary-subdivision-audit-001/nearest-edge-diagnostic.json').read_text())['skin']['rows']
steiner=set()
for r in diag:
 v=int(rawmap[r['output_node_row']])
 if v>=original:
  steiner.add(v);vp[v].update(set(r['incident_missing_input_face_rows'])&set(missing))
extra_parent={}
for f in B:
 k=key(f)
 if k in source:continue
 parents=set.intersection(*(vp[int(v)]for v in f))
 assert len(parents)==1,(f,parents)
 extra_parent[k]=next(iter(parents))
expected=Counter(extra_parent.values())
def parent(f):
 k=key(f);return source[k]if k in source else extra_parent[k]
sel=np.flatnonzero(np.any(np.isin(t,list(steiner)),axis=1))
def pack_tri(f):
 f=np.sort(f.astype(np.uint64),axis=1);return(f[:,0]<<40)|(f[:,1]<<20)|f[:,2]
allkeys=np.empty((len(t),4),np.uint64)
for j,face in enumerate(pat):allkeys[:,j]=pack_tri(t[:,face])
wanted=np.unique(allkeys[sel]);neighbors=np.flatnonzero(np.any(np.isin(allkeys,wanted),axis=1));sel=np.union1d(sel,neighbors);del allkeys
st=t[sel];fs=st[:,pat].reshape(-1,3)
_,first,inv,cnt=np.unique(np.sort(fs,axis=1),axis=0,return_index=True,return_inverse=True,return_counts=True)
root=np.arange(len(sel))
def find(x):
 while root[x]!=x:root[x]=root[root[x]];x=int(root[x])
 return x
def union(x,y):
 x,y=find(x),find(y)
 if x!=y:root[max(x,y)]=min(x,y)
owners=defaultdict(list)
for i,k in enumerate(inv):owners[int(k)].append(i//4)
for ids in owners.values():
 for x in ids[1:]:union(ids[0],x)
parentowners=defaultdict(list)
for i in first[cnt==1]:
 f=fs[i]
 if key(f)in globalB:parentowners[parent(f)].append(int(i)//4)
for ids in parentowners.values():
 for x in ids[1:]:union(ids[0],x)
components=defaultdict(list)
for i in range(len(sel)):components[find(i)].append(int(sel[i]))
owner=P/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(owner.parent.parent));exact=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
helper=H/'repair_boundary_vertex_regions_064.py';hs=helper.read_text()
# Same exact fan certificate, now against the restored shell's exact volume.
hs=hs.replace("def solve(ids):","def solve(ids,custom_boundary):").replace("b=f[ix[ct==1]]","b=custom_boundary;ct=np.ones(len(b),np.int64)")
hs=hs.replace("if sum(values)!=sum(det(*(coords(x)for x in tet))for tet in t[ids]):return None","if sum(values)!=sum(det((0,0,0),*(coords(x)for x in face))for face in b):return None")
tree=ast.parse(hs);funcs=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[])
ele=np.zeros((len(t),6),np.int64);ele[:,5]=mat;ns=dict(np=np,itertools=itertools,linprog=linprog,exact=exact,p=p,t=t,ele=ele,pat=pat,ints={});exec(compile(funcs,str(helper),'exec'),ns)
reports=[];accepted=[];newpoints=[];newtets=[];removed=[];newmats=[]
for ci,ids in enumerate(components.values()):
 ids=np.asarray(ids,np.int64);f=t[ids][:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True);oldb=f[ix[ct==1]]
 cut=[];prows=Counter()
 for face in oldb:
  if key(face)in globalB:prows[parent(face)]+=1
  else:cut.append(face)
 rr={'component':ci,'source_cells':len(ids),'parent_original_PLC_faces':list(prows)}
 if any(n!=(expected[r]if r in expected else 1)for r,n in prows.items()):
  rr['status']='REJECT_INCOMPLETE_SOURCE_FACE_CHILDREN';reports.append(rr);continue
 b=np.vstack([np.asarray(cut,np.int64).reshape(-1,3),sf[list(prows)]])
 assert not set(map(int,np.unique(b)))&steiner
 cdir=D/f'component-{ci:03d}';cdir.mkdir()
 np.savez_compressed(cdir/'restored-boundary.npz',boundary_faces_global_node_rows=b,source_cell_rows=ids,original_PLC_face_rows=np.asarray(list(prows)))
 gate,proposal=ns['solve'](ids,b);rr['fan_gate']=gate
 if proposal is None:
  rr['status']='FAN_REJECTED_LOCAL_TETGEN_PENDING';reports.append(rr);continue
 _,fan,c,material=proposal;out=fan.copy()
 if c is not None:out[out<0]=len(p)+len(newpoints);newpoints.append(c)
 newtets.append(out);removed.extend(ids.tolist());newmats.extend([material]*len(fan));rr['status']='PASS_RESTORED_BOUNDARY_EXACT_FAN';rr['new_cells']=len(fan);accepted.append(ci);reports.append(rr)
mask=np.ones(len(t),bool);mask[removed]=False;pp=np.vstack([p,np.asarray(newpoints,np.float32).reshape(-1,3)]);tt=np.vstack([t[mask],*newtets]);mm=np.r_[mat[mask],newmats].astype(np.int32)
used=np.unique(tt);assert set(range(original))<=set(map(int,used));rows=np.r_[np.arange(original),used[used>=original]];nm=np.full(len(pp),-1,np.int64);nm[rows]=np.arange(len(rows));pp=pp[rows];tt=nm[tt]
q=pp[tt].astype(np.float64);dd=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
f=tt[:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True);bb=f[ix[ct==1]];bkeys={key(f)for f in bb};skeys=set(source)
r={'status':'BATCHED_RESTORED_SKIN_CAVITY_PARTIAL_NOT_PHYSICAL_ADMISSION','physical_admission':False,'runtime':0,'accepted_steps':0,'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'components':len(components),'accepted_components':len(accepted),'candidate_nodes':len(pp),'candidate_cells':len(tt),'F32_zero':int(np.sum(dd==0)),'F32_negative':int(np.sum(dd<0)),'missing_original_PLC_facets':len(skeys-bkeys),'extra_boundary_facets':len(bkeys-skeys),'nonmanifold_tet_faces':int(np.sum(ct>2)),'material_histogram':{str(k):int(v)for k,v in zip(*np.unique(mm,return_counts=True))},'details':reports}
np.savez_compressed(D/'skin-restored-boundary-proposal.npz',points_world_m=pp,tetrahedra=tt,material_indices=mm,boundary_faces=bb,source068_node_to_candidate_node=nm[:len(p)],retained_source068_cell_rows=np.flatnonzero(mask))
(D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='details'}))
