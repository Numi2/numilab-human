"""One bounded Skin interior-only edge-quotient pass, with exact379/198 predicates and immutable registered surfaces."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=imp('existing379',H/'transfer_surface_contractions_indexed_379_root.py');m=T.m
def link(t,vertex):
 ans=set()
 for tet in t:
  other=sorted(set(map(int,tet))-set(vertex))
  for k in range(len(other)+1):ans.update(combinations(other,k))
 return ans
def admissible(t,rows_u,rows_w,u,w):
 edge=np.intersect1d(rows_u,rows_w,assume_unique=True)
 return bool(len(edge)) and link(t[rows_u],(u,))&link(t[rows_w],(w,))==link(t[edge],(u,w))
# Collapsing an interior construction vertex back to its parent corner must
# preserve the exact original tetrahedron and improve its near-corner slivers.
fp=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1e-8,1e-8,1e-8]],np.float32)
ft=m.orient(fp,[[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
fn=np.where(ft==4,0,ft);sn=np.sort(fn,axis=1);fn=fn[~np.any(sn[:,1:]==sn[:,:-1],axis=1)]
assert m.certify(fp,ft,fn)==(True,'PASS') and np.all(m.passes(*m.measurements(fp,fn)))
assert admissible(ft,np.flatnonzero(np.any(ft==0,axis=1)),np.arange(4),0,4)
assert 4 not in T.boundary(ft)
start=time.monotonic();out=H/'skin-interior-fixed-boundary-quotient-602-root';out.mkdir(exist_ok=False)
ip=H/'current-paired-fixed-boundary-composition-597-root/skin-current-connected-volume-proposal.npz'
assert sha(ip)=='3f79b7dfab07fb956e173cc05e5a95d80552da2d076861a7f8aab74684037c68'
with np.load(ip) as z:a={k:z[k] for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'];mat=a['material_indices'];d,q=m.measurements(p,t);good=m.passes(d,q)
assert np.all(d>0) and int((~good).sum())==890 and np.all(mat==15)
mask=np.ones(len(t),bool)
R=imp('registered309',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
source=H/'combined-child-handoff-142/skin.node';registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
sp=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32)
registered,counts=R.registered_source_rows_from_map(registration,sp)
assert np.array_equal(p[:len(sp)][registered].view('u4'),sp[registered].view('u4'))
boundary=T.boundary(t);isboundary=np.zeros(len(p),bool);isboundary[np.unique(boundary)]=True;isboundary[:len(sp)]|=registered
flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
def incident(v):return order[offset[v]:offset[v+1]]//4
pat=np.array(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),np.int64)
e=np.sort(t[mask&~good][:,pat].reshape(-1,2),axis=1);e=np.unique(e,axis=0)
e=e[~(isboundary[e[:,0]]&isboundary[e[:,1]])]
length=np.linalg.norm(p[e[:,0]].astype(float)-p[e[:,1]],axis=1);idx=np.lexsort((e[:,1],e[:,0],length));e=e[idx];length=length[idx]
used=set();removed=[];newall=[];newmat=[];ops=[];rej=Counter();visited=0
for (u0,w0),dist in zip(e,length):
 if time.monotonic()-start>20 or len(ops)>=2000:break
 visited+=1;u0=int(u0);w0=int(w0)
 if u0 in used or w0 in used:rej['conflict']+=1;continue
 best=None
 for u,w in ((u0,w0),(w0,u0)):
  if isboundary[w]:rej['retired_boundary_forbidden']+=1;continue
  ru=incident(u);rw=incident(w);both=np.union1d(ru,rw)
  if not np.all(mask[both]) or set(map(int,t[both].ravel()))&used:rej['outside_protection_or_conflict']+=1;continue
  if len(np.unique(mat[both]))!=1:rej['material_interface']+=1;continue
  if not admissible(t,ru,rw,u,w):rej['simplicial_link_condition']+=1;continue
  old=t[rw];new=np.where(old==w,u,old);sn=np.sort(new,axis=1);delete=np.any(sn[:,1:]==sn[:,:-1],axis=1);new=new[~delete]
  if not len(new):rej['empty_cavity']+=1;continue
  nd,nq=m.measurements(p,new);oldgood=good[rw]
  before=int((~oldgood).sum());after=int((~m.passes(nd,nq)).sum())
  if np.any(nd<=0) or np.any(oldgood[~delete]&~m.passes(nd,nq)) or after>=before:rej['quality_no_progress']+=1;continue
  ok,why=m.certify(p,old,new)
  if not ok:rej[why]+=1;continue
  value=(before-after,-after,float(nq.min()),-w)
  if best is None or value>best[0]:best=(value,u,w,rw,new,both,before,after)
 if best is None:continue
 _,u,w,rows,new,both,before,after=best
 used.update(map(int,t[both].ravel()));removed.extend(rows.tolist());newall.extend(new.tolist());newmat.extend([int(mat[rows[0]])]*len(new))
 ops.append({'old_rows':rows.tolist(),'new_rows':new.tolist(),'retained_point_row':u,'retired_interior_point_row':w,'full_endpoint_star_rows':both.tolist(),'edge_length_m':float(dist),'bad_before':before,'bad_after':after})
keep=np.ones(len(t),bool);keep[removed]=False;new=np.asarray(newall,np.int64).reshape(-1,4)
candidate=np.vstack((t[keep],new));cm=np.r_[mat[keep],np.asarray(newmat,np.int32)];cd,cq=m.measurements(p,candidate)
assert np.all(cd>0) and int((~m.passes(cd,cq)).sum())==890-sum(o['bad_before']-o['bad_after'] for o in ops)
assert np.array_equal(T.boundary_identity(T.boundary(candidate)),T.boundary_identity(boundary))
assert len(np.unique(np.sort(candidate,axis=1),axis=0))==len(candidate)
parents=[];po=[0]
for o in ops:
 for _ in o['new_rows']:parents.extend(o['old_rows']);po.append(len(parents))
mapping=np.r_[np.flatnonzero(keep),np.full(len(new),-1,np.int64)];reverse=np.full(len(t),-1,np.int64);reverse[np.flatnonzero(keep)]=np.arange(keep.sum())
quotient=np.arange(len(p),dtype=np.int64)
for o in ops:quotient[o['retired_interior_point_row']]=o['retained_point_row']
a.update(tetrahedra=candidate,material_indices=cm,current_geometry_parent_path=np.asarray([str(ip)]),current_geometry_parent_sha256=np.asarray([sha(ip)]),output_cell_to_input_cell=mapping,input_cell_to_output_cell=reverse,retained_input_cell_rows=np.flatnonzero(keep),replacement_parent_offsets=np.asarray(po,np.int64),replacement_parent_cell_rows=np.asarray(parents,np.int64),replacement_tet_offsets=np.asarray([0,len(new)],np.int64),replacement_tetrahedra=new,source_interior_vertex_quotient=quotient,output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
cp=None
if ops:cp=out/'skin-interior-quotient-volume-proposal.npz';np.savez_compressed(cp,**a)
r={'status':'PASS_SKIN_INTERIOR_FIXED_BOUNDARY_QUOTIENT_PROPOSAL' if ops else 'NO_VALID_CONNECTIVITY_PROGRESS','input':pin(ip),'registered_source_map':pin(registration),'candidate':pin(cp) if cp else None,'producer':pin(Path(__file__)),'operations':ops,'eligible_edges':len(e),'visited_edges':visited,'passing_operations':len(ops),'bad_before':890,'bad_after':int((~m.passes(cd,cq)).sum()),'rejections':dict(rej),'all_coordinates_fixed':True,'complete_oriented_boundary_fixed':True,'retired_points_strictly_interior':True,'simplicial_link_condition':True,'per_cavity_exact_F32_volume_fixed':True,'native_thresholds_unchanged':True,'source330_mass_unchanged':True,'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='operations'}),flush=True)
