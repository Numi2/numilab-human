#!/usr/bin/env python3
"""Bounded batched cavity construction; preserves all anatomical coordinates."""
from pathlib import Path
import numpy as np,json,hashlib,sys,time,itertools
from scipy.optimize import linprog
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
R=H/'tetgen-scale-relative-tolerance-059/fat009-run001';D=H/'fat009-batch-cavity-064';D.mkdir(exist_ok=False)
owner=P/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(owner.parent.parent));exact=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
C=H/'tetgen-scale-relative-tolerance-059/fat009-local-patch-classification-001/fat009-run001/bad-cell-one-ring-patches.npz'
with np.load(C)as z:meta={k:z[k]for k in z.files}
nodes=np.loadtxt(R/'fat.1.node',skiprows=1);ele=np.loadtxt(R/'fat.1.ele',skiprows=1,dtype=np.int64)
p=nodes[:,1:4].astype(np.float32);t=ele[:,1:5]-1;pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
def pack(f):
 a=np.sort(f.astype(np.uint64),axis=1);return(a[:,0]<<40)|(a[:,1]<<20)|a[:,2]
fkeys=np.empty((len(t),4),np.uint64)
for j,face in enumerate(pat):fkeys[:,j]=pack(t[:,face])
flat=fkeys.ravel();order=np.argsort(flat,kind='stable');sk=flat[order];same=np.flatnonzero(sk[1:]==sk[:-1]);adj=np.full(len(flat),-1,np.int64);adj[order[same]]=order[same+1]//4;adj[order[same+1]]=order[same]//4;adj=adj.reshape(-1,4)
del flat,order,sk,same,fkeys
ints={}
def ip(x):return tuple(int(float(v).as_integer_ratio()[0])<<(149-(float(v).as_integer_ratio()[1].bit_length()-1))for v in x)
def coords(row):
 row=int(row)
 if row not in ints:ints[row]=ip(p[row])
 return ints[row]
def det(a,b,c,d):
 u=[b[k]-a[k]for k in range(3)];v=[c[k]-a[k]for k in range(3)];w=[d[k]-a[k]for k in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
def certify(ids,b,fan,center,material):
 # -1 denotes a newly constructed free interior point; other rows are existing.
 ci=ip(center)if center is not None else None
 def xyzint(row):return ci if row<0 else coords(row)
 values=[det(*(xyzint(int(x))for x in tet))for tet in fan]
 if not values or any(x<=0 for x in values):return None
 if sum(values)!=sum(det(*(coords(x)for x in tet))for tet in t[ids]):return None
 ff=fan[:,pat].reshape(-1,3);_,first,ct=np.unique(np.sort(ff,axis=1),axis=0,return_index=True,return_counts=True)
 if np.any(ct>2):return None
 boundary=ff[first[ct==1]]
 def cyc(tri):
  a,b,c=map(int,tri);return min((a,b,c),(b,c,a),(c,a,b))
 if sorted(cyc(x)for x in boundary)!=sorted(cyc(x)for x in b):return None
 used=np.unique(ff);local=np.vstack([p[used[used>=0]],np.asarray(center,np.float32).reshape(1,3)])if center is not None else p[used]
 lookup={int(x):j for j,x in enumerate(used[used>=0])}
 if center is not None:lookup[-1]=len(local)-1
 lf=np.array([[lookup[int(x)]for x in tri]for tri in ff[first]],np.int64)
 rec,_=exact.float32_surface_records(local,lf);gate=exact._audit_pair(rec,rec,same_surface=True)
 if gate['count']:return None
 return(ids,fan,center,int(material))
def solve(ids):
 rr={'cells':len(ids)};f=t[ids][:,pat].reshape(-1,3);_,ix,ct=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True);b=f[ix[ct==1]]
 edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1);_,ec=np.unique(edges,axis=0,return_counts=True)
 if np.any(ct>2)or np.any(ec!=2):return dict(rr,status='REJECT_TOPOLOGY'),None
 labs=np.unique(ele[ids,5])
 if len(labs)!=1:return dict(rr,status='REJECT_MIXED_ATTR',attributes=labs.tolist()),None
 q=p[b].astype(np.float64);normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(normal,axis=1)
 if np.any(area==0):return dict(rr,status='REJECT_ZERO_AREA'),None
 # Boundary-vertex fan is exact and uses no new coordinates, including on thin cavities.
 for anchor in np.unique(b):
  fb=b[~np.any(b==anchor,axis=1)]
  if not len(fb):continue
  v=coords(anchor);ds=[det(v,*(coords(x)for x in tri))for tri in fb]
  if any(x<=0 for x in ds):continue
  fan=np.column_stack([np.full(len(fb),int(anchor)),fb]);proposal=certify(ids,b,fan,None,labs[0])
  if proposal is not None:return dict(rr,status='PASS_BOUNDARY_VERTEX_FAN',anchor_node_row=int(anchor),new_cells=len(fan),material=int(labs[0]),exact_volume_conserved=True,exact_crossing_count=0),proposal
 normal/=area[:,None];v=p[np.unique(b)].astype(np.float64);origin=(v.max(0)+v.min(0))*.5;scale=float(np.max(v.max(0)-v.min(0)))
 sol=linprog([0,0,0,-1],A_ub=np.column_stack([normal,np.ones(len(b))]),b_ub=np.einsum('ij,ij->i',normal,(q[:,0]-origin)/scale),bounds=[(-2,2)]*3+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
 if not sol.success or sol.x[3]<=0:return dict(rr,status='REJECT_NO_STAR'),None
 c=(origin+sol.x[:3]*scale).astype(np.float32);axes=[[np.nextafter(x,np.float32(-np.inf)),x,np.nextafter(x,np.float32(np.inf))]for x in c];best=None;margin=-np.inf
 for a in itertools.product(*axes):
  a=np.asarray(a,np.float32);m=float(np.min(np.einsum('ij,ij->i',normal,q[:,0]-a.astype(np.float64))))
  if m>margin:best=a.copy();margin=m
 if margin<=0:return dict(rr,status='REJECT_NO_F32_KERNEL',margin_m=margin),None
 fan=np.column_stack([np.full(len(b),-1),b]);proposal=certify(ids,b,fan,best,labs[0])
 if proposal is None:return dict(rr,status='REJECT_FAN_CERTIFICATION'),None
 return dict(rr,status='PASS_INTERIOR_KERNEL_FAN',new_cells=len(b),margin_m=margin,material=int(labs[0]),center=best.tolist(),exact_volume_conserved=True,exact_crossing_count=0),proposal
attempts=[];proposals=[];badset=set(map(int,meta['bad_ele_rows_0based']))
for k in range(len(meta['patch_cell_offsets'])-1):
 ids=np.unique(meta['patch_cell_rows_0based'][meta['patch_cell_offsets'][k]:meta['patch_cell_offsets'][k+1]])
 for radius in range(5):
  rr,proposal=solve(ids);rr.update(patch=k,radius=radius);attempts.append(rr)
  if proposal is not None:
   proposals.append((len(badset&set(map(int,ids))),k,radius,proposal));break
  neighbors=adj[ids].ravel();neighbors=neighbors[neighbors>=0];ids=np.union1d(ids,neighbors)
  if len(ids)>2048:break
 accepted=[];taken=set();newpoints=[];newtets=[];newattrs=[];selected=[]
for covered,k,radius,(ids,fan,c,mat)in sorted(proposals,key=lambda x:(-x[0],len(x[3][0]),x[1])):
 if taken&set(map(int,ids)):continue
 taken.update(map(int,ids));selected.extend(ids.tolist());outfan=fan.copy()
 if c is not None:outfan[outfan<0]=len(p)+len(newpoints);newpoints.append(c)
 newtets.append(outfan);newattrs.extend([mat]*len(fan));accepted.append({'patch':k,'radius':radius,'replaced_cells':len(ids),'new_cells':len(fan),'covered_bad_cells':covered,'new_interior_point':c is not None})
mask=np.ones(len(t),bool);mask[selected]=False;pp=np.vstack([p,np.asarray(newpoints,np.float32).reshape(-1,3)]);tt=np.vstack([t[mask],*newtets]);mm=np.r_[ele[mask,5],newattrs].astype(np.int32)
q=pp[tt].astype(np.float64);dd=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
np.savez_compressed(D/'fat-partial-proposal.npz',points_world_m=pp,tetrahedra=tt,material_indices=mm,retained_source_element_ids=ele[mask,0],replaced_source_element_ids=ele[selected,0])
report={'status':'BATCHED_FIXED_CURRENT_BOUNDARY_REPAIR_PARTIAL_NOT_PHYSICAL_ADMISSION','physical_admission':False,'runtime':0,'accepted_steps':0,'source_cells':len(t),'candidate_cells':len(tt),'source_F32_zero':16,'source_F32_negative':93,'remaining_F32_zero':int(np.sum(dd==0)),'remaining_F32_negative':int(np.sum(dd<0)),'new_cells_F32_nonpositive':int(np.sum(dd[len(t)-len(selected):]<=0)),'accepted':accepted,'attempts':attempts,'original_PLC_facet_restoration':'PENDING_OTHER_CONNECTED_STEINER_CAVITIES','exact_owner_sha256':hashlib.sha256(owner.read_bytes()).hexdigest()}
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in('attempts',)}))
