#!/usr/bin/env python3
"""Bounded star-kernel cavity construction on existing fat bad-cell patches.
Diagnostic patch proposals only: no boundary ancestry/mass/native admission.
"""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.optimize import linprog
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
R=H/'tetgen-scale-relative-tolerance-059/fat009-run001'
C=H/'tetgen-scale-relative-tolerance-059/fat009-local-patch-classification-001/fat009-run001/bad-cell-one-ring-patches.npz'
O=H/'fat009-star-kernel-pilot-062';O.mkdir(exist_ok=False)
pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]],np.int64)
nodes=np.loadtxt(R/'fat.1.node',skiprows=1);ele=np.loadtxt(R/'fat.1.ele',skiprows=1,dtype=np.int64);p=nodes[:,1:4].astype(np.float32);t=ele[:,1:5]-1
assert np.array_equal(nodes[:,0],np.arange(1,len(p)+1))
with np.load(C)as z:meta={k:z[k]for k in z.files}
def exact_det(q):
 ratios=[[float(v).as_integer_ratio()for v in row]for row in q];e=max(d.bit_length()-1 for row in ratios for n,d in row)
 x=[[n<<(e-(d.bit_length()-1))for n,d in row]for row in ratios];u,v,w=[[x[j][k]-x[0][k]for k in range(3)]for j in (1,2,3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]),e*3
def exact_sum(tets,points):
 vals=[exact_det(points[x])for x in tets];e=max(k for d,k in vals);return sum(d<<(e-k)for d,k in vals),e
reports=[];accepted=[];points=[];conns=[];replaced=[];po=[0];to=[0]
for k in range(len(meta['patch_cell_offsets'])-1):
 tic=time.monotonic();ids=meta['patch_cell_rows_0based'][meta['patch_cell_offsets'][k]:meta['patch_cell_offsets'][k+1]]
 f=t[ids][:,pat].reshape(-1,3);keys,first,n=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True);b=f[first[n==1]]
 rr={'patch':k,'cells':len(ids),'boundary_faces':len(b)}
 edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1)
 _,ec=np.unique(edges,axis=0,return_counts=True)
 q=p[b].astype(np.float64);norm=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(norm,axis=1)
 if np.any(ec!=2)or np.any(area==0)or np.any(n>2):rr['status']='REJECT_BOUNDARY_TOPOLOGY_OR_AREA';reports.append(rr);continue
 norm/=area[:,None];v=p[np.unique(b)].astype(np.float64);origin=(v.max(0)+v.min(0))*.5;scale=float(np.max(v.max(0)-v.min(0)))
 a=np.column_stack([norm,np.ones(len(b))]);ub=np.einsum('ij,ij->i',norm,(q[:,0]-origin)/scale)
 sol=linprog([0,0,0,-1],A_ub=a,b_ub=ub,bounds=[(-2,2)]*3+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
 rr.update(lp_status=int(sol.status),lp_message=sol.message)
 if not sol.success or sol.x[3]<=0:rr['status']='REJECT_NO_STRICT_STAR_KERNEL';reports.append(rr);continue
 c=(origin+sol.x[:3]*scale).astype(np.float32)
 candidates=[c]
 axes=[[np.nextafter(c[j],np.float32(-np.inf)),c[j],np.nextafter(c[j],np.float32(np.inf))]for j in range(3)]
 import itertools
 candidates.extend(np.asarray(x,np.float32)for x in itertools.product(*axes))
 best=None;bestmargin=-np.inf
 for center in candidates:
  dets=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,0]-center.astype(np.float64))
  margin=float(np.min(dets/area))
  if margin>bestmargin:bestmargin=margin;best=center.copy()
 rr.update(kernel_inradius_m=float(sol.x[3]*scale),best_F32_margin_m=bestmargin,center=best.tolist())
 if bestmargin<=0:rr['status']='REJECT_NO_F32_INTERIOR_POINT';reports.append(rr);continue
 # Fan orientation is (center, oriented face), positive for outward-oriented faces.
 tmp=np.vstack([p,best]);fan=np.column_stack([np.full(len(b),len(p)),b])
 exacts=[exact_det(tmp[x])[0]for x in fan]
 if any(x<=0 for x in exacts):rr['status']='REJECT_EXACT_NONPOSITIVE_FAN';reports.append(rr);continue
 old,eo=exact_sum(t[ids],p);new,en=exact_sum(fan,tmp);e=max(eo,en);equal=(old<<(e-eo))==(new<<(e-en))
 rr['exact_signed_cavity_volume_conserved']=equal
 if not equal:rr['status']='REJECT_VOLUME';reports.append(rr);continue
 # Surface self/intersections and restored PLC identity are mandatory downstream.
 rr['status']='PASS_STAR_KERNEL_CONSTRUCTION_PENDING_EXACT_SURFACE_GATES'
 accepted.append(k);points.append(best);conns.append(np.column_stack([np.full(len(b),len(p)+len(points)-1),b]));replaced.extend(ids.tolist());to.append(to[-1]+len(b));po.append(po[-1]+len(ids))
 rr['wall_s']=time.monotonic()-tic;reports.append(rr)
np.savez_compressed(O/'patch-proposals.npz',new_interior_points_world_m=np.asarray(points,np.float32).reshape(-1,3),new_tetrahedra_global_rows=np.concatenate(conns)if conns else np.empty((0,4),np.int64),new_tet_offsets=np.array(to,np.int64),replaced_cell_rows=np.array(replaced,np.int64),replaced_cell_offsets=np.array(po,np.int64),accepted_patch_ids=np.array(accepted,np.int64))
report={'status':'BOUNDED_STAR_KERNEL_CONSTRUCTION_PROPOSALS','physical_admission':False,'runtime':0,'accepted_steps':0,'accepted_constructions':len(accepted),'patches':len(reports),'replaced_cells':len(replaced),'new_cells':sum(len(x)for x in conns),'remaining_required_gates':['exact boundary self/nonadjacent intersections','restore original PLC facets including boundary Steiner removal','full consistent F32 mesh','original labels and source330 mass'],'patches_detail':reports}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='patches_detail'}))
