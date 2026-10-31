"""Fixed-coordinate, material-local 2-to-3 cavities. Proposals only; no native admission."""
from pathlib import Path
import numpy as np,hashlib,json,time,argparse
from collections import Counter
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)
def sha(p):
 return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def measurements(p,t):
 q=p[t].astype(np.float64);u=q[:,1:]-q[:,:1]
 det=np.einsum('ij,ij->i',u[:,0],np.cross(u[:,1],u[:,2]))
 maxedge=np.max(np.stack([np.linalg.norm(q[:,i]-q[:,j],axis=1) for i in range(4) for j in range(i+1,4)]),axis=0)
 return det,det/np.maximum(maxedge**3,np.finfo(float).tiny)
def passes(det,nd):
 return np.isfinite(det)&np.isfinite(nd)&(det/6>1e-18)&(nd>=1e-5)
def orient(p,t):
 t=np.asarray(t,np.int64).copy()
 det,_=measurements(p,t)
 for r in np.flatnonzero(det<0):t[r,0],t[r,1]=t[r,1],t[r,0]
 return t
def cyc(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def chain(t):
 counts=Counter()
 for f in t[:,PAT].reshape(-1,3):
  a,b,c=map(int,f);s=tuple(sorted((a,b,c)))
  parity=1 if ((a>b)+(a>c)+(b>c))%2==0 else -1
  counts[s]+=parity
 return {k:v for k,v in counts.items() if v}
def integer_points(p):
 # Float32 has an exact integer representation on the 2^-149 lattice.
 a=np.asarray(p,np.float32)
 out=[]
 for row in a:
  r=[]
  for x in row:
   n,d=float(x).as_integer_ratio();r.append(n<<(149-(d.bit_length()-1)))
  out.append(tuple(r))
 return out
def exact_det(p,t):
 a,b,c,d=[p[int(i)] for i in t];u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];w=[d[i]-a[i] for i in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
def certify(p,old,new):
 # The two triangulations must have exactly the same oriented cavity boundary.
 # Positive determinants and no external occurrence of the new edge are checked by caller.
 if chain(old)!=chain(new):return False,'oriented_boundary'
 used=np.unique(np.r_[old.ravel(),new.ravel()]);look={int(v):i for i,v in enumerate(used)}
 ip=integer_points(p[used])
 dsold=[exact_det(ip,[look[int(v)] for v in tet]) for tet in old]
 dsnew=[exact_det(ip,[look[int(v)] for v in tet]) for tet in new]
 if min(dsold+dsnew)<=0:return False,'nonpositive_exact'
 if sum(dsold)!=sum(dsnew):return False,'exact_volume'
 return True,'PASS'
def propose_pair(p,ta,tb):
 shared=sorted(set(map(int,ta))&set(map(int,tb)))
 if len(shared)!=3:return None
 aa=list(set(map(int,ta))-set(shared));bb=list(set(map(int,tb))-set(shared))
 if len(aa)!=1 or len(bb)!=1 or aa[0]==bb[0]:return None
 a,b,c=shared;d,e=aa[0],bb[0]
 return orient(p,[[d,e,a,b],[d,e,b,c],[d,e,c,a]]),(d,e)
def fixture():
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1e-7],[.2,.2,-1]],np.float32)
 old=orient(p,[[0,1,2,3],[0,2,1,4]])
 new,edge=propose_pair(p,*old)
 assert sum(~passes(*measurements(p,old)))==1
 assert np.all(passes(*measurements(p,new)))
 assert certify(p,old,new)==(True,'PASS')
 assert certify(p,old,new[:2])[0] is False
 bad=new.copy();bad[0,[0,1]]=bad[0,[1,0]]
 assert certify(p,old,bad)[0] is False
 assert propose_pair(p,old[0],old[0]) is None
 from tempfile import TemporaryDirectory
 from types import SimpleNamespace
 with TemporaryDirectory(prefix='numi-face-flip-npz-control-') as td:
  td=Path(td);np.savez_compressed(td/'input.npz',points_world_m=p,tetrahedra=old,material_indices=np.full(len(old),15,np.int32))
  run(SimpleNamespace(input_npz=td/'input.npz',output=td/'output',seconds=30,max_operations=20))
  rr=json.loads((td/'output/report.json').read_text())
  assert rr['passing_operations']==1 and rr['bad_cells_after']==0 and rr['output_cells']==3
  with np.load(td/'output/fixed-boundary-volume-proposal.npz') as z:assert np.array_equal(z['points_world_m'],p)
 return {'status':'PASS_FIXED_BOUNDARY_2_TO_3_NPZ_DRIVER_CONTROLS','controls':['quality_improvement','oriented_cavity_boundary','exact_F32_volume','negative_orientation','invalid_cavity'],'native_steps':0,'physical_admission':False}
def run(args):
 started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=False)
 pin={str(v):sha(v) for v in [args.input_npz,Path(__file__)]}
 with np.load(args.input_npz) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'];mat=v['material_indices']
 det,quality=measurements(p,t);good=passes(det,quality)
 if np.any(det<=0):raise ValueError('baseline must be positive')
 faces=np.sort(t[:,PAT].reshape(-1,3),axis=1)
 # Stable deterministic face key order; only two-cell faces can be flipped.
 order=np.lexsort((faces[:,2],faces[:,1],faces[:,0]))
 sf=faces[order];starts=np.r_[0,1+np.flatnonzero(np.any(sf[1:]!=sf[:-1],axis=1))]
 counts=np.diff(np.r_[starts,len(sf)])
 if np.any(counts>2):raise ValueError('nonmanifold baseline')
 pair_starts=starts[counts==2];pairs=order[np.c_[pair_starts,pair_starts+1]]//4
 pairs=pairs[(~good[pairs[:,0]]|~good[pairs[:,1]])&(mat[pairs[:,0]]==mat[pairs[:,1]])]
 flat=t.ravel();corner_order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 del faces,order,sf,starts,counts,pair_starts,flat
 used_nodes=set();remove=[];newrows=[];newmats=[];ops=[];reject=Counter()
 for ia,ib in pairs:
  if time.monotonic()-started>args.seconds or len(ops)>=args.max_operations:break
  old=t[[ia,ib]];nodes=set(map(int,old.ravel()))
  if nodes&used_nodes:reject['conflict']+=1;continue
  proposed=propose_pair(p,*old)
  if proposed is None:reject['not_five_vertices']+=1;continue
  new,(d,e)=proposed
  incident_d=corner_order[offset[d]:offset[d+1]]//4;incident_e=corner_order[offset[e]:offset[e+1]]//4
  if len(np.intersect1d(incident_d,incident_e,assume_unique=True)):reject['new_edge_exists']+=1;continue
  dn,qn=measurements(p,new)
  if np.any(dn<=0) or np.sum(~passes(dn,qn))>=np.sum(~good[[ia,ib]]):reject['quality_no_progress']+=1;continue
  ok,why=certify(p,old,new)
  if not ok:reject[why]+=1;continue
  used_nodes.update(nodes);remove.extend([int(ia),int(ib)]);newrows.extend(new.tolist());newmats.extend([int(mat[ia])]*3)
  ops.append({'old_rows':[int(ia),int(ib)],'new_rows_start':len(newrows)-3,'new_edge':[d,e],'before_bad':int(np.sum(~good[[ia,ib]])),'after_bad':int(np.sum(~passes(dn,qn))),'minimum_normalized_det_before':float(quality[[ia,ib]].min()),'minimum_normalized_det_after':float(qn.min())})
 keep=np.ones(len(t),bool);keep[remove]=False
 candidate=np.vstack([t[keep],np.asarray(newrows,np.int64).reshape(-1,4)])
 candidate_mat=np.r_[mat[keep],np.asarray(newmats,np.int32)]
 cd,cq=measurements(p,candidate)
 assert np.sum(~passes(cd,cq))<np.sum(~good) or not ops
 # Disjoint vertex stars make cavity updates independent; unchanged exact boundary follows from per-operation chain proof.
 np.savez_compressed(out/'fixed-boundary-volume-proposal.npz',points_world_m=p,tetrahedra=candidate,material_indices=candidate_mat,retained_input_cell_rows=np.flatnonzero(keep),replacement_parent_cell_rows=np.asarray([o['old_rows'] for o in ops],np.int64).reshape(-1,2),replacement_tetrahedra=np.asarray(newrows,np.int64).reshape(-1,3,4),desired_boundary_faces=v.get('desired_boundary_faces',np.empty((0,3),np.int64)),solver_infeasible_cavity_sha256=v.get('solver_infeasible_cavity_sha256',np.empty((0,32),np.uint8)))
 report={'status':'PASS_FIXED_BOUNDARY_VOLUME_CONNECTIVITY_PROPOSAL' if ops else 'NO_VALID_CONNECTIVITY_PROGRESS','physical_admission':False,'native_steps':0,'input_pins':pin,'node_coordinates_unchanged':True,'oriented_boundary_unchanged_exact':True,'exact_F32_volume_unchanged':True,'material_ownership_preserved':True,'tested_pair_count':int(sum(reject.values())+len(ops)),'total_eligible_pairs':len(pairs),'passing_operations':len(ops),'rejections':dict(reject),'input_cells':len(t),'output_cells':len(candidate),'bad_cells_before':int(sum(~good)),'bad_cells_after':int(sum(~passes(cd,cq))),'minimum_volume_before_m3':float(det.min()/6),'minimum_volume_after_m3':float(cd.min()/6),'minimum_normalized_det_before':float(quality.min()),'minimum_normalized_det_after':float(cq.min()),'elapsed_s':time.monotonic()-started,'operations':ops}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='operations'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input-npz',type=Path);ap.add_argument('--node',type=Path);ap.add_argument('--ele',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--max-operations',type=int,default=10000);args=ap.parse_args()
 if args.fixture:print(json.dumps(fixture()))
 else:run(args)
