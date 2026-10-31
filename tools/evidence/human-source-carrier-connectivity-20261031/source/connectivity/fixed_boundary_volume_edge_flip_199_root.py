"""Fixed-coordinate, material-local 3-to-2 edge cavities. Proposals only; no native admission."""
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
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[.5,1e-7,1],[.5,1e-7,-1]],np.float32)
 old=orient(p,[[3,4,0,1],[3,4,1,2],[3,4,2,0]])
 new=orient(p,[[0,1,2,3],[0,2,1,4]])
 assert sum(~passes(*measurements(p,old)))==1
 assert np.all(passes(*measurements(p,new)))
 assert certify(p,old,new)==(True,'PASS')
 assert certify(p,old,new[:1])[0] is False
 bad=new.copy();bad[0,[0,1]]=bad[0,[1,0]]
 assert certify(p,old,bad)[0] is False
 return {'status':'PASS_FIXED_BOUNDARY_3_TO_2_CONNECTIVITY_FIXTURES','controls':['quality_improvement','oriented_cavity_boundary','exact_F32_volume','negative_orientation'],'native_steps':0,'physical_admission':False}
def run(args):
 started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=False)
 if args.input_npz:
  pin={str(v):sha(v) for v in [args.input_npz,Path(__file__)]}
  with np.load(args.input_npz) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
 else:
  pin={str(v):sha(v) for v in [args.node,args.ele,Path(__file__)]}
  nd=np.loadtxt(args.node,skiprows=1);ids=nd[:,0].astype(np.int64);p=nd[:,1:4].astype(np.float32)
  el=np.loadtxt(args.ele,skiprows=1);t=np.searchsorted(ids,el[:,1:5].astype(np.int64));mat=el[:,5].astype(np.int32)
  if not np.array_equal(ids[t],el[:,1:5].astype(np.int64)):raise ValueError('node identity')
 det,quality=measurements(p,t);good=passes(det,quality)
 if np.any(det<=0):raise ValueError('baseline must be positive')
 edge_pattern=np.asarray(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)),np.int64)
 edges=np.sort(t[:,edge_pattern].reshape(-1,2),axis=1)
 order=np.lexsort((edges[:,1],edges[:,0]))
 se=edges[order];starts=np.r_[0,1+np.flatnonzero(np.any(se[1:]!=se[:-1],axis=1))]
 counts=np.diff(np.r_[starts,len(se)])
 edge_starts=starts[counts==3]
 triples=order[np.c_[edge_starts,edge_starts+1,edge_starts+2]]//6
 chosen_edges=se[edge_starts].copy()
 eligible=(np.any(~good[triples],axis=1))&np.all(mat[triples]==mat[triples[:,0,None]],axis=1)
 triples=triples[eligible];chosen_edges=chosen_edges[eligible]
 flat=t.ravel();corner_order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 del edges,order,se,starts,counts,edge_starts,flat
 used_nodes=set();remove=[];newrows=[];newmats=[];ops=[];reject=Counter()
 for old_rows,edge in zip(triples,chosen_edges):
  if time.monotonic()-started>args.seconds or len(ops)>=args.max_operations:break
  old=t[old_rows];nodes=set(map(int,old.ravel()))
  if nodes&used_nodes:reject['conflict']+=1;continue
  if len(nodes)!=5:reject['not_five_vertices']+=1;continue
  d,e=map(int,edge);base=sorted(nodes-set((d,e)))
  if len(base)!=3:reject['not_three_vertex_link']+=1;continue
  a,b,c=base
  # A triangular link is required, rather than just a degree-three edge.
  if {tuple(sorted(set(map(int,tet))-set((d,e)))) for tet in old}!={tuple(sorted(z)) for z in [(a,b),(b,c),(c,a)]}:reject['not_triangular_link']+=1;continue
  incident=[corner_order[offset[v]:offset[v+1]]//4 for v in base]
  if len(np.intersect1d(np.intersect1d(incident[0],incident[1],assume_unique=True),incident[2],assume_unique=True)):reject['new_face_exists']+=1;continue
  new=orient(p,[[a,b,c,d],[a,c,b,e]])
  dn,qn=measurements(p,new)
  if np.any(dn<=0) or np.sum(~passes(dn,qn))>=np.sum(~good[old_rows]):reject['quality_no_progress']+=1;continue
  ok,why=certify(p,old,new)
  if not ok:reject[why]+=1;continue
  used_nodes.update(nodes);remove.extend(map(int,old_rows));newrows.extend(new.tolist());newmats.extend([int(mat[old_rows[0]])]*2)
  ops.append({'old_rows':old_rows.tolist(),'new_rows_start':len(newrows)-2,'removed_edge':[d,e],'new_face':base,'before_bad':int(np.sum(~good[old_rows])),'after_bad':int(np.sum(~passes(dn,qn))),'minimum_normalized_det_before':float(quality[old_rows].min()),'minimum_normalized_det_after':float(qn.min())})
 keep=np.ones(len(t),bool);keep[remove]=False
 candidate=np.vstack([t[keep],np.asarray(newrows,np.int64).reshape(-1,4)])
 candidate_mat=np.r_[mat[keep],np.asarray(newmats,np.int32)]
 cd,cq=measurements(p,candidate)
 assert np.sum(~passes(cd,cq))<np.sum(~good) or not ops
 # Disjoint vertex stars make cavity updates independent; unchanged exact boundary follows from per-operation chain proof.
 np.savez_compressed(out/'fixed-boundary-volume-proposal.npz',points_world_m=p,tetrahedra=candidate,material_indices=candidate_mat,retained_input_cell_rows=np.flatnonzero(keep),replacement_parent_cell_rows=np.asarray([o['old_rows'] for o in ops],np.int64).reshape(-1,3),replacement_tetrahedra=np.asarray(newrows,np.int64).reshape(-1,2,4))
 report={'status':'PASS_FIXED_BOUNDARY_VOLUME_CONNECTIVITY_PROPOSAL' if ops else 'NO_VALID_CONNECTIVITY_PROGRESS','physical_admission':False,'native_steps':0,'input_pins':pin,'node_coordinates_unchanged':True,'oriented_boundary_unchanged_exact':True,'exact_F32_volume_unchanged':True,'material_ownership_preserved':True,'tested_edge_count':int(sum(reject.values())+len(ops)),'total_eligible_edges':len(triples),'passing_operations':len(ops),'rejections':dict(reject),'input_cells':len(t),'output_cells':len(candidate),'bad_cells_before':int(sum(~good)),'bad_cells_after':int(sum(~passes(cd,cq))),'minimum_volume_before_m3':float(det.min()/6),'minimum_volume_after_m3':float(cd.min()/6),'minimum_normalized_det_before':float(quality.min()),'minimum_normalized_det_after':float(cq.min()),'elapsed_s':time.monotonic()-started,'operations':ops}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='operations'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input-npz',type=Path);ap.add_argument('--node',type=Path);ap.add_argument('--ele',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--max-operations',type=int,default=10000);args=ap.parse_args()
 if args.fixture:print(json.dumps(fixture()))
 else:run(args)
