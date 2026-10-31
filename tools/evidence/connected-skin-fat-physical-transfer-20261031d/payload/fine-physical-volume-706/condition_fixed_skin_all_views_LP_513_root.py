"""Trust-region LP relocation of inferred interior nodes; exact-F32 safeguarded construction.
Same native thresholds, fixed complete surface, source330 mass, material and connectivity.
"""
from pathlib import Path
import argparse,importlib.util,json,hashlib,time,collections
import numpy as np
from scipy.optimize import linprog
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=mod('volume454',H/'transfer_surface_contractions_indexed_379_root.py');M=T.m
R=mod('registration454',H/'interior-quality-conditioning-309-banks/condition_registered_interior_309.py')
def local_points(p,star,node):
 ids,inv=np.unique(star,return_inverse=True)
 return ids,inv.reshape(-1,4),int(np.flatnonzero(ids==node)[0]),p[ids].copy()
def propose(p,star,node,iteration_limit=4):
 try:b=T.boundary(star)
 except ValueError:return None,{'status':'INVALID_ORIENTED_STAR'}
 if np.any(b==node) or not R.star_boundary_closed(b):return None,{'status':'NONCLOSED_INTERIOR_STAR'}
 ids,lt,n,lp=local_points(p,star,node);d,q=M.measurements(lp,lt);oldgood=M.passes(d,q)
 if np.any(d<=0):return None,{'status':'NONPOSITIVE_INPUT'}
 if oldgood.all():return None,{'status':'NO_BAD_CELL'}
 xyz=lp[lt].astype(float);v=xyz[:,1:]-xyz[:,:1]
 gradients=np.stack((-np.cross(v[:,1],v[:,2])-np.cross(v[:,2],v[:,0])-np.cross(v[:,0],v[:,1]),np.cross(v[:,1],v[:,2]),np.cross(v[:,2],v[:,0]),np.cross(v[:,0],v[:,1])),axis=1)
 rr,cc=np.where(lt==n);assert np.array_equal(rr,np.arange(len(star)))
 g=gradients[rr,cc];edges=np.stack([np.linalg.norm(xyz[:,i]-xyz[:,j],axis=1) for i in range(4) for j in range(i+1,4)],axis=1)
 ell=edges.max(1);scale=float(np.median(ell));target=np.maximum(6e-18,1e-5*ell**3);g=g*scale/target[:,None];r0=d/target
 oldsum=sum(R.exact_star_determinants(lp,lt));bad=np.flatnonzero(~oldgood)
 # A min-max group and distinct individual bad rows prevent one immovable cell masking others.
 groups=[bad]+[np.asarray([i]) for i in bad[np.argsort(r0[bad],kind='stable')][:iteration_limit]]
 best=None;reasons=collections.Counter();trials=0
 for radius in (.002,.02,.1):
  if best is not None and best[0][0]==0:break
  for selected in groups:
   # Existing good rows must stay above the original native target; every other
   # bad row stays strictly positive. Actual nonlinear q is checked after F32 emission.
   floor=np.where(oldgood,1.002,np.maximum(r0*.02,1e-12))
   a=np.c_[-g,np.zeros(len(star))];bb=r0-floor
   aa=np.c_[-g[selected],np.ones(len(selected))]
   result=linprog(np.array([0.,0.,0.,-1.]),A_ub=np.vstack([a,aa]),b_ub=np.r_[bb,r0[selected]],bounds=[(-radius,radius)]*3+[(0.,4.)],method='highs',options={'time_limit':.25})
   if result.x is None:reasons['LP_NO_CANDIDATE']+=1;continue
   for back in (1.,.5,.25,.125,.0625):
    trials+=1;trial=lp.copy();trial[n]=np.asarray(lp[n].astype(float)+scale*back*result.x[:3],np.float32)
    if np.array_equal(trial[n].view(np.uint32),lp[n].view(np.uint32)):continue
    nd,nq=M.measurements(trial,lt);ng=M.passes(nd,nq)
    if np.any(nd<=0) or np.any(oldgood&~ng):reasons['F32_SIGN_OR_GOOD_REGRESSION']+=1;continue
    count=int((~ng).sum())
    if count>=int((~oldgood).sum()):reasons['NO_STRICT_COUNT_DESCENT']+=1;continue
    exact=R.exact_star_determinants(trial,lt)
    if any(x<=0 for x in exact) or sum(exact)!=oldsum:reasons['EXACT_SIGN_OR_STAR_VOLUME']+=1;continue
    displacement=float(np.linalg.norm(trial[n].astype(float)-lp[n].astype(float)))
    if displacement>.2*scale:reasons['DISPLACEMENT']+=1;continue
    value=(count,-float(np.minimum(nq/1e-5,nd/6e-18).min()),displacement)
    if best is None or value<best[0]:
     best=(value,trial[n].copy(),{'old_bad':int((~oldgood).sum()),'new_bad':count,'displacement_m':displacement,'maximum_displacement_m':.2*scale,'minimum_q':float(nq.min()),'minimum_volume_m3':float(nd.min()/6),'exact_det6_sum':str(oldsum),'new_good_regressions':0})
    break
 return (None,{'status':'NO_SAFE_LP_RELOCATION','trials':trials,'rejections':dict(reasons)}) if best is None else ((best[1],best[2]),{'status':'PASS','trials':trials,'rejections':dict(reasons)})
def fixture():
 p=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01],[1e-8,1e-8,1e-8]],np.float32)
 outer=np.array([[0,1,2,3]],np.int64);t=M.orient(p,np.c_[np.full(4,4),T.boundary(outer)])
 ans,diag=propose(p,t,4);assert ans is not None,diag
 point,op=ans;out=p.copy();out[4]=point
 d,q=M.measurements(out,t);assert np.all(M.passes(d,q)) and op['new_good_regressions']==0
 assert sum(R.exact_star_determinants(p,t))==sum(R.exact_star_determinants(out,t))
 assert propose(p,t,0)[0] is None
 inv=t.copy();inv[0,[0,1]]=inv[0,[1,0]]
 assert propose(p,inv,4)[0] is None
 return {'status':'PASS_EXACT_INTERIOR_LP_CONTROLS','controls':['actual_LP_repairs_bad_star','good_cells_preserved','same_exact_volume','boundary_node_rejected','invalid_orientation_rejected'],'physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__)}
def run(a):
 start=time.monotonic();a.output.mkdir(exist_ok=False,parents=True);assert sha(a.input)==a.input_sha256
 with np.load(a.input) as z:base={k:z[k] for k in z.files}
 p=base['points_world_m'].copy();original=p.copy();t=base['tetrahedra'];mat=base['material_indices'];d,q=M.measurements(p,t);initial=M.passes(d,q);assert np.all(d>0)
 source=H/'combined-child-handoff-142/skin.node';registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
 sp=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32);reg,counts=R.registered_source_rows_from_map(registration,sp)
 assert np.array_equal(p[:len(sp)][reg].view(np.uint32),sp[reg].view(np.uint32))
 boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True;fixed[:len(sp)]|=reg
 flat=t.ravel();order=np.argsort(flat,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
 hits=np.bincount(t[~initial].ravel(),minlength=len(p));nodes=np.flatnonzero((hits>0)&~fixed);nodes=nodes[np.lexsort((nodes,-hits[nodes]))]
 used=set();ops=[];reasons=collections.Counter();tested=0
 for node in nodes:
  if time.monotonic()-start>a.seconds or len(ops)>=a.maximum_operations:break
  rows=np.unique(order[offset[node]:offset[node+1]]//4)
  if used&set(map(int,rows)):reasons['OVERLAPPING_CELLS']+=1;continue
  if len(np.unique(mat[rows]))!=1:reasons['MATERIAL_INTERFACE']+=1;continue
  ans,diag=propose(p,t[rows],int(node));tested+=1
  if ans is None:reasons[diag['status']]+=1;continue
  point,op=ans;old=p[node].copy();p[node]=point
  op.update(node=int(node),old_f32=old.tolist(),new_f32=point.tolist(),cell_rows=rows.tolist(),diagnostics=diag);ops.append(op);used.update(map(int,rows))
 nd,nq=M.measurements(p,t);ng=M.passes(nd,nq)
 assert np.all(nd>0) and not np.any(initial&~ng) and int((~ng).sum())==int((~initial).sum())-sum(x['old_bad']-x['new_bad'] for x in ops)
 assert np.array_equal(p[fixed].view(np.uint32),original[fixed].view(np.uint32))
 base['points_world_m']=p
 if 'points_world_m_f32' in base:base['points_world_m_f32']=p.copy()
 npz=a.output/'conditioned-volume-proposal.npz';np.savez_compressed(npz,**base)
 r={'status':'PASS_INTERIOR_LP_FIXED_BOUNDARY_DESCENT' if ops else 'NO_SAFE_INTERIOR_LP_PROGRESS','physical_admission':False,'native_steps':0,'input':str(a.input),'input_sha256':sha(a.input),'candidate':str(npz),'candidate_sha256':sha(npz),'bad_before':int((~initial).sum()),'bad_after':int((~ng).sum()),'operations':ops,'accepted_operations':len(ops),'tested_nodes':tested,'eligible_nodes':len(nodes),'rejections':dict(reasons),'native_thresholds_unchanged':True,'connectivity_material_surface_rows_unchanged':True,'registration_map':str(registration),'registration_map_sha256':sha(registration),'source330_mass':'IMMUTABLE_REBIND_REQUIRED','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
 (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='operations'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input',type=Path);ap.add_argument('--input-sha256');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=45);ap.add_argument('--maximum-operations',type=int,default=10000);a=ap.parse_args()
 if a.fixture:
  a.output.mkdir(exist_ok=False,parents=True);r=fixture();(a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 else:run(a)
