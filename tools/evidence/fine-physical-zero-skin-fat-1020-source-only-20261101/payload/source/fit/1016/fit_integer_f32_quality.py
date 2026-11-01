#!/usr/bin/env python3
"""Compact integer-F32-lattice quality construction. No geometry admission."""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
SRC=H/"integer-f32-quality-1016-root/local-initial-bundle.npz"
EXPECT="8df1a74c1d9a8f9a6c24e0abf676978a707d9eb9cf6f1cbb89e3d14c81df6517"
OUT=H/"integer-f32-quality-1016-root"
RESUME=H/"integer-f32-quality-1013-root/actual-001/report.json"
RESUME_SHA="53aa638b416995e20aaae49b171628d374ff377eb17328674fecf7b6187ff503"
Q=1e-5;V6=6e-18
EDGES=[(i,j)for i in range(4)for j in range(i+1,4)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measures(p,t):
 x=p[t].astype(float);a,b,c,d=[x[:,i]for i in range(4)]
 ds=np.einsum("ij,ij->i",b-a,np.cross(c-a,d-a))
 lens=np.stack([np.linalg.norm(x[:,i]-x[:,j],axis=1)for i,j in EDGES],axis=1)
 return ds,ds/lens.max(axis=1)**3
def dgrad(p):
 a,b,c,d=p
 return np.array([-np.cross(c-b,d-b),np.cross(c-a,d-a),np.cross(d-a,b-a),np.cross(b-a,c-a)])
def jsondefault(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def main():
 assert sha(SRC)==EXPECT
 OUT.mkdir(exist_ok=True);dest=OUT/"actual-001";dest.mkdir(exist_ok=False)
 started=time.monotonic();deadline=started+120
 z=np.load(SRC,allow_pickle=False);variables=json.loads(z["variable_records_json"].item())
 base=np.array([v["base"]for v in variables],np.float32)
 cur=base.copy()
 step=np.spacing(np.abs(base)).astype(float);assert np.all(step>0)
 n=base.size;maps={};rd={};bi={};baseq={}
 for role in ("Skin","Fat"):
  rows=z[role+"_local_point_rows"];lookup={int(r):i for i,r in enumerate(rows)}
  t=np.array([[lookup[int(r)]for r in tt]for tt in z[role+"_local_tets_global_rows"]])
  p=z[role+"_initial_points_f32"].copy()
  mp={}
  for i,v in enumerate(variables):
   row=v["role_rows"][role] if v["kind"]=="I_surface" else v["volume_row"] if v["role"]==role else None
   if row is not None:
    k=lookup[int(row)];assert np.array_equal(p[k],base[i]);mp[k]=i
  d,q=measures(p,t);baseq[role]=q.copy()
  for k in np.flatnonzero(q<Q):bi[(role,int(k))]=len(bi)
  maps[role]=mp;rd[role]={"points":p,"tets":t,"cells":z[role+"_cell_rows"].copy()}
 ip=z["I_local_points_f32"].copy();iff=z["I_local_faces"];ipr=z["I_local_point_rows"]
 ilook={int(row):i for i,row in enumerate(ipr)}
 ivars={ilook[v["source_row"]]:i for i,v in enumerate(variables) if v["kind"]=="I_surface"}
 tx=ip[iff].astype(float);n0=np.cross(tx[:,1]-tx[:,0],tx[:,2]-tx[:,0]);a0=np.linalg.norm(n0,axis=1)
 assert np.all(a0>0)
 def source_data(coords):
  pp=ip.copy()
  for row,i in ivars.items():pp[row]=coords[i]
  tx=pp[iff].astype(float);normals=np.cross(tx[:,1]-tx[:,0],tx[:,2]-tx[:,0]);area=np.linalg.norm(normals,axis=1)
  ratios=area/a0;cos=np.einsum("ij,ij->i",normals,n0)/(np.maximum(area,1e-300)*a0)
  good=bool(np.all(ratios>=.5)&np.all(ratios<=2)&np.all(cos>=.5))
  return pp,normals,area,ratios,cos,good
 ns=len(bi);history=[];initial=sum(np.maximum(0,1-q/Q).sum()for q in baseq.values());best=initial
 def render(coords):
  pts={}
  for role,v in rd.items():
   p=v["points"].copy()
   for row,i in maps[role].items():p[row]=coords[i]
   pts[role]=p
  return pts
 def metric(coords):
  pts=render(coords);ans={};merit=0;positive=True;bad_deficits=[]
  for role,v in rd.items():
   d,q=measures(pts[role],v["tets"])
   bad_deficits.extend(np.maximum(0,1-q[baseq[role]<Q]/Q).tolist())
   merit+=10*np.maximum(0,1-q[baseq[role]>=Q]/Q).sum()
   positive &= bool(np.all(d>=V6))
   ans[role]={"bad":int((q<Q).sum()),"qmin":float(q.min()),"det6min":float(d.min()),"oldgood_regressions":int(((baseq[role]>=Q)&(q<Q)).sum())}
  return float(merit+max(bad_deficits,default=0.)),bool(positive and source_data(coords)[-1]),ans
 initial=metric(base)[0];best=metric(cur)[0]
 radius=128
 for iteration in range(160):
  if time.monotonic()>deadline:break
  pts=render(cur);ridx=[];cidx=[];vals=[];rhs=[];r=0
  bi={(role,int(k)):0 for role in rd for k in np.flatnonzero(baseq[role]<Q)}
  ns=1
  for role,v in rd.items():
   p=pts[role];t=v["tets"];ds,qs=measures(p,t)
   for k,tt in enumerate(t):
    pp=p[tt].astype(float);gd=dgrad(pp);d=float(ds[k]);bad=(role,k)in bi
    for a,b in EDGES:
     edge=pp[a]-pp[b];L=np.linalg.norm(edge);g=gd/L**3;eg=3*d*edge/L**5;g[a]-=eg;g[b]+=eg
     coeff={}
     for slot,row in enumerate(tt):
      vi=maps[role].get(int(row))
      if vi is None:continue
      for axis in range(3):
       j=3*vi+axis;coeff[j]=coeff.get(j,0.)-g[slot,axis]*step[vi,axis]/Q
     if bad:coeff[n+bi[(role,k)]]=-1.
     bval=d/L**3/Q-1;scale=max(1.,abs(bval),max(map(abs,coeff.values()),default=0.))
     for j,aij in coeff.items():ridx.append(r);cidx.append(j);vals.append(aij/scale)
     rhs.append(bval/scale);r+=1
    coeff={}
    for slot,row in enumerate(tt):
     vi=maps[role].get(int(row))
     if vi is None:continue
     for axis in range(3):
      j=3*vi+axis;coeff[j]=coeff.get(j,0.)-gd[slot,axis]*step[vi,axis]
    scale=max(abs(d),max(map(abs,coeff.values()),default=0.),V6)
    for j,aij in coeff.items():ridx.append(r);cidx.append(j);vals.append(aij/scale)
    rhs.append((d-V6)/scale);r+=1
  # Linearized source triangle area and orientation constraints, independently screened after every integer trial.
  sip,snorm,sarea,sratio,scos,sourceok=source_data(cur)
  assert sourceok
  for fr,face in enumerate(iff):
   ptri=sip[face].astype(float);aa,bb,cc=ptri
   unit=snorm[fr]/sarea[fr]
   gc=n0[fr]/(sarea[fr]*a0[fr])-scos[fr]*snorm[fr]/sarea[fr]**2
   area_terms={};cos_terms={}
   for slot,vertex in enumerate(face):
    vi=ivars.get(int(vertex))
    if vi is None:continue
    for axis in range(3):
     e=np.eye(3)[axis]
     dn=(np.cross(-e,cc-aa)+np.cross(bb-aa,-e)) if slot==0 else np.cross(e,cc-aa) if slot==1 else np.cross(bb-aa,e)
     col=3*vi+axis
     area_terms[col]=float(np.dot(unit,dn)*step[vi,axis]/a0[fr])
     cos_terms[col]=float(np.dot(gc,dn)*step[vi,axis])
   for coeffs,sgn,bval in ((area_terms,-1.,sratio[fr]-.5),(area_terms,1.,2.-sratio[fr]),(cos_terms,-1.,scos[fr]-.5)):
    scale=max(1.,abs(bval),max(map(abs,coeffs.values()),default=0.))
    for col,val in coeffs.items():ridx.append(r);cidx.append(col);vals.append(sgn*val/scale)
    rhs.append(float(bval/scale));r+=1
  A=coo_matrix((vals,(ridx,cidx)),shape=(r,n+ns)).tocsr()
  lows=[];highs=[]
  for i,v in enumerate(variables):
   cap=v["cap_m"]/np.sqrt(3)
   for ax in range(3):
    s=step[i,ax];offset=float(cur[i,ax])-float(base[i,ax])
    lo=max(-radius,int(np.ceil((-cap-offset)/s)))
    hi=min(radius,int(np.floor((cap-offset)/s)))
    # A fixed local exponent lattice is mandatory for exact representability.
    for k in (lo,hi):
     val=float(cur[i,ax])+s*k
     assert float(np.float32(val))==val,(i,ax,k,val)
    lows.append(lo);highs.append(hi)
  lows.extend([0.]*ns);highs.extend([np.inf]*ns)
  # Proximal L1 tie-breaker removes arbitrary bound-seeking integer motion.
  A=hstack([A,csr_matrix((r,n))],format="csr")
  ai=[];aj=[];av=[]
  for j in range(n):
   ai.extend((2*j,2*j+1,2*j,2*j+1));aj.extend((j,j,n+ns+j,n+ns+j));av.extend((1.,-1.,-1.,-1.))
  AA=coo_matrix((av,(ai,aj)),shape=(2*n,2*n+ns)).tocsr()
  A=vstack([A,AA],format="csr");rhs=np.r_[rhs,np.zeros(2*n)];r+=2*n
  lows.extend([0.]*n);highs.extend([np.inf]*n)
  cost=np.r_[np.zeros(n),np.ones(ns),np.full(n,1e-7)];integrality=np.r_[np.ones(n),np.zeros(ns+n)]
  res=milp(cost,integrality=integrality,bounds=Bounds(lows,highs),
   constraints=LinearConstraint(A,np.full(r,-np.inf),rhs),
   options={"time_limit":min(12.,max(.1,deadline-time.monotonic())),"mip_rel_gap":1e-5,"presolve":True})
  rec={"iteration":iteration,"solver_status":int(res.status),"message":str(res.message),"objective":float(res.fun)if res.fun is not None else None,"trials":[]}
  if res.x is None:history.append(rec);break
  kvec=np.rint(res.x[:n]).reshape(base.shape).astype(np.int64)
  assert np.max(np.abs(res.x[:n]-kvec.ravel()))<1e-4
  accepted=False
  for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
   kk=np.rint(alpha*kvec).astype(np.int64)
   proposed=cur.astype(float)+step*kk
   candidate=proposed.astype(np.float32)
   assert np.array_equal(proposed,candidate.astype(float))
   merit,positive,stats=metric(candidate)
   trial={"alpha":alpha,"deficit":merit,"positive":positive,"roles":stats,"max_move_m":float(np.linalg.norm(candidate.astype(float)-base,axis=1).max())}
   rec["trials"].append(trial)
   # Internal construction may temporarily redistribute quality errors; no state is admitted here.
   if positive and merit<best-1e-8:
    cur=candidate;best=merit;accepted=True;trial["accepted"]=True;break
  rec["accepted"]=accepted;history.append(rec)
  (dest/"checkpoint.json").write_text(json.dumps({"iteration":iteration,"final_variables_f32":cur,"base_variables_f32":base,"deficit":best,"physical_admission":False,"native_steps":0},default=jsondefault)+"\n")
  print(json.dumps({"iteration":iteration,"accepted":accepted,"deficit":best,"status":rec["solver_status"]}),flush=True)
  if best==0:break
  if accepted and rec["trials"][-1]["alpha"]==1.0:radius=min(256,2*radius)
  if not accepted:
   if radius==1:break
   radius=max(1,radius//2)
 finalmerit,pos,stats=metric(cur)
 output={"status":"LOCAL_ALL_GOOD_PENDING_FULL_REPLAY"if finalmerit==0 and pos else"CONSTRUCTION_INCOMPLETE",
 "input":{"path":str(SRC),"sha256":sha(SRC)},"producer_sha256":sha(__file__),"initial_deficit":initial,"final_deficit":finalmerit,
 "final_roles":stats,"positive":pos,"source_constraints_pass":source_data(cur)[-1],"source_area_ratio_min":float(source_data(cur)[3].min()),"source_area_ratio_max":float(source_data(cur)[3].max()),"source_cosine_min":float(source_data(cur)[4].min()),"history":history,"variable_records":variables,"final_variables_f32":cur,
 "base_variables_f32":base,"source_prefix_positions_unchanged_except_declared_I_updates":True,
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","elapsed_s":time.monotonic()-started}
 (dest/"report.json").write_text(json.dumps(output,indent=2,default=jsondefault)+"\n")
 print(json.dumps({k:output[k]for k in("status","initial_deficit","final_deficit","final_roles","elapsed_s")}),flush=True)
if __name__=="__main__":main()
