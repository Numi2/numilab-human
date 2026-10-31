#!/usr/bin/env python3
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706";S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
G=R/"attempt-005/lp-input-current406.npz";M=R/"attempt-022-source-exhaustion-reserved-batch/run-001/merged-held-centroid-noncentroid-source-exhaustion.npz"
T=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
OUT=R/"attempt-025-fulltet-sites-for-held-exchange/run-002";OUT.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def project(x,V):
 n=len(V);best=np.full(n,np.inf);out=np.zeros((n,4))
 for mask in range(1,16):
  ids=np.array([k for k in range(4) if mask&(1<<k)],dtype=np.int64);W=V[:,ids,:]
  if len(ids)==1: small=np.ones((n,1));valid=np.ones(n,bool);proj=W[:,0,:]
  else:
   D=W[:,1:,:]-W[:,:1,:];G=np.einsum("nki,nli->nkl",D,D);rhs=np.einsum("nki,ni->nk",D,x[None,:]-W[:,0,:])
   try:a=np.linalg.solve(G,rhs[...,None])[...,0]
   except np.linalg.LinAlgError:a=np.einsum("nkl,nl->nk",np.linalg.pinv(G),rhs)
   small=np.column_stack((1-a.sum(1),a));valid=np.isfinite(small).all(1)&(small.min(1)>=-1e-12)&(small.max(1)<=1+1e-12)
   small=np.maximum(small,0);ss=small.sum(1);valid&=ss>0;small[valid]/=ss[valid,None];proj=np.einsum("nm,nmi->ni",small,W)
  d2=np.einsum("ni,ni->n",proj-x,proj-x);take=valid&(d2<best)
  if take.any():best[take]=d2[take];out[take]=0;out[np.ix_(take,ids)]=small[take]
 return np.sqrt(best),out
with np.load(G,allow_pickle=False) as z:
 gp=z["parent_role_rows"].astype(np.int32);gc=z["child_rows"].astype(np.int32);gb=z["barycentric"].astype(np.float64);gd=z["distance_m"].astype(np.float64);cv=z["child_volume_hybrid_m3"].astype(np.float64)
with np.load(M,allow_pickle=False) as z:
 cm=z["child_current_mass_kg"].astype(np.float64);pm=z["parent_source_mass_kg"].astype(np.float64);q=z["parent_corner_residual_after_kg"].astype(np.float64)
need=np.maximum(850*cv-cm,0);fullq=pm[:,None]*.25;fullcap_sum=np.zeros(len(cv))
for lo in range(0,len(gp),200000):
 hi=min(len(gp),lo+200000);p=gp[lo:hi];c=gc[lo:hi];b=gb[lo:hi];r=np.full_like(b,np.inf);np.divide(fullq[p],b,out=r,where=b>0);cap=r.min(1)
 good=np.isfinite(b).all(1)&(b>=0).all(1)&(np.abs(b.sum(1)-1)<=2e-14)&(gd[lo:hi]<=.019353)&(cap>0)
 f=np.minimum(cap,950*cv[c]);f[~good]=0;np.add.at(fullcap_sum,c,f)
uncov=np.flatnonzero((need>0)&(np.minimum(fullcap_sum,950*cv)<need))
with np.load(T,allow_pickle=False) as z:
 pts=z["points_world_m"].astype(np.float64);tet=z["tetrahedra"].astype(np.int64);mat=z["material_indices"].astype(np.int32)
cent=np.mean(pts[tet],axis=1)
with np.load(SUP,allow_pickle=False) as z:off=z["cell_admissible_region_offsets"];rid=z["cell_admissible_region_ids"]
pr=rows(PARENT);preg=np.array([int(x["region"]) for x in pr],np.int32);pmat=np.array([int(x["material"]) for x in pr],np.int32);pglob=np.array([int(x["global_source330_row"]) for x in pr],np.int64);pmas=np.array([float(x["mass_kg"]) for x in pr])
nrows=rows(SRC/"compiled-parent-cell-mass.csv.nodes.csv");xyz=np.array([[float(x["rest_"+a+"_m"]) for a in "xyz"] for x in nrows])
tr=rows(SRC/"compiled-parent-cell-mass.csv.tets.csv");tg=np.array([int(x["parent_tet_row"]) for x in tr],np.int64);tn=np.array([[int(x[f"parent_node{k}"]) for k in range(4)] for x in tr],np.int64)
pmap={int(g):i for i,g in enumerate(pglob)}; keep=np.flatnonzero(np.isin(tg,pglob)); tpr=np.array([pmap[int(tg[i])] for i in keep],np.int32); tv=xyz[tn[keep]]
res=[]
for c in uncov:
 regs=np.asarray(rid[off[c]:off[c+1]],np.int32);mask=(pmat[tpr]==mat[c])&np.isin(preg[tpr],regs);ix=np.flatnonzero(mask)
 if not len(ix):res.append({"child_row":int(c),"need_kg":float(need[c]),"allowed_regions":regs.tolist(),"source_tets":0,"within_R":0,"capacity_kg":0,"nearest_m":None,"top":[]});continue
 d,b=project(cent[c],tv[ix]);ok=d<=.019353;ii=ix[ok];dd=d[ok];bb=b[ok];rr=np.full_like(bb,np.inf);np.divide(fullq[tpr[ii]],bb,out=rr,where=bb>0);caps=rr.min(1);ok=np.isfinite(caps)&(caps>0);ii=ii[ok];dd=dd[ok];bb=bb[ok];caps=caps[ok];order=np.argsort(dd)
 top=[]
 for j in order[:12]:
  role=int(tpr[ii[j]])
  top.append({"role_parent_row":role,"global_source330_row":int(pglob[role]),"region":int(preg[role]),"distance_m":float(dd[j]),"barycentric":bb[j].tolist(),"full_M4_single_site_cap_kg":float(caps[j])})
 res.append({"child_row":int(c),"need_kg":float(need[c]),"allowed_regions":regs.tolist(),"source_tets":int(len(ix)),"within_R":int(len(ii)),"capacity_kg":float(caps.sum()),"nearest_m":float(dd.min()) if len(dd) else None,"top":top})
inputs=[G,M,T,SUP,PARENT,SRC/"compiled-parent-cell-mass.csv.nodes.csv",SRC/"compiled-parent-cell-mass.csv.tets.csv"]
report={"schema":"numi.human.current406-held-exchange-fulltet-site-diagnostic.v1","status":"SOURCE_ONLY_FULLTET_SITE_SUPPORT_DIAGNOSTIC","physical_admission":False,"native_steps":0,"complete_allocation":False,"scope":"For receivers below the existing graph's optimistic all-held-release bound, project the widened current child centroid onto every same-material source330 tetrahedron in that child's authenticated 143 region CSR, then enforce unchanged 19.353mm and M/4 single-site caps. These independent receiver bounds omit shared source and child coupling; not an allocation or proof of infeasibility.","inputs":{str(p):sha(p) for p in inputs},"counts":{"fulltet_uncovered_children":len(res),"with_positive_fulltet_radius_support":sum(x["within_R"]>0 for x in res),"all_underfloor_children":int(np.count_nonzero(need>0))},"mass_kg":{"all_remaining_source_corners":float(q.sum()),"all_lower_deficit":float(need.sum()),"witness_child_deficits":float(sum(x["need_kg"] for x in res))},"hard_constraints":{"radius_m":.019353,"same_material":True,"authenticated_region_support":str(SUP),"corner_cap":"min((parent source mass/4)/barycentric corner)"},"witnesses":res,"elapsed_seconds":time.monotonic()-t0}
rp=OUT/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"counts":report["counts"],"mass_kg":report["mass_kg"],"witnesses":[{**{k:v for k,v in x.items() if k!="top"},"top":x["top"][:3]} for x in res],"elapsed_seconds":report["elapsed_seconds"],"report":str(rp),"sha256":sha(rp)},indent=2))

