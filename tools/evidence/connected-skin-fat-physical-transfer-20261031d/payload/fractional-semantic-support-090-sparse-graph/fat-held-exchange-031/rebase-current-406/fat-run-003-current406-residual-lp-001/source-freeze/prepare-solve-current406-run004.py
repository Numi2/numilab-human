#!/usr/bin/env python3
"""Bounded current-406 Fat residual source-site LP; source-only candidate, no admission."""
import csv,hashlib,json,importlib.util,time,os,resource,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

T0=time.monotonic()
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"; S=ROOT/"fractional-semantic-support-090-sparse-graph"
OUT=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-002"
CAND=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
CREPORT=CAND.parent/"report.json"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUREPORT=SUP.parent/"independent-verification.json"
BASE=H/"combined-child-handoff-142"
LEDGER=S/"fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/rebased-held-flow-ledger.npz"
LREPORT=LEDGER.parent/"ledger-report.json"
LVERIFY=LEDGER.parent/"independent-ledger-replay-v3.json"
EXACT=LEDGER.parent/"exact-map.npz"; ROWMAP=LEDGER.parent/"cell-row-map.npz"
CHILDCSV=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-child-mass.csv"
PARENTCSV=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
PAIR=Path("/tmp/fat_candidate_pairs.npz")
PAIRPINS=S/"fat-residual-source-site-lp-001/run-011/preflight-input-pins.json"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
SOURCE_NODES=SRC/"compiled-parent-cell-mass.csv.nodes.csv"
SOURCE_TETS=SRC/"compiled-parent-cell-mass.csv.tets.csv"
SOURCE_LABELS=SRC/"compiled-parent-cell-mass.csv.labels.csv"
VOLUME_AUDIT=LEDGER.parent/"hybrid-volume-audit-001/report.json"
BASE_PARTIAL=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008"

R=0.019353; LOW=850.0; HIGH=950.0; MAX_COLUMNS=12_000_000
PARENT_K=32; CORNER_K=8; RECEIVER_K=2
def say(msg):
 print(f"{time.monotonic()-T0:8.2f}s {msg}",flush=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def writej(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def peakrss():
 r=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 return r if sys.platform=="darwin" else r*1024
def readcsv(path):
 with open(path,newline="") as f:return list(csv.DictReader(f))
def detvol(points,tets):
 vv=points[tets].astype(np.float64)
 d=np.einsum("ij,ij->i",np.cross(vv[:,1]-vv[:,0],vv[:,2]-vv[:,0]),vv[:,3]-vv[:,0])
 return d,np.abs(d)/6.0
def project_active(X,V,active):
 X=np.asarray(X,dtype=np.float64);active=np.asarray(active,dtype=np.int64)
 n=len(X);best=np.full(n,np.inf);out=np.zeros((n,4),dtype=np.float64)
 for mask in range(1,1<<len(active)):
  ids=np.asarray([active[j] for j in range(len(active)) if mask&(1<<j)],dtype=np.int64)
  W=V[ids];m=len(ids)
  if m==1:
   small=np.ones((n,1));valid=np.ones(n,dtype=bool);proj=np.broadcast_to(W[0],X.shape)
  else:
   D=W[1:]-W[0];G=D@D.T
   try:alpha=np.linalg.solve(G,((X-W[0])@D.T).T).T
   except np.linalg.LinAlgError:continue
   small=np.column_stack([1.-alpha.sum(axis=1),alpha])
   valid=np.isfinite(small).all(axis=1)&(small.min(axis=1)>=-1e-12)&(small.max(axis=1)<=1.+1e-12)
   small=np.maximum(small,0.);ss=small.sum(axis=1);valid &= ss>0
   good=valid.copy()
   small[good]/=ss[good,None]
   proj=small@W
  d2=np.einsum("ij,ij->i",proj-X,proj-X);take=valid&(d2<best)
  if take.any():
   best[take]=d2[take];out[take]=0.;out[np.ix_(take,ids)]=small[take]
 finite=np.isfinite(best)
 return out,np.sqrt(best),finite
def corner_sites(X,V,B0,active):
 P0=B0@V;off=P0-X;outB=[];outD=[];outK=[];outRows=[]
 r_safe=np.nextafter(R,0.0)
 C=np.einsum("ij,ij->i",off,off)-r_safe*r_safe
 baseok=C<=0
 for k in active:
  delta=V[k]-P0
  A=np.einsum("ij,ij->i",delta,delta)
  Bq=2*np.einsum("ij,ij->i",off,delta)
  disc=np.maximum(0.,Bq*Bq-4*A*C)
  alpha=np.zeros(len(X))
  moving=A>0
  endpoint=P0+delta
  endpoint_dist=np.linalg.norm(endpoint-X,axis=1)
  inside_end=endpoint_dist<=r_safe
  alpha[inside_end]=1.0
  solve=moving&baseok&~inside_end
  alpha[solve]=(-Bq[solve]+np.sqrt(disc[solve]))/(2*A[solve])
  alpha=np.clip(alpha,0.,1.)
  alpha[(alpha>0)&(alpha<1)]*=1.-1e-12
  b=(1.-alpha[:,None])*B0
  b[:,k]+=alpha
  point=b@V;dist=np.linalg.norm(point-X,axis=1)
  ok=baseok&(alpha>1e-14)&(dist<=R)&np.isfinite(b).all(axis=1)&(b.min(axis=1)>=0)&(np.abs(b.sum(axis=1)-1)<=2e-12)
  ok &= np.all((b==0.)| (np.asarray([True]*4)[None,:]),axis=1)
  if np.any(ok):
   outB.append(b[ok]);outD.append(dist[ok]);outK.append(np.full(np.count_nonzero(ok),int(k),dtype=np.int8));outRows.append(np.flatnonzero(ok).astype(np.int32))
 return outB,outD,outK,outRows

if OUT.exists():raise RuntimeError(f"attempt output already exists: {OUT}")
OUT.mkdir(parents=True)
inputs=[CAND,CREPORT,SUP,SUREPORT,LEDGER,LREPORT,LVERIFY,EXACT,ROWMAP,CHILDCSV,PARENTCSV,PAIR,PAIRPINS,SOURCE_NODES,SOURCE_TETS,SOURCE_LABELS,VOLUME_AUDIT,BASE/"fat.node",BASE/"fat.ele",BASE_PARTIAL/"transport-source-parent.csv"]
pins={str(p):sha(p) for p in inputs}
pins["producer_090_pair_cache_prior_run011"] = json.loads(PAIRPINS.read_text())["inputs"].get(str(PAIR))
if pins["producer_090_pair_cache_prior_run011"]!=pins[str(PAIR)]:raise RuntimeError("cached pair table no longer matches recorded run011 hash")
cand_report=json.loads(CREPORT.read_text());support_report=json.loads(SUREPORT.read_text());ledger_report=json.loads(LREPORT.read_text());ledger_ver=json.loads(LVERIFY.read_text())
if cand_report.get("physical_admission") is not False or cand_report.get("native_steps")!=0:raise RuntimeError("current candidate is not source-only")
if support_report.get("status")!="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT":raise RuntimeError("current 143 support lacks independent pass")
if support_report.get("current_fat_target",{}).get("sha256")!=sha(CAND):raise RuntimeError("current support is not bound to 406")
if ledger_ver.get("status")!="PASS_INDEPENDENT_CURRENT_HELD_LEDGER_REPLAY_SOURCE_ONLY":raise RuntimeError("held ledger lacks independent current-406 replay")
writej(OUT/"preparation-in-progress.json",{"status":"RUNNING","stage":"input-load","inputs":pins,"physical_admission":False,"native_steps":0})
say("load geometry, source rows, support, baseline held ledger")
v=np.load(CAND,allow_pickle=False);points=v["points_world_m"].astype(np.float32,copy=False);tets=v["tetrahedra"].astype(np.int64,copy=False);mat_cur=v["material_indices"].astype(np.int32,copy=False)
ncell=len(tets)
support=np.load(SUP,allow_pickle=False);off=support["cell_admissible_region_offsets"].astype(np.int64,copy=False);regions=support["cell_admissible_region_ids"].astype(np.int32,copy=False)
if len(off)!=ncell+1 or off[0]!=0 or off[-1]!=len(regions) or np.any(np.diff(off)<1):raise RuntimeError("current region support CSR is malformed")
if not np.array_equal(support["volume_sha256"],np.asarray(sha(CAND))):raise RuntimeError("current support volume pin mismatch")
with np.load(LEDGER,allow_pickle=False) as z:
 fp=z["flow_parent_role_rows"].astype(np.int64,copy=True);fc=z["flow_current_child_rows"].astype(np.int64,copy=True);fm=z["flow_mass_kg"].astype(np.float64,copy=True);fb=z["flow_source_barycentric"].astype(np.float64,copy=True);ret=z["flow_retained"].astype(bool,copy=True)
 fglobal=z["flow_global_source330_rows"].astype(np.int64,copy=True)
z=None
parents=readcsv(PARENTCSV);children=readcsv(CHILDCSV)
nparent=len(parents)
if [int(x["parent_row"]) for x in parents]!=list(range(nparent)) or nparent!=14316:raise RuntimeError("current source parent role row domain mismatch")
gparent=np.asarray([int(x["global_source330_row"]) for x in parents],dtype=np.int64)
pmat=np.asarray([int(x["material"]) for x in parents],dtype=np.int32);donor=np.asarray([int(x["geometric_donor"]) for x in parents],dtype=np.int32);region=np.asarray([int(x["region"]) for x in parents],dtype=np.int32);pmass=np.asarray([float(x["mass_kg"]) for x in parents],dtype=np.float64)
if not np.array_equal(gparent,np.arange(14316,28632,dtype=np.int64)) or not np.all(pmat==16):raise RuntimeError("source global-row/material map mismatch")
source_nodes=readcsv(SOURCE_NODES);source_tets=readcsv(SOURCE_TETS);source_labels=readcsv(SOURCE_LABELS)
sx=np.asarray([[float(r["rest_"+a+"_m"]) for a in "xyz"] for r in source_nodes],dtype=np.float64)
stet=np.asarray([[int(r["parent_node"+str(k)]) for k in range(4)] for r in source_tets],dtype=np.int64)
smat=np.asarray([int(r["material_index"]) for r in source_labels],dtype=np.int32);sdonor=np.asarray([int(r["geometric_donor_body"]) for r in source_labels],dtype=np.int32);sregion=np.asarray([int(r["anatomical_region_id"]) for r in source_labels],dtype=np.int32)
if not(np.array_equal(smat[gparent],pmat) and np.array_equal(sdonor[gparent],donor) and np.array_equal(sregion[gparent],region)):raise RuntimeError("source330 parent semantic tuple differs from pinned full parent rows")
Vsrc=sx[stet[gparent]]
if len(children)!=int(ledger_report["held_ledger_rebase"]["baseline_child_count"]):raise RuntimeError("base child volume CSV domain mismatch")
base_vol=np.asarray([float(r["reference_volume_m3"]) for r in children],dtype=np.float64)
em=np.load(EXACT,allow_pickle=False);old_to_new=em["old_to_new_exact"].astype(np.int64,copy=False)
if len(old_to_new)!=len(base_vol):raise RuntimeError("exact cell map row domain mismatch")
dets,vol=detvol(points,tets)
oldrows=np.flatnonzero(old_to_new>=0);newrows=old_to_new[oldrows]
if len(np.unique(newrows))!=len(newrows):raise RuntimeError("exact cell map is not injective")
vol[newrows]=base_vol[oldrows]
if np.any(~np.isfinite(vol)) or np.any(vol<=0):raise RuntimeError("hybrid current volume has nonpositive rows")
cent=np.zeros((ncell,3),dtype=np.float64)
for k in range(4):cent+=points[tets[:,k]].astype(np.float64)*.25
# Parent/corner moment remainder. Any held overage is released proportionally from
# that parent's rows; source mass and all four M/4 targets remain fixed.
hr=np.flatnonzero(ret);hp=fp[ret];hc=fc[ret];hm=fm[ret];hb=fb[ret];hg=fglobal[ret]
if len(hr)==0 or np.any((hp<0)|(hp>=nparent)|(hc<0)|(hc>=ncell)):raise RuntimeError("retained held-flow domains invalid")
if not np.array_equal(hg,gparent[hp]):raise RuntimeError("held global source row map mismatch")
held=np.bincount(hp,weights=hm,minlength=nparent)
heldB=np.column_stack([np.bincount(hp,weights=hm*hb[:,k],minlength=nparent) for k in range(4)])
target=np.broadcast_to(pmass[:,None]*.25,(nparent,4)).copy()
pscale=np.ones(nparent)
for p in range(nparent):
 f=1.0
 if held[p]>pmass[p] and held[p]>0:f=min(f,pmass[p]/held[p])
 for k in range(4):
  if heldB[p,k]>target[p,k] and heldB[p,k]>0:f=min(f,target[p,k]/heldB[p,k])
 if f<1:
  while held[p]*f>pmass[p] or np.any(heldB[p]*f>target[p]):f=np.nextafter(f,0.)
  pscale[p]=f
for _ in range(8):
 afterB=heldB*pscale[:,None]
 bad=np.flatnonzero(np.any(afterB>target,axis=1))
 if not len(bad):break
 pscale[bad]=np.nextafter(pscale[bad],0.)
if np.any(heldB*pscale[:,None]>target):raise RuntimeError("source M/4 downscale could not establish nonnegative residuals")
q=target-heldB*pscale[:,None]
if np.any(q<0):raise RuntimeError("negative residual corner mass")
qm=q.sum(axis=1);active=np.flatnonzero(qm>0);qbar=np.zeros_like(q);qbar[active]=q[active]/qm[active,None]
qpoint=np.einsum("ij,ijk->ik",qbar,Vsrc)
scaled_hm=hm*pscale[hp]
heldcur=np.bincount(hc,weights=scaled_hm,minlength=ncell)
low=np.maximum(0.,LOW*vol-heldcur);high=np.maximum(0.,HIGH*vol-heldcur)
unmet=np.flatnonzero(low>0)
if np.any(heldcur>HIGH*vol*(1+1e-12)):raise RuntimeError("scaled historical flows exceed current density upper bound")
# Create region->current row buckets from the independently verified CSR.
child_rows=np.repeat(np.arange(ncell,dtype=np.int32),np.diff(off))
order=np.argsort(regions,kind="stable");sr=regions[order];sc=child_rows[order]
cuts=np.r_[0,np.flatnonzero(sr[1:]!=sr[:-1])+1,len(order)]
region_children={int(sr[cuts[i]]):sc[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)}
active_by_region={int(r):active[region[active]==r] for r in np.unique(region[active])}
def allowed_pairs(c,p):
 c=np.asarray(c,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
 base=int(max(np.max(regions,initial=0),np.max(region,initial=0)))+1
 keys=child_rows.astype(np.int64)*base+regions.astype(np.int64)
 # Row-major CSR already makes these keys strictly increasing.
 query=c*base+region[p];ix=np.searchsorted(keys,query);ok=ix<len(keys)
 hit=np.zeros(len(c),dtype=bool)
 hit[ok]=keys[ix[ok]]==query[ok]
 return (mat_cur[c]==16)&(pmat[p]==16)&hit
writej(OUT/"preparation-in-progress.json",{"status":"RUNNING","stage":"hybrid-volume-and-residual-ledger","inputs":pins,"current_cells":int(ncell),"active_source_parents":int(len(active)),"residual_mass_kg":float(qm.sum()),"source_moment_scale_release_kg":float(np.sum(held*(1-pscale))),"current_low_deficit_kg":float(low.sum()),"physical_admission":False,"native_steps":0})
say(f"hybrid volume rows={len(vol)} exact-reused={len(newrows)} activeParents={len(active)} q={qm.sum():.12g}kg low-deficit={low.sum():.12g}kg")

# Candidate source-parent/current-child graph: mapped old sparse pairs plus bounded
# current-support nearest neighbours from each residual parent and each unmet child.
say("map old sparse pair cache to current 406 child rows")
pairz=np.load(PAIR,allow_pickle=False);pc_old=pairz["child_rows"].astype(np.int64,copy=False);pp=pairz["parent_local_rows"].astype(np.int32,copy=False);pg=pairz["parent_global_rows"].astype(np.int64,copy=False)
if not np.array_equal(gparent[pp],pg):raise RuntimeError("cached pair source parent row map mismatch")
currentmap=np.load(ROWMAP,allow_pickle=False)["baseline_cell_row_to_current_cell_row"].astype(np.int64,copy=False)
if len(currentmap)!=len(base_vol):raise RuntimeError("baseline/current child map mismatch")
pc=currentmap[pc_old]
ok=(pc>=0)&(pc<ncell)&(qm[pp]>0)
pc,pp=pc[ok].astype(np.int32),pp[ok]
ok=allowed_pairs(pc,pp)&(high[pc]>0)
pc,pp=pc[ok],pp[ok]
pairC=[pc];pairP=[pp];cache_pairs=len(pc)
say(f"mapped cached pairs with current row/region/headroom={cache_pairs}")
# Per-parent query: nearest 32 current receivers with authenticated source region and upper slack.
for r,pr in active_by_region.items():
 ch=region_children.get(r,np.empty(0,dtype=np.int32));ch=ch[high[ch]>0]
 if not len(ch):continue
 tree=cKDTree(cent[ch])
 k=min(PARENT_K,len(ch))
 _,ix=tree.query(qpoint[pr],k=k,workers=1)
 ix=np.asarray(ix)
 if ix.ndim==1:ix=ix[:,None]
 pairC.append(ch[ix.ravel()].astype(np.int32));pairP.append(np.repeat(pr,k).astype(np.int32))
# Parent-corner query: nearest 8 current receivers to each positive source corner.
corner_query_count=0
for r,pr in active_by_region.items():
 ch=region_children.get(r,np.empty(0,dtype=np.int32));ch=ch[high[ch]>0]
 if not len(ch):continue
 tree=cKDTree(cent[ch])
 for kcorner in range(4):
  owners=pr[q[pr,kcorner]>0]
  if not len(owners):continue
  k=min(CORNER_K,len(ch))
  _,ix=tree.query(Vsrc[owners,kcorner],k=k,workers=1)
  ix=np.asarray(ix)
  if ix.ndim==1:ix=ix[:,None]
  pairC.append(ch[ix.ravel()].astype(np.int32));pairP.append(np.repeat(owners,k).astype(np.int32));corner_query_count+=len(owners)*k
# Receiver query: for every child with a lower-density deficit, add the nearest
# residual source parent for each permitted source region.
receiver_pair_count=0
for r,pr in active_by_region.items():
 ch=region_children.get(r,np.empty(0,dtype=np.int32));ch=ch[low[ch]>0]
 if not len(ch):continue
 tree=cKDTree(qpoint[pr])
 _,ix=tree.query(cent[ch],k=min(RECEIVER_K,len(pr)),workers=1)
 ix=np.asarray(ix)
 if ix.ndim==1:ix=ix[:,None]
 valid=ix<len(pr)
 pairC.append(np.broadcast_to(ch[:,None],ix.shape)[valid].astype(np.int32));pairP.append(pr[ix[valid]].astype(np.int32));receiver_pair_count+=int(valid.sum())
# Unique parent-child pairs.
allC=np.concatenate(pairC);allP=np.concatenate(pairP)
pairkey=allC.astype(np.uint64)*np.uint64(nparent)+allP.astype(np.uint64)
pairkey=np.unique(pairkey)
allC=(pairkey//np.uint64(nparent)).astype(np.int32);allP=(pairkey%np.uint64(nparent)).astype(np.int32)
keep=allowed_pairs(allC,allP)&(high[allC]>0)&(qm[allP]>0)
allC,allP=allC[keep],allP[keep]
if len(allC)>4_500_000:raise RuntimeError(f"bounded candidate pair graph exceeds 4.5M prep cap: {len(allC)}")
say(f"candidate pair union={len(allC)} (old={cache_pairs}, receiver_added={receiver_pair_count}, corner_queries={corner_query_count})")

# Generate exact feasible barycentric site options within the unchanged hard radius.
siteC=[];siteP=[];siteB=[];siteD=[];pair_has_site=np.zeros(len(allC),dtype=bool)
perm=np.lexsort((allC,allP));c_sorted=allC[perm];p_sorted=allP[perm]
starts=np.r_[0,np.flatnonzero(p_sorted[1:]!=p_sorted[:-1])+1,len(p_sorted)]
for gi in range(len(starts)-1):
 lo=int(starts[gi]);hi=int(starts[gi+1]);p=int(p_sorted[lo]);idx=perm[lo:hi];ccrows=allC[idx];X=cent[ccrows];active_k=np.flatnonzero(q[p]>0)
 if not len(active_k):continue
 B0,dist0,finite=project_active(X,Vsrc[p],active_k)
 valid=finite&(dist0<=R)
 if not valid.any():continue
 pair_has_site[idx[valid]]=True
 # Closest point on the residual-moment simplex.
 siteC.append(ccrows[valid]);siteP.append(np.full(np.count_nonzero(valid),p,dtype=np.int32));siteB.append(B0[valid]);siteD.append(dist0[valid])
 # Residual barycentric mean site when it is inside the receiver ball.
 qd=np.linalg.norm(X-qpoint[p],axis=1);qok=qd<=R
 if qok.any():
  siteC.append(ccrows[qok]);siteP.append(np.full(np.count_nonzero(qok),p,dtype=np.int32));siteB.append(np.broadcast_to(qbar[p],(np.count_nonzero(qok),4)).copy());siteD.append(qd[qok])
 # Corner-directed feasible sites extend each supported moment toward its source corner.
 bparts,dparts,kparts,rowparts=corner_sites(X,Vsrc[p],B0,active_k)
 for bk,dk,k,rowidx in zip(bparts,dparts,kparts,rowparts):
  # Keep the exact source-corner index and candidate-row crosswalk returned by the helper.
  k=int(k[0]) if len(k) else -1
  if k not in active_k:raise RuntimeError("corner site returned an inactive source corner")
  if np.any(bk[:,k]<=0) or np.any(np.abs(bk.sum(axis=1)-1)>2e-12):raise RuntimeError("corner site barycentric replay failed")
  replay=np.linalg.norm(bk@Vsrc[p]-X[rowidx],axis=1)
  if not np.allclose(replay,dk,rtol=0.,atol=2e-15) or np.any(replay>R):raise RuntimeError("corner site radius replay failed")
  siteC.append(ccrows[rowidx]);siteP.append(np.full(len(rowidx),p,dtype=np.int32));siteB.append(bk);siteD.append(dk)
if not siteC:raise RuntimeError("no source-site candidate columns")
vc=np.concatenate(siteC).astype(np.int32);vp=np.concatenate(siteP).astype(np.int32);vb=np.vstack(siteB).astype(np.float64);vd=np.concatenate(siteD).astype(np.float64)
# Exact barycentric-bit dedup for duplicate sites on the same child/source parent.
vb[vb==0.0]=0.0
bits=np.ascontiguousarray(vb,dtype="<f8").view("<u8").reshape(-1,4)
order=np.lexsort((bits[:,3],bits[:,2],bits[:,1],bits[:,0],vp,vc))
vc,vp,vb,vd,bits=vc[order],vp[order],vb[order],vd[order],bits[order]
unique=np.ones(len(vc),dtype=bool)
unique[1:]=(vc[1:]!=vc[:-1])|(vp[1:]!=vp[:-1])|np.any(bits[1:]!=bits[:-1],axis=1)
vc,vp,vb,vd=vc[unique],vp[unique],vb[unique],vd[unique]
if len(vc)>MAX_COLUMNS:
 raise RuntimeError(f"current active LP columns exceed preserved 12M owner cap: {len(vc)}")
if not np.all(allowed_pairs(vc,vp)) or np.any(vd>R):raise RuntimeError("generated candidate violates current support or radius")
if np.any((vb>0)&(q[vp]<=0)):raise RuntimeError("generated site places mass on a zero residual source corner")
scale=np.minimum(qm[vp],high[vc])
for k in range(4):
 pos=vb[:,k]>0
 scale[pos&(q[vp,k]<=0)]=0.
 good=pos&(q[vp,k]>0)
 scale[good]=np.minimum(scale[good],q[vp[good],k]/vb[good,k])
keep=scale>0
vc,vp,vb,vd,scale=vc[keep],vp[keep],vb[keep],vd[keep],scale[keep]
if len(vc)>MAX_COLUMNS:raise RuntimeError("scaled candidate columns exceed 12M owner cap")
# Necessary single-row capacity/coverage checks; failure is a bounded graph miss,
# not a proof that the full physical transport problem is infeasible.
cap_corner=np.zeros_like(q)
for k in range(4):np.add.at(cap_corner[:,k],vp,scale*vb[:,k])
cap_child=np.bincount(vc,weights=scale,minlength=ncell)
missing_low=unmet[cap_child[unmet]<=0]
short_low=unmet[cap_child[unmet]+1e-14*np.maximum(low[unmet],1e-300)<low[unmet]]
positive_q=np.argwhere(q>0)
short_corner=np.asarray([(int(p),int(k)) for p,k in positive_q if cap_corner[p,k]+1e-12*max(q[p,k],1e-300)<q[p,k]],dtype=np.int64).reshape(-1,2)
say(f"sites={len(vc)} qmean/corner-point candidates; low_missing={len(missing_low)} low_capacity_short={len(short_low)} qcorner_capacity_short={len(short_corner)}")
# Persist exact current graph + numeric bounds before solver; source IDs and domains explicit.
np.savez_compressed(OUT/"lp-input-current406.npz",child_rows=vc,parent_role_rows=vp,parent_global_source330_rows=gparent[vp],barycentric=vb,distance_m=vd,column_mass_scale_kg=scale,residual_moments_kg=q,residual_mass_kg=qm,held_parent_scale=pscale,held_current_child_mass_kg=heldcur,child_low_kg=low,child_high_kg=high,child_volume_hybrid_m3=vol,unmet_child_rows=unmet,capacity_corner_kg=cap_corner,capacity_child_kg=cap_child)
graphhash=sha(OUT/"lp-input-current406.npz")
stage={"schema":"numi.human.current-fat-residual-site-graph.v1","status":"PREPARED_SOURCE_ONLY","physical_admission":False,"native_steps":0,"inputs":pins,"candidate_pair_count":int(len(allC)),"candidate_site_column_count":int(len(vc)),"column_graph_sha256":graphhash,"active_parent_count":int(len(active)),"positive_source_corner_count":int(np.count_nonzero(q>0)),"unmet_child_count":int(len(unmet)),"children_without_candidate_capacity":int(len(missing_low)),"children_below_lower_bound_capacity":int(len(short_low)),"source_corners_below_upper_bound_capacity":int(len(short_corner)),"current_cells":int(ncell),"current_points":int(len(points)),"hybrid_exact_identity_row_count":int(len(newrows)),"hybrid_volume_sha256":sha(OUT/"lp-input-current406.npz"),"held_flow_rows":int(len(hr)),"retained_rows":int(ret.sum()),"scaled_held_mass_kg":float(scaled_hm.sum()),"source_total_mass_kg":float(pmass.sum()),"residual_source_mass_kg":float(qm.sum()),"receiver_low_deficit_kg":float(low.sum()),"receiver_upper_slack_kg":float(high.sum()),"source_corner_capacity_shortfall_kg":float(np.maximum(q-cap_corner,0.).sum()),"receiver_capacity_shortfall_kg":float(np.maximum(low-cap_child,0.).sum()),"m4_scale_release_kg":float(np.sum(hm-scaled_hm)),"hard_radius_m":R,"density_bounds_kg_m3":[LOW,HIGH],"candidate_geometry_status":cand_report.get("status"),"elapsed_seconds":time.monotonic()-T0,"peak_rss_bytes":peakrss()}
writej(OUT/"graph-preparation-report.json",stage)
writej(OUT/"preparation-in-progress.json",{"status":"GRAPH_PREPARED","stage":"pre-solver","graph_sha256":graphhash,"inputs":pins,"physical_admission":False,"native_steps":0})
if len(missing_low) or len(short_corner):
 stage["status"]="CANDIDATE_GRAPH_INCOMPLETE_SOURCE_ONLY"
 stage["limitations"]=["This bounded sparse graph omits capacity for at least one current receiver lower bound or source-corner remainder; no LP was launched."]
 writej(OUT/"lp-report.json",stage)
 print(json.dumps(stage,sort_keys=True),flush=True)
 raise SystemExit(0)
# Assemble normalized feasibility LP with every active source-corner equality and every
# touched receiver upper/lower bound. No density/radius/region relaxation.
say("build sparse LP matrix")
eqrow=np.full((nparent,4),-1,dtype=np.int64);nr=0
for p,k in positive_q:
 eqrow[int(p),int(k)]=nr;nr+=1
cols=np.arange(len(vc),dtype=np.int64);rr=[];ccol=[];vv=[]
for k in range(4):
 use=(vb[:,k]>0)&(q[vp,k]>0)
 ix=np.flatnonzero(use)
 if len(ix):
  rr.append(eqrow[vp[use],k]);ccol.append(ix);vv.append(scale[use]*vb[use,k]/q[vp[use],k])
Aeq=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(ccol))),shape=(nr,len(vc))).tocsr()
touched=np.unique(vc);cmap=np.full(ncell,-1,dtype=np.int32);cmap[touched]=np.arange(len(touched),dtype=np.int32);cr=cmap[vc].astype(np.int64)
coef=scale/high[vc]
Aub=coo_matrix((np.r_[coef,-coef],(np.r_[2*cr,2*cr+1],np.r_[cols,cols])),shape=(2*len(touched),len(vc))).tocsr()
bub=np.empty(2*len(touched));bub[::2]=1.;bub[1::2]=-low[touched]/high[touched]
beq=np.ones(nr);cost=np.zeros(len(vc),dtype=np.float64)
np.savez_compressed(OUT/"lp-matrix-summary.npz",equality_count=np.asarray(nr),upper_row_count=np.asarray(2*len(touched)),variable_count=np.asarray(len(vc)),equality_nnz=np.asarray(Aeq.nnz),upper_nnz=np.asarray(Aub.nnz))
writej(OUT/"solver-in-progress.json",{"status":"RUNNING","inputs":pins,"graph_sha256":graphhash,"variable_count":int(len(vc)),"equality_count":int(nr),"touched_child_count":int(len(touched)),"Aeq_nnz":int(Aeq.nnz),"Aub_nnz":int(Aub.nnz),"solver":"SciPy HiGHS zero-objective feasibility","time_limit_seconds":120,"physical_admission":False,"native_steps":0})
say(f"HiGHS feasibility solve vars={len(vc)} eq={nr} touched={len(touched)} nnz={Aeq.nnz+Aub.nnz}")
res=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=(0.,1.),method="highs",options={"time_limit":120.,"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,"presolve":True})
rep=dict(stage);rep.update({"schema":"numi.human.current-fat-residual-source-site-lp.v1","status":"LP_"+str(res.status),"message":str(res.message),"solver":"SciPy HiGHS","solver_objective":"zero feasibility objective; no cost-optimality claim","solver_time_limit_s":120.,"solver_success":bool(res.success),"elapsed_seconds":time.monotonic()-T0,"peak_rss_bytes":peakrss(),"recorded_flow_feasibility":"NOT_YET_VERIFIED","deterministic_optimizer_replay":"NOT_RUN","mass_transport":"NOT_ADMITTED"})
if res.x is not None:
 x=scale*res.x
 addChild=np.bincount(vc,weights=x,minlength=ncell)
 addCorner=np.zeros_like(q)
 for k in range(4):np.add.at(addCorner[:,k],vp,x*vb[:,k])
 finalMass=heldcur+addChild;density=finalMass/vol
 momentDelta=addCorner-q
 momentRel=np.zeros_like(q);np.divide(np.abs(momentDelta),np.maximum(q,1e-300),out=momentRel,where=q>0)
 lowSlack=finalMass-LOW*vol;highSlack=HIGH*vol-finalMass
 exactPoint=np.einsum("ij,ijk->ik",vb,Vsrc[vp])
 actualDist=np.linalg.norm(exactPoint-cent[vc],axis=1)
 checks={
  "all_source_corner_moments_close":bool(np.max(momentRel,initial=0.)<=1e-8),
  "all_parent_total_mass_close":bool(np.max(np.abs((scaled_hm.sum()*0+np.bincount(hp,weights=scaled_hm,minlength=nparent)+np.bincount(vp,weights=x,minlength=nparent))-pmass)/np.maximum(pmass,1e-300),initial=0.)<=1e-8),
  "all_current_receiver_lower_bounds":bool(np.all(lowSlack>=-1e-10*LOW*vol)),
  "all_current_receiver_upper_bounds":bool(np.all(highSlack>=-1e-10*HIGH*vol)),
  "all_added_site_radii":bool(np.all(actualDist<=R)),
  "all_added_sites_inside_source_parent_simplex":bool(np.all(vb>=0)&np.all(np.abs(vb.sum(axis=1)-1)<=2e-12)),
  "all_added_source_regions_authenticated":bool(np.all(allowed_pairs(vc,vp))),
  "all_added_materials_match":bool(np.all(mat_cur[vc]==pmat[vp])),
  "no_added_mass_on_zero_residual_corner":bool(not np.any((vb>0)&(q[vp]<=0))),
 }
 rep.update({"independent_primal_checks":checks,"max_parent_corner_relative_residual":float(np.max(momentRel,initial=0.)),"max_parent_corner_absolute_residual_kg":float(np.max(np.abs(momentDelta),initial=0.)),"max_parent_total_relative_residual":float(np.max(np.abs(np.bincount(hp,weights=scaled_hm,minlength=nparent)+np.bincount(vp,weights=x,minlength=nparent)-pmass)/np.maximum(pmass,1e-300),initial=0.)),"current_density_min_kg_m3":float(density.min()),"current_density_max_kg_m3":float(density.max()),"under_850_cells":int(np.count_nonzero(density<LOW)),"over_950_cells":int(np.count_nonzero(density>HIGH)),"unallocated_source_mass_kg":float(pmass.sum()-(scaled_hm.sum()+x.sum())),"residual_added_mass_kg":float(x.sum()),"max_added_distance_m":float(actualDist.max(initial=0.)),"positive_added_rows":int(np.count_nonzero(x>0)),"max_abs_density_violation_kg_m3":float(max(np.max(np.maximum(-lowSlack/vol,0.),initial=0.),np.max(np.maximum(-highSlack/vol,0.),initial=0.)))})
 finalpass=bool(res.success and all(checks.values()))
 rep["status"]="PASS_SOURCE_ONLY_CURRENT406_RESIDUAL_LP_CANDIDATE" if finalpass else "FAIL_OR_UNQUALIFIED_CURRENT406_RESIDUAL_LP_CANDIDATE"
 np.savez_compressed(OUT/"candidate-current406-mass-allocation.npz",
    held_source_parent_role_rows=hp.astype(np.int32),held_source_global_rows=gparent[hp],held_current_child_rows=hc.astype(np.int32),held_flow_mass_kg=scaled_hm,held_source_barycentric=hb,
    added_source_parent_role_rows=vp.astype(np.int32),added_source_global_rows=gparent[vp],added_current_child_rows=vc,added_flow_mass_kg=x,added_source_barycentric=vb,added_distance_m=actualDist,
    current_child_mass_kg=finalMass,current_child_volume_hybrid_m3=vol,current_child_density_kg_m3=density,parent_source_mass_kg=pmass,parent_corner_mass_kg=(heldB*pscale[:,None]+addCorner),parent_corner_target_kg=target)
 rep["proposal_path"]=str(OUT/"candidate-current406-mass-allocation.npz");rep["proposal_sha256"]=sha(OUT/"candidate-current406-mass-allocation.npz")
writej(OUT/"lp-report.json",rep);writej(OUT/"solver-in-progress.json",{"status":"SOLVER_COMPLETE","lp_report_sha256":sha(OUT/"lp-report.json"),"inputs":pins,"physical_admission":False,"native_steps":0})
say(f"RESULT {rep['status']} {res.message}")
print(json.dumps(rep,indent=2,sort_keys=True),flush=True)

