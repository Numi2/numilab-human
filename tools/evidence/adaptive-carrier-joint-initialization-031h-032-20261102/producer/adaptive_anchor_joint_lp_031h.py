#!/usr/bin/env python3
"""Versioned source-only adaptive carrier capacity proposal (031)."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK=H/"current-source-carrier-joint-initialization-1054/cut-closure-1054"
RUN=TASK/"actual-028-complete-predicate-joint-lp-001/run-001"
MATRIX=RUN/"anchor-cut-matrix.csv"; EXEC=RUN/"execution.json"; MANIFEST=RUN/"run-manifest.json"
OWNER030=TASK/"actual-030-joint-lp-001/h706_joint_hall_lp_030.py"
HELPER030=TASK/"actual-025-joint-lp-003-maximin-prep/cut_capacity_lp_1054_actual025_maximin.py"
PROJECTOR=H/"batched_reference_tet_projection_149_root.py"
OWNER1046H=H/"current_source_carrier_bridge_1046h.py"
C027=H/"current-fractional-semantic-native-1047/actual-027-proof-slot-closure"
COMP=C027/"actual-027-full-deficit-component-cap-009"
COMP_SUMMARY=COMP/"full-unreachable-component-summary.json"
COMP_CENSUS=COMP/"component-census/component-census.json"
SUPPORT_MANIFEST=C027/"verified-flow/capacity-witness/manifest.json"
CURRENT027_RUN=C027/"run-manifest.json"
EXPECTED_SUPPORT="76479632eae7c84f6a905f741f7ec7e84dab0eba9f8ef94fe4336cef8d77f854"
EXPECTED_MATRIX="0f824600b986af1458349033f14d4f9a28edcb627cd1fa395c4922b23de95600"
EXPECTED_EXEC="871f40b55703a4b22d662b3c18ded02354c808303b656f07a224f063f78ce674"
EXPECTED_RUN="bbae56df5acbcb203afd53c583d6346b09a76db9278d6cbc89eb5c68709b3f07"
EXPECTED_030_SOURCE="019d263b969303f61b7baec26501e109d3aac768b0b88f2d642a3b53cb24838d"
EXPECTED_030_PY_SOURCE="1a0385ce79ed4fff016d04bd86464e5149586704beb4f500c2ea50b2cdf188ad"
EXPECTED_COMP_SUMMARY="4e9b32fb7fc08cbd5f8b97f210ed1dfc275d58811b91260409c41520c5672737"
EXPECTED_COMP_CENSUS="977b2558f72a4897763fce87f016857cfb7d67442808d1a891cdd557bc23c679"
EXPECTED_SUPPORT_MANIFEST="48b3675d82433011ff80fb56574d3f4c6cbed9cb368dc053d340ba97e9781e0f"
EXPECTED_CURRENT027_MANIFEST="c6ebc27b9944f8a39da4420f4a9c56fff6a0c70b34a6384322d8939bf9537b0e"
EXPECTED_1046H="947e3e9d9a4993be7e73c5f06a1f25d92721ceceed8f2e05cef751807b8d09ec"
EXPECTED_PROJECTOR="02b04cff325b4adea24d055fdfd2121ab4d596a3c075ab6fe8183094eb46e1d6"
EXPECTED_PROJECTOR="02b04cff325b4adea24d055fdfd2121ab4d596a3c075ab6fe8183094eb46e1d6"
R=0.019353; MARGIN=1e-7
CUT_NAMES=["actual014","actual024"]+[f"actual025_component_{i:02d}" for i in range(10)]+["actual025_fullcut","actual026_fullcut","actual027_fullcut"]
GOALS=np.vstack([np.eye(4),np.array([[.5 if k in pair else 0. for k in range(4)] for pair in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]),np.full((1,4),.25)])

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()

def pin(path,digest,size=None):
 p=Path(path)
 if not p.is_file() or (size is not None and p.stat().st_size!=size) or sha(p)!=digest:
  raise RuntimeError(f"pinned input mismatch: {p}")

def loadmod(path,name):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None: raise RuntimeError(f"cannot import owner {path}")
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def run030preflight(mod,out):
 p=out.parent/(out.name+".030-preflight")
 if p.exists(): raise RuntimeError(f"refusing to overwrite preflight {p}")
 old=sys.argv
 try:
  sys.argv=[str(OWNER030),"--run-dir",str(RUN),"--out-dir",str(p),"--helper",str(HELPER030),"--preflight-only"]
  rc=mod.main()
 finally: sys.argv=old
 if rc: raise RuntimeError(f"030 preflight exit {rc}")
 receipt=p/"preflight.json";d=json.loads(receipt.read_text())
 if (d.get("status")!="PASS_SOURCE_ONLY_LP_INPUT_PREFLIGHT"
     or d.get("physical_admission",False) not in (False,None)
     or d.get("native_steps",0) not in (0,None)):
  raise RuntimeError("030 preflight was not a source-only pass")
 return receipt

def parse_matrix(path,n):
 ints={k:np.empty(n,np.int64) for k in ["anchor_index","source_parent_row","parent_input_row","packet_carrier_row","material","donor","region","route_mask","global_route","global_witness_child_row"]}
 floats={k:np.empty(n,np.float64) for k in ["parent_mass_kg","prior_weight","bary0","bary1","bary2","bary3","source_x_m","source_y_m","source_z_m","global_witness_distance_m"]}
 wit=np.full((n,15),-1,np.int64);dist=np.full((n,15),np.nan,np.float64)
 with Path(path).open(newline="",encoding="utf-8") as f:
  rd=csv.DictReader(f)
  if not rd.fieldnames: raise RuntimeError("matrix header missing")
  for i,row in enumerate(rd):
   if i>=n: raise RuntimeError("matrix row overflow")
   for k in ints:
    if row[k]=="" and k in ("packet_carrier_row","global_witness_child_row"): ints[k][i]=-1
    else: ints[k][i]=int(row[k])
   for k in floats: floats[k][i]=float(row[k]) if row[k] else np.nan
   for j in range(15):
    v=row[f"cut{j:02d}_witness_child_row"];d=row[f"cut{j:02d}_witness_distance_m"]
    wit[i,j]=int(v) if v else -1;dist[i,j]=float(d) if d else np.nan
 if i+1!=n: raise RuntimeError(f"matrix row count changed: {i+1} != {n}")
 return ints,floats,wit,dist

def read_targets(mod,path,expected):
 x=mod.read_child_targets(Path(path),int(expected))
 return x

def best_witness(parent,wit,dist,pcount):
 bc=np.full((pcount,15),-1,np.int64);bd=np.full((pcount,15),np.inf,np.float64);pairs=[]
 for j in range(15):
  ix=np.flatnonzero(wit[:,j]>=0)
  if not len(ix): continue
  order=np.lexsort((wit[ix,j],dist[ix,j],parent[ix]));ix=ix[order];pp=parent[ix]
  first=np.r_[True,pp[1:]!=pp[:-1]];ix=ix[first];pp=parent[ix]
  bc[pp,j]=wit[ix,j];bd[pp,j]=dist[ix,j]
  pairs.extend((int(p),j,int(wit[r,j]),float(dist[r,j])) for p,r in zip(pp,ix))
 return bc,bd,pairs

def current11_reconcile(manifest,cutset,masks,wit,nchild,outpath):
 pin(COMP_SUMMARY,EXPECTED_COMP_SUMMARY);pin(COMP_CENSUS,EXPECTED_COMP_CENSUS);pin(SUPPORT_MANIFEST,EXPECTED_SUPPORT_MANIFEST);pin(CURRENT027_RUN,EXPECTED_CURRENT027_MANIFEST)
 summ=json.loads(COMP_SUMMARY.read_text());census=json.loads(COMP_CENSUS.read_text());sp=json.loads(SUPPORT_MANIFEST.read_text())
 if summ.get("component_count")!=11 or census.get("component_count")!=11: raise RuntimeError("current 027 no longer has 11 target components")
 if summ.get("typed_support_sha256")!=EXPECTED_SUPPORT or census.get("expanded_typed_support_sha256")!=EXPECTED_SUPPORT:
  raise RuntimeError("current component reports do not bind exact expanded support")
 if sp.get("status")!="INFEASIBLE_FULL_GRAPH_CUT" or sp.get("metadata",{}).get("typed_support_sha256")!=EXPECTED_SUPPORT:
  raise RuntimeError("current 027 fullcut witness is not the pinned exact-support cut")
 if summ.get("current027_manifest_sha256")!=EXPECTED_CURRENT027_MANIFEST or manifest.get("seed_run_manifest",{}).get("sha256")!=EXPECTED_CURRENT027_MANIFEST:
  raise RuntimeError("030 source export seed differs from current 027 manifest")
 comp_by=np.full(nchild,-1,np.int16);rows=[];refs=[]
 for c in census["components"]:
  cid=int(c["component_id"]);p=Path(c["target_rows_path"])
  a=read_targets(loadmod(OWNER030,"owner030_for_cutreader"),p,c["target_child_count"])
  if sha(p)!=c["target_rows_sha256"] or np.any(a>=nchild) or np.any(comp_by[a]>=0):
   raise RuntimeError("current component child map is stale, overlapping or out of range")
  comp_by[a]=cid;rows.append(a);refs.append({"id":cid,"path":str(p),"sha256":c["target_rows_sha256"],"rows":len(a),
      "minimum_demand_kg":float(c["minimum_demand_kg"]),"optimistic_existing_anchor_capacity_kg":float(c["existing_anchor_capacity_kg"])})
 union=comp_by>=0
 if not np.array_equal(union,masks[14]) or int(union.sum())!=588632: raise RuntimeError("11 current targets do not partition exact actual027 fullcut union")
 overlaps=[]
 for j in range(2,12):
  r=np.flatnonzero(masks[j]);counts=np.bincount(comp_by[r][comp_by[r]>=0],minlength=11)
  overlaps.append({"historical_cut":CUT_NAMES[j],"historical_rows":len(r),"current_component_intersections":counts.tolist(),
                  "historical_rows_in_current_union":int(np.count_nonzero(comp_by[r]>=0))})
 cw=wit[:,14];valid=cw>=0;hist=np.bincount(comp_by[cw[valid]][comp_by[cw[valid]]>=0],minlength=11)
 result={"schema":"numi.h706.current11-target-reconciliation.031.v1","status":"PASS_TARGET_SET_RECONCILIATION",
  "physical_admission":False,"native_steps":0,"current_support_sha256":EXPECTED_SUPPORT,
  "current027_manifest_sha256":summ["current027_manifest_sha256"],"current_union_rows":int(union.sum()),
  "current_union_equals_030_cut14":True,"current_components":refs,"historical_actual025_component_intersections":overlaps,
  "cut14_representative_witness_rows_by_component":hist.tolist(),
  "cut14_witnesses_without_current_component":int(np.count_nonzero(valid&(comp_by[cw.clip(0,nchild-1)]<0))),
  "scope":"target mapping and representative-witness histogram only; 031 objective retains the pinned 15 cuts and does not claim 11 componentwise feasibility"}
 Path(outpath).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 return result

def selected_child_geometry(rows,node_path,ele_path,labels_path,nchild):
 wanted=set(map(int,np.unique(rows)))
 with Path(ele_path).open() as f:
  h=f.readline().split()
  if len(h)<3 or int(h[1])!=4: raise RuntimeError("invalid child mesh.ele header")
  nt=int(h[0]);tn={}
  for i,line in enumerate(f):
   if i>=nt: raise RuntimeError("mesh.ele row overflow")
   if i not in wanted: continue
   q=line.split()
   if len(q)<5 or int(q[0])!=i+1: raise RuntimeError(f"selected .ele identity mismatch {i}")
   tn[i]=np.array([int(x)-1 for x in q[1:5]],np.int64)
  if len(tn)!=len(wanted): raise RuntimeError("mesh.ele misses selected witnesses")
 nodes=set(map(int,np.concatenate(list(tn.values()))))
 with Path(node_path).open() as f:
  h=f.readline().split()
  if len(h)<4 or int(h[1])!=3: raise RuntimeError("invalid child mesh.node header")
  nn=int(h[0]);xyz={}
  for i,line in enumerate(f):
   if i>=nn: raise RuntimeError("mesh.node row overflow")
   if i not in nodes: continue
   q=line.split()
   if len(q)<4 or int(q[0])!=i+1: raise RuntimeError(f"selected .node identity mismatch {i}")
   xyz[i]=np.array([float(v) for v in q[1:4]],np.float64)
  if len(xyz)!=len(nodes): raise RuntimeError("mesh.node misses selected witness nodes")
 mat={};role_local={};counts={15:0,16:0}
 with Path(labels_path).open(newline="",encoding="utf-8") as f:
  rd=csv.DictReader(f)
  if not rd.fieldnames or not {"child_tet_row","material_index"}.issubset(rd.fieldnames): raise RuntimeError("child labels missing required fields")
  for i,row in enumerate(rd):
   if i>=nchild: raise RuntimeError("child labels row overflow")
   m=int(row["material_index"])
   if i in wanted:
    if int(row["child_tet_row"])!=i: raise RuntimeError("selected child-label row ID mismatch")
    mat[i]=m;role_local[i]=counts.get(m,-1)
   if m in counts: counts[m]+=1
 if len(mat)!=len(wanted): raise RuntimeError("child labels miss selected witness rows")
 sr=np.array(sorted(wanted),np.int64);cent=np.empty((len(sr),3),np.float64)
 for k,r in enumerate(sr): cent[k]=np.mean(np.vstack([xyz[int(x)] for x in tn[int(r)]]),axis=0)
 return sr,cent,mat,role_local

def generate_sites(tetxyz,queries,pairp,paircut,pairchild,pairdist,projector_path):
 inv=np.linalg.inv(np.transpose(tetxyz[:,1:,:]-tetxyz[:,0:1,:],(0,2,1)))
 pm=pairp[:,None]*4+np.arange(4,dtype=np.int64)[None,:]
 point_pool=tetxyz.reshape(-1,3)
 prj=loadmod(projector_path,"projection149_for031")
 proj=prj.project_pairs(queries,np.arange(len(pairp),dtype=np.int64),point_pool,pm,inv[pairp])
 if np.any(proj["distance"]>pairdist+1e-12): raise RuntimeError("projected point farther than its pinned in-tet anchor witness")
 bp=[proj["bary"]];pp=[pairp.copy()];sc=[paircut.copy()];sw=[pairchild.copy()];sk=[np.zeros(len(pairp),np.int8)];sq=[queries.copy()]
 for gi,g in enumerate(GOALS):
  t=tetxyz[pairp];goal=np.einsum("i,nij->nj",g,t);direction=goal-proj["projected"];rel=proj["projected"]-queries
  aa=np.einsum("ij,ij->i",direction,direction);bb=2*np.einsum("ij,ij->i",direction,rel);cc=np.einsum("ij,ij->i",rel,rel)-R*R
  disc=np.maximum(0,bb*bb-4*aa*cc);tmax=np.divide(-bb+np.sqrt(disc),2*aa,out=np.zeros_like(aa),where=aa>0)
  tt=.95*np.minimum(1,np.maximum(0,tmax));b=proj["bary"]*(1-tt[:,None])+g[None,:]*tt[:,None]
  d=np.linalg.norm(np.einsum("ni,nij->nj",b,t)-queries,axis=1);ok=(d<=R+1e-12)&(tt>1e-14)
  if np.any(ok):
   bp.append(b[ok]);pp.append(pairp[ok]);sc.append(paircut[ok]);sw.append(pairchild[ok]);sk.append(np.full(int(ok.sum()),gi+1,np.int8));sq.append(queries[ok])
 b=np.vstack(bp);p=np.concatenate(pp);c=np.concatenate(sc);ch=np.concatenate(sw);k=np.concatenate(sk);q=np.vstack(sq)
 if np.any(b<0) or np.max(np.abs(b.sum(axis=1)-1))>2e-12: raise RuntimeError("generated barycentrics fail simplex replay")
 # Exact de-duplication only, preserving the first deterministic witness seed.
 _,ix=np.unique(np.column_stack((p.astype(np.float64),b)),axis=0,return_index=True);ix=np.sort(ix)
 outp=p[ix];outb=b[ix];outq=q[ix];outxyz=np.einsum("ni,nij->nj",outb,tetxyz[outp]);outdist=np.linalg.norm(outxyz-outq,axis=1)
 if np.any(outdist>R+1e-12): raise RuntimeError("adaptive seed-child primary route exceeds fixed radius")
 return outp,outb,c[ix],ch[ix],k[ix],outxyz,outdist,{"seed_parent_cut_pairs":len(pairp),"raw_site_rows":len(p),"unique_exact_site_rows":len(ix),"projection_rows":len(pairp),"goal_count":len(GOALS)}

def lp_solve(mass,ap,bary,mask,base_global_route,demands,time_limit):
 n=len(ap);p=len(mass);k=len(demands)
 cidx=np.arange(n,dtype=np.int64)
 eqr=np.concatenate([ap,p+3*ap,p+3*ap+1,p+3*ap+2]);eqc=np.tile(cidx,4)
 eqv=np.concatenate([np.ones(n),bary[:,0],bary[:,1],bary[:,2]])
 Aeq=coo_matrix((eqv,(eqr,eqc)),shape=(4*p,n+k)).tocsr();beq=np.r_[np.ones(p),np.full(3*p,.25)]
 req=demands+MARGIN;ur=[];uc=[];uv=[]
 for j in range(k):
  ix=np.flatnonzero(mask&(1<<j))
  if len(ix):
   ur.extend([j]*len(ix));uc.extend(ix.tolist());uv.extend((-mass[ap[ix]]/req[j]).tolist())
  ur.append(j);uc.append(n+j);uv.append(-1.)
 Aub=coo_matrix((uv,(ur,uc)),shape=(k,n+k)).tocsr()
 obj=np.r_[np.zeros(n),np.ones(k)]
 lb=np.zeros(n+k);ub=np.full(n+k,np.inf);base_n=len(base_global_route)
 if base_n>n: raise RuntimeError("base global-route vector exceeds LP anchor domain")
 unroutable=np.flatnonzero(~np.asarray(base_global_route,dtype=bool))
 ub[unroutable]=0.0
 res=linprog(obj,A_ub=Aub,b_ub=-np.ones(k),A_eq=Aeq,b_eq=beq,bounds=np.column_stack((lb,ub)),method="highs",
  options={"time_limit":time_limit,"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,"presolve":True})
 if res.x is None:return res,np.empty(0),{"passed_parent_moments":False}
 w=np.asarray(res.x[:n]);slack=np.asarray(res.x[n:]);sums=np.bincount(ap,weights=w,minlength=p)
 mom=np.column_stack([np.bincount(ap,weights=w*bary[:,a],minlength=p) for a in range(4)])
 caps=np.array([np.sum(mass[ap[(mask&(1<<j))!=0]]*w[(mask&(1<<j))!=0]) for j in range(k)])
 norm=np.maximum(0,(req-caps)/req)
 rep={"solver_success":bool(res.success),"solver_status":int(res.status),"solver_message":str(res.message),
  "parent_weight_sum_error_max":float(np.max(np.abs(sums-1))),"parent_barycentric_moment_error_max":float(np.max(np.abs(mom-.25))),
  "positive_weight_count":int(np.count_nonzero(w>0)),"normalized_shortfall_variable":slack.tolist(),
  "normalized_shortfall_replay":norm.tolist(),"shortfall_kg":np.maximum(0,req-caps).tolist(),
  "capacity_kg":caps.tolist(),"required_kg_with_margin":req.tolist(),
  "shortfall_replay_error_max":float(np.max(np.abs(norm-slack))),
  "passed_parent_moments":bool(np.max(np.abs(sums-1))<=2e-10 and np.max(np.abs(mom-.25))<=2e-10),
  "passed_slack_replay":bool(np.max(np.abs(norm-slack))<=2e-8)}
 return res,w,rep

def self_test():
 tet=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
               [[5.,0.,0.],[6.,0.,0.],[5.,1.,0.],[5.,0.,1.]],
               [[10.,0.,0.],[11.,0.,0.],[10.,1.,0.],[10.,0.,1.]]])
 query=np.array([[10.2,.2,.2]])
 pp=np.array([2],np.int64);cc=np.array([14],np.int64);ch=np.array([4],np.int64);dd=np.array([0.1])
 p,b,sc,sw,sk,xyz,dist,proof=generate_sites(tet,query,pp,cc,ch,dd,PROJECTOR)
 assert len(p)>=1 and np.all(b>=0) and np.max(np.abs(b.sum(axis=1)-1))<1e-12
 assert proof["projection_rows"]==1
 assert np.array_equal(p,np.full(len(p),2,dtype=np.int64))
 expected=np.einsum("ni,nij->nj",b,tet[p])
 assert np.max(np.abs(xyz-expected))<1e-14 and np.all(dist<=R+1e-12)
 res,w,r=lp_solve(np.array([2.]),np.array([0,0,0,0]),np.vstack([np.full(4,.25),np.eye(4)[0],np.eye(4)[1],np.eye(4)[2]]),np.ones(4,dtype=np.uint32),np.ones(4,bool),np.array([1.]),5.)
 assert r["passed_parent_moments"] and r["passed_slack_replay"]
 btest=np.vstack([np.full(4,.25),np.eye(4)])
 rt,wt,rr=lp_solve(np.array([2.]),np.zeros(5,dtype=np.int64),btest,np.ones(5,dtype=np.uint32),np.array([False,True,True,True,True]),np.array([1.]),5.)
 assert rr["passed_parent_moments"] and wt[0]==0.0
 print(json.dumps({"status":"PASS_031B_SYNTHETIC_CONTROLS","adaptive_site_count":len(p),"lp_status":r,"unroutable_anchor_fixed_zero":bool(wt[0]==0.0)},sort_keys=True))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=("prepare","solve"),required=True);ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--prepared-dir",type=Path);ap.add_argument("--time-limit-seconds",type=float,default=300.);ap.add_argument("--self-test",action="store_true")
 a=ap.parse_args()
 if a.self_test:self_test();return 0
 out=a.out_dir.resolve()
 if out.exists():raise RuntimeError("refusing to overwrite "+str(out))
 out.parent.mkdir(parents=True,exist_ok=True)
 pin(OWNER030,EXPECTED_030_PY_SOURCE);pin(MATRIX,EXPECTED_MATRIX);pin(EXEC,EXPECTED_EXEC);pin(MANIFEST,EXPECTED_RUN);pin(OWNER1046H,EXPECTED_1046H);pin(PROJECTOR,EXPECTED_PROJECTOR)
 mod=loadmod(OWNER030,"h706_030_preflight_for031")
 preflight=run030preflight(mod,out)
 ex=json.loads(EXEC.read_text());mf=json.loads(MANIFEST.read_text());named=mf["input_named_pins"]
 masstab,labeltab,nodetab,tettab,geom=mod.source_arrays(Path(named["source330_mass"]["path"]),Path(named["source330_labels"]["path"]),Path(named["source330_nodes"]["path"]),Path(named["source330_tets"]["path"]))
 mass=masstab["mass_kg"];mat=masstab["material_index"].astype(np.int64);don=labeltab["geometric_donor_body"].astype(np.int64);reg=labeltab["anatomical_region_id"].astype(np.int64);tetxyz=geom["tet_xyz"]
 nchild=len(mod.read_csv_numeric(Path(named["child_skin_labels"]["path"]),("child_tet_row",))["child_tet_row"]);n=ex["matrix_csv"]["rows"]
 ints,fl,wit,dist=parse_matrix(MATRIX,n);parents=ints["parent_input_row"];baseb=np.column_stack([fl[f"bary{i}"] for i in range(4)]);route=ints["route_mask"]
 if not np.array_equal(ints["source_parent_row"],parents): raise RuntimeError("030 source_parent_row and parent_input_row mappings differ")
 if len(parents)!=n or np.any(parents<0) or np.any(parents>=len(mass)): raise RuntimeError("030 matrix parent row is outside source330 domain")
 if not np.array_equal(ints["material"],mat[parents]) or not np.array_equal(ints["donor"],don[parents]) or not np.array_equal(ints["region"],reg[parents]):
  raise RuntimeError("030 matrix source family labels do not match pinned source330 rows")
 global_ok=ints["global_route"]!=0;global_child=ints["global_witness_child_row"];global_dist=fl["global_witness_distance_m"]
 if np.any(global_ok&(global_child<0)) or np.any(global_ok&(global_child>=nchild)) or np.any(global_ok&~np.isfinite(global_dist)) or np.any(global_ok&(global_dist>R+1e-12)):
  raise RuntimeError("030 global-route primary witness is missing, nonfinite or outside the fixed radius")
 if np.any(~global_ok&(global_child!=-1)) or np.any(~global_ok&np.isfinite(global_dist)):
  raise RuntimeError("030 unroutable base row has spurious primary witness data")
 if np.max(np.abs(fl["parent_mass_kg"]-mass[parents]))>1e-15: raise RuntimeError("030 matrix parent masses do not match source330")
 prior=np.zeros(len(mass));np.add.at(prior,parents,fl["prior_weight"])
 if np.max(np.abs(prior-1.0))>2e-12: raise RuntimeError("030 matrix does not retain full per-parent prior quadrature")
 cutdoc=json.loads(Path(mf["cut_set_manifest"]["path"]).read_text());masks=[];dem=[]
 for j,it in enumerate(cutdoc["cut_order"]):
  rr=read_targets(mod,it["target"]["path"],it["target"]["rows"]);m=np.zeros(nchild,bool);m[rr]=1;masks.append(m)
  d=it["origin"].get("demand_kg")
  if d is None:
   prev=[dem[k] for k in range(j) if cutdoc["cut_order"][k]["target"]["sha256"]==it["target"]["sha256"]]
   if not prev:raise RuntimeError("cut target has no authenticated demand")
   d=prev[0]
  dem.append(float(d))
 dem=np.asarray(dem)
 if [x["name"] for x in cutdoc["cut_order"]]!=CUT_NAMES or len(dem)!=15:raise RuntimeError("pinned 15 cut order mismatch")
 if a.mode=="prepare":
  # current11 report is derived from current027, while LP scope remains 15.
  out.mkdir(parents=True)
  current=current11_reconcile(mf,cutdoc,masks,wit,nchild,out/"current11-target-reconciliation.json")
  bc,bd,pairs=best_witness(parents,wit,dist,len(mass))
  pairp=np.asarray([x[0] for x in pairs],np.int64);pairc=np.asarray([x[1] for x in pairs],np.int64);pairchild=np.asarray([x[2] for x in pairs],np.int64);pairdist=np.asarray([x[3] for x in pairs],np.float64)
  unique_child=np.unique(pairchild)
  rows,childxyz,childmat,rolelocal=selected_child_geometry(unique_child,Path(named["child_skin_node"]["path"]),Path(named["child_skin_elements"]["path"]),Path(named["child_skin_labels"]["path"]),nchild)
  q=childxyz[np.searchsorted(rows,pairchild)]
  if any(childmat[int(c)]!=mat[int(p)] for p,c in zip(pairp,pairchild)):raise RuntimeError("pinned matrix witness crosses material")
  p1={}
  for material,key in ((15,"p1_skin_support"),(16,"p1_fat_support")):
   with np.load(named[key]["path"],allow_pickle=False) as z:
    offsets=z["cell_admissible_region_offsets"];ids=z["cell_admissible_region_ids"]
    if offsets.ndim!=1 or ids.ndim!=1 or len(offsets)<2 or int(offsets[0])!=0 or int(offsets[-1])!=len(ids) or np.any(offsets[1:]<offsets[:-1]):
     raise RuntimeError(f"invalid pinned P1 region CSR for material {material}")
    p1[material]=(offsets,ids)
  print(f"P1_REGION_CSR_CACHED skin_offsets={len(p1[15][0])} skin_ids={len(p1[15][1])} fat_offsets={len(p1[16][0])} fat_ids={len(p1[16][1])}",flush=True)
  for p,c in zip(pairp,pairchild):
   local=rolelocal[int(c)];o,ids=p1[int(mat[p])]
   if local<0 or int(reg[p]) not in ids[o[local]:o[local+1]]:raise RuntimeError("030 witness fails independently replayed 029 one-hop region")
  cp,cb,seedc,seedchild,sitekind,sitexyz,seed_dist,gen=generate_sites(tetxyz,q,pairp,pairc,pairchild,pairdist,PROJECTOR)
  mask=np.zeros(len(cp),np.uint32);cw=np.full((len(cp),15),-1,np.int32);cd=np.full((len(cp),15),np.nan,np.float64)
  allchild=np.unique(bc[bc>=0]);cloc=np.searchsorted(rows,allchild);allxyz=childxyz[cloc];locmap=np.full(bc.shape,-1,np.int64);vv=bc>=0;locmap[vv]=np.searchsorted(allchild,bc[vv])
  bycut=[]
  for j in range(15):
   idx=locmap[cp,j];ok=idx>=0;dd=np.full(len(cp),np.inf);dd[ok]=np.linalg.norm(sitexyz[ok]-allxyz[idx[ok]],axis=1);good=ok&(dd<=R+1e-12)
   mask[good]|=(1<<j);cw[good,j]=bc[cp[good],j];cd[good,j]=dd[good];bycut.append(int(good.sum()))
  keep=mask!=0;cp,cb,sitexyz,seedc,seedchild,sitekind,mask,cw,cd,seed_dist=(cp[keep],cb[keep],sitexyz[keep],seedc[keep],seedchild[keep],sitekind[keep],mask[keep],cw[keep],cd[keep],seed_dist[keep])
  if len(cp)==0:raise RuntimeError("adaptive candidate set has no routeable points")
  if np.any(cb<0) or np.max(np.abs(cb.sum(axis=1)-1))>2e-12:raise RuntimeError("adaptive barycentric replay failed")
  cand=out/"adaptive-sites-1046h.npz"
  np.savez_compressed(cand,source_parent_row=cp.astype(np.int32),barycentric=cb,source_point_world_m=sitexyz,seed_cut_index=seedc.astype(np.int16),seed_child_row=seedchild.astype(np.int32),site_kind=sitekind.astype(np.int8),route_mask=mask.astype(np.uint16),cut_witness_child_rows=cw,cut_witness_distance_m=cd,global_route=np.ones(len(cp),np.uint8),global_witness_child_row=seedchild.astype(np.int32),global_witness_distance_m=seed_dist)
  summary={"schema":"numi.h706.adaptive-source-tet-joint-lp-031-preparation.v1","status":"PASS_SOURCE_ONLY_ADAPTIVE_SITE_GENERATION","physical_admission":False,"native_steps":0,"allocator_run":False,
   "matrix_sha256":sha(MATRIX),"expanded_typed_support_sha256":EXPECTED_SUPPORT,"cut_names":CUT_NAMES,"cut_demands_kg":dem.tolist(),
   "030_preflight":{"path":str(preflight),"sha256":sha(preflight),"status":json.loads(preflight.read_text()).get("status"),"invocation_mode":"--preflight-only","physical_admission":False,"native_steps":0},
   "current11_reconciliation":current,"seed_parent_cut_pairs":len(pairp),"unique_seed_children":len(rows),"generated":gen,"owner_route_counts_by_cut":bycut,
   "adaptive_sites_npz":{"path":str(cand),"sha256":sha(cand),"bytes":cand.stat().st_size},
   "construction":"source_tetra_anchor_refinement_1046d","provenance_mode":"CurrentCarrierPointProjection","radius_m":R,
   "limitations":["Objective retains the 15 frozen cuts. The current 11 component targets are mapped but not separately constrained.",
    "Each adaptive route bit uses one pinned per-parent witness child for that cut, with distance replay. This is conservative, not an exhaustive route search.",
    "Source-only quadrature proposal; native allocator remains the acceptance measurement."]}
  (out/"preparation-report.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,sort_keys=True));return 0
 if a.prepared_dir is None:raise RuntimeError("solve requires --prepared-dir")
 prep=a.prepared_dir.resolve();prp=prep/"preparation-report.json";pd=json.loads(prp.read_text())
 cr=pd["adaptive_sites_npz"];pin(cr["path"],cr["sha256"],cr["bytes"])
 if pd.get("expanded_typed_support_sha256")!=EXPECTED_SUPPORT or pd.get("cut_names")!=CUT_NAMES:raise RuntimeError("prepared packet support/cut pins mismatch")
 with np.load(cr["path"],allow_pickle=False) as z:
  cp=z["source_parent_row"].astype(np.int64);cb=z["barycentric"];cm=z["route_mask"].astype(np.uint32);cw=z["cut_witness_child_rows"].astype(np.int64);cd=z["cut_witness_distance_m"]
  cpoint=z["source_point_world_m"];seedcut=z["seed_cut_index"];seedchild=z["seed_child_row"];seed_dist=z["global_witness_distance_m"];cgr=z["global_route"].astype(bool);cgw=z["global_witness_child_row"].astype(np.int64)
 if cb.shape!=(len(cp),4) or cw.shape!=(len(cp),15) or cd.shape!=(len(cp),15) or cgr.shape!=(len(cp),) or cgw.shape!=(len(cp),) or seed_dist.shape!=(len(cp),) or np.any(cp<0) or np.any(cp>=len(mass)):raise RuntimeError("adaptive sidecar shape/range mismatch")
 if not np.all(cgr) or not np.array_equal(cgw,seedchild): raise RuntimeError("adaptive primary global witness fields are incomplete")
 cxyz=np.einsum("ni,nij->nj",cb,tetxyz[cp])
 if np.any(cb<0) or np.max(np.abs(cb.sum(axis=1)-1))>2e-12 or np.max(np.abs(cxyz-cpoint))>2e-12:raise RuntimeError("adaptive source point does not replay")
 selected=np.unique(np.r_[cw[cw>=0],seedchild]);rows,childxyz,childmat,rolelocal=selected_child_geometry(selected,Path(named["child_skin_node"]["path"]),Path(named["child_skin_elements"]["path"]),Path(named["child_skin_labels"]["path"]),nchild)
 for j in range(15):
  valid=cw[:,j]>=0
  if np.any(valid):
   xyz=childxyz[np.searchsorted(rows,cw[valid,j])];actual=np.linalg.norm(cxyz[valid]-xyz,axis=1)
   if np.max(np.abs(actual-cd[valid,j]))>2e-12 or np.any(actual>R+1e-12):raise RuntimeError("adaptive distance failed solve-mode replay")
 actual_primary=np.linalg.norm(cxyz-childxyz[np.searchsorted(rows,seedchild)],axis=1)
 if np.max(np.abs(actual_primary-seed_dist))>2e-12 or np.any(actual_primary>R+1e-12): raise RuntimeError("adaptive primary global witness failed solve-mode replay")
 allp=np.r_[parents,cp];allb=np.vstack([baseb,cb]);allm=np.r_[route.astype(np.uint32),cm];globalroute=np.r_[ints["global_route"]!=0,cgr]
 eligible=np.arange(len(allp),dtype=np.int64)
 print(f"LP_START anchors={len(eligible)} parents={len(mass)} cuts=15 adaptive={len(cp)}",flush=True)
 t=time.monotonic();res,w,rep=lp_solve(mass,allp[eligible],allb[eligible],allm[eligible],globalroute[eligible],dem,a.time_limit_seconds);elapsed=time.monotonic()-t
 out.mkdir(parents=True)
 if len(w):
  wf=np.zeros(len(eligible));wf[:]=w
  posnew=np.flatnonzero(wf>0);selglobal=eligible[posnew];newmask=selglobal>=len(parents)
  selected_new=selglobal[newmask]-len(parents)
  op=out/"adaptive-joint-weight-proposal.npz"
  np.savez_compressed(op,source_parent_row=cp[selected_new].astype(np.int32),barycentric=cb[selected_new],weight=wf[posnew][newmask],
   source_point_world_m=cxyz[selected_new],seed_cut_index=seedcut[selected_new],
   seed_child_row=seedchild[selected_new],
   cut_witness_child_rows=cw[selected_new],cut_witness_distance_m=cd[selected_new],route_mask=cm[selected_new],
   global_route=cgr[selected_new].astype(np.uint8),global_witness_child_row=cgw[selected_new].astype(np.int32),global_witness_distance_m=seed_dist[selected_new],
   matrix_anchor_row_weights=w[:len(parents)],matrix_anchor_global_route=ints["global_route"].astype(np.uint8),
   matrix_anchor_global_witness_child_row=ints["global_witness_child_row"].astype(np.int32),matrix_anchor_global_witness_distance_m=fl["global_witness_distance_m"])
  output={"path":str(op),"sha256":sha(op),"bytes":op.stat().st_size,"positive_adaptive_rows":len(selected_new)}
 else: output=None
 current=json.loads((prep/"current11-target-reconciliation.json").read_text())
 report={"schema":"numi.h706.adaptive-source-tet-joint-slack-lp-031.v1","status":"PASS_SOURCE_ONLY_ADAPTIVE_SLACK_PROPOSAL" if res.success and rep["passed_parent_moments"] and rep["passed_slack_replay"] else "INCONCLUSIVE_SOURCE_ONLY_LP",
  "physical_admission":False,"native_steps":0,"allocator_run":False,"objective":{"cuts":15,"names":CUT_NAMES,"demands_kg":dem.tolist(),"margin_kg":MARGIN,"minimized":"sum normalized cut deficits"},
  "input_pins":{"030_execution_sha256":sha(EXEC),"030_run_manifest_sha256":sha(MANIFEST),"030_matrix_sha256":sha(MATRIX),"typed_support_sha256":EXPECTED_SUPPORT,
   "preparation_report":{"path":str(prp),"sha256":sha(prp)},"adaptive_sites_npz":cr},
  "solver":{"method":"SciPy HiGHS sparse LP","status":str(res.message),"status_code":int(res.status),"optimal":bool(res.success),"elapsed_seconds":elapsed,
   "anchor_variables":len(eligible),"base_matrix_rows":len(parents),"adaptive_candidate_rows":len(cp),"positive_adaptive_rows":int(len(selected_new)) if len(w) else 0},
  "replay":rep,"cut_results":[{"index":j,"name":CUT_NAMES[j],"demand_kg":float(dem[j]),"capacity_kg":rep.get("capacity_kg",[None]*15)[j],
    "shortfall_kg":rep.get("shortfall_kg",[None]*15)[j],"route_anchor_rows":int(np.count_nonzero((allm&(1<<j))!=0))} for j in range(15)],
  "current11_reconciliation":current,"output":output,
  "limitations":["15 frozen-cut source-only objective; current 11 component targets are not separate LP constraints.",
   "New candidate routes are witness-specific and conservative; native actual flow is required to measure progress.",
   "No geometry, source mass, density bounds, semantic policy, radius or native code changed."]}
 (out/"joint-lp-report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 exo={"status":report["status"],"physical_admission":False,"native_steps":0,"allocator_run":False,"report":{"path":str(out/"joint-lp-report.json"),"sha256":sha(out/"joint-lp-report.json")},"output":output,"elapsed_seconds":elapsed}
 (out/"execution.json").write_text(json.dumps(exo,indent=2,sort_keys=True)+"\n");print(json.dumps(exo,sort_keys=True));return 0 if report["status"].startswith("PASS") else 3

if __name__=="__main__":
 p=argparse.ArgumentParser(add_help=True);p.add_argument("--self-test",action="store_true");known,rest=p.parse_known_args()
 if known.self_test: self_test();raise SystemExit(0)
 raise SystemExit(main())

