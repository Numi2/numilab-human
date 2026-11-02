#!/usr/bin/env python3
from __future__ import annotations
import csv,gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
A=H/"current-fractional-semantic-native-1047"
MAN=A/"postpass-030d-canonical-038/out-006/run-manifest.json"
RUN=MAN.parent
RUN_SPEC=json.loads(MAN.read_text())
OUT=A/"actual-038-cut-driven-augmentation-001"
RUN_OUTPUT=Path(RUN_SPEC["output_dir"])
CUT=RUN_OUTPUT/"capacity-witness/full-cut-witness.csv.gz"
CUT_MAN=RUN_OUTPUT/"capacity-witness/manifest.json"
CUT_HEAD=RUN_OUTPUT/"capacity-witness/full-cut-witness.header.txt"
V14_REPORT=A/"actual-037-existing-032d-points-current-envelope-001/current-envelope-point-support-report-v14.json"
V14=V14_REPORT.parent
OLD_HEAD=None
P1=None
PACKET={}
WITNESS={}
SRC_MASS=SRC_LABELS=SRC_NODES=SRC_TETS=None
CHILD_LABELS=CHILD_NODE=CHILD_ELE=None
R=None
SCALE=10**19
EXPECTED={
 "run_manifest":"30611953b537a9179c22228aae7bbd250316531035fcfff5b0148ff0ba6419ba",
 "cut_gzip":"6aa7a79dcc2df190d926cf93709112140c275ffb78395209cebc779215c14c22",
 "cut_header":"9d07ad49482beb3cdd8fd4f9628685292c279fab1d65621c408f86e628927c13",
 "old037_manifest":"0cb995b7f481e377543add3014532f20cac9c7f478c14d1c267e4db18a9d9b01",
 "v14_report":"76421da11426faa0327f99b8d1cbc6a47761e9ede19b42b1653eaa2fbca81081",
 "skin_packet":"7f648262440539761f5cc6f14c9a2a7c5bf73c5cf75b310ec747872d83c0efb3",
 "fat_packet":"12068b0a15bcc24dd8adae0d452ad6bf49b443a8a4b78712093aaa26713126a8",
 "source_mass":"c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679",
 "source_labels":"99d643f71a735ee49223cdcd7b863d2d0978b8cc490a138bad4a63012438a457",
 "source_nodes":"7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044",
 "source_tets":"82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992",
 "child_labels":"c48c944a0f4080ecc3df3d6430bf6f8ec8abca00c90c8b990ffbfdd72dc2156d",
 "child_node":"8f47890b86b3a71ce247caf69eff6e72cf4a7080af28a287b8f4cd40efa27424",
 "child_ele":"ca1ec9c3ac3afc8964dbddf92bc6db5d1c854e0dc873ed8c4f3dfe4b0c6addeb",
 "p1_skin":"86eeda29d80ee8e65651d044bdb6842a3d5cf1877966743c65a5144396ad685a",
 "old037_header":"aec0b41e4c7c64d914f9e8a1a23ba80f52c48ed259d213f60ef2217a04dd14a0",
 "transport_envelope_receipt":"d0285ea1716cbb26b8ddd4967ed6b266a92cb0ef0a0697a46484545b9c7ef224",
}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def bind_named_inputs():
 global P1,PACKET,WITNESS,SRC_MASS,SRC_LABELS,SRC_NODES,SRC_TETS,CHILD_LABELS,CHILD_NODE,CHILD_ELE,OLD_HEAD,R
 named=RUN_SPEC.get("named_input_pins")
 if not isinstance(named,dict):raise ValueError("038 manifest lacks named_input_pins")
 expected={"carrier_skin_packet":"skin_packet","carrier_fat_packet":"fat_packet",
  "carrier_witness_skin":None,"carrier_witness_fat":None,
  "source330_mass":"source_mass","source330_labels":"source_labels","source330_nodes":"source_nodes","source330_tets":"source_tets",
  "child_skin_labels":"child_labels","child_skin_node":"child_node","child_skin_elements":"child_ele","p1_skin_support":"p1_skin"}
 bound={}
 for name,expect_key in expected.items():
  item=named.get(name)
  if not isinstance(item,dict) or not item.get("path") or not item.get("sha256"):raise ValueError(f"038 manifest missing named input {name}")
  p=Path(item["path"]);got=sha(p)
  if got!=item["sha256"]:raise ValueError(f"038 manifest named input digest mismatch {name}: {got}")
  if expect_key and got!=EXPECTED[expect_key]:raise ValueError(f"038 named input differs from frozen task pin {name}: {got}")
  if int(item.get("bytes",p.stat().st_size))!=p.stat().st_size:raise ValueError(f"038 named input byte-size mismatch {name}")
  bound[name]=p
 P1=bound["p1_skin_support"]
 PACKET={15:bound["carrier_skin_packet"],16:bound["carrier_fat_packet"]}
 WITNESS={15:bound["carrier_witness_skin"],16:bound["carrier_witness_fat"]}
 SRC_MASS=bound["source330_mass"];SRC_LABELS=bound["source330_labels"];SRC_NODES=bound["source330_nodes"];SRC_TETS=bound["source330_tets"]
 CHILD_LABELS=bound["child_skin_labels"];CHILD_NODE=bound["child_skin_node"];CHILD_ELE=bound["child_skin_elements"]
 envpin=RUN_SPEC.get("mass_transport_envelope_receipt")
 if not isinstance(envpin,dict) or not envpin.get("path"):raise ValueError("038 manifest lacks transport-envelope receipt pin")
 envpath=Path(envpin["path"])
 envsha=sha(envpath)
 if envsha!=envpin.get("sha256") or envsha!=EXPECTED["transport_envelope_receipt"]:raise ValueError("transport-envelope receipt pin mismatch")
 env=json.loads(envpath.read_text())
 envelope=env.get("mass_transport_envelope",{})
 R=float(envelope.get("predeclared_transport_envelope_m",0.0))
 if envelope.get("schema")!="numi.human.mass-transport-envelope.v1" or abs(R-float(env.get("mass_mapping_maximum_transport_distance_m",-1)))>1e-15:
  raise ValueError("transport-envelope receipt schema/value mismatch")
 if abs(float(envelope.get("maximum_declared_source_centroid_distance_m",0.0))-0.1874423157229725)>1e-12 or R<=float(envelope["maximum_declared_source_centroid_distance_m"]):
  raise ValueError("transport-envelope ceiling does not match current declared source-centroid maximum")
 v14=json.loads(V14_REPORT.read_text())
 OLD_HEAD=Path(v14["pins"]["cut_header"]["path"])
 if sha(OLD_HEAD)!=v14["pins"]["cut_header"]["sha256"] or sha(OLD_HEAD)!=EXPECTED["old037_header"]:raise ValueError("037 source header pin mismatch")
 return bound
def fields(line):
 return dict(x.split("=",1) for x in line.strip().split() if "=" in x)
def section_rows(f,begin,end):
 for line in f:
  if line.strip()==begin:break
 else:raise ValueError("missing section "+begin)
 rd=csv.reader(f);hdr=next(rd)
 def iter_rows():
  for row in rd:
   if row and row[0]==end:return
   yield row
  raise ValueError("unterminated section "+begin)
 return hdr,iter_rows()
def mask_t(h,key,n):
 return ~np.unpackbits(np.frombuffer(bytes.fromhex(h[key]),np.uint8),bitorder="little")[:n].astype(bool)
def nearest_stable(tree,candidate_ids,candidate_xyz,query_xyz):
 candidate_ids=np.asarray(candidate_ids,np.int64);candidate_xyz=np.asarray(candidate_xyz,np.float64);query_xyz=np.asarray(query_xyz,np.float64)
 k=min(2,len(candidate_ids));d,ix=tree.query(query_xyz,k=k,workers=1)
 if k==1:d=np.asarray(d,np.float64).reshape(-1,1);ix=np.asarray(ix,np.int64).reshape(-1,1)
 else:d=np.asarray(d,np.float64);ix=np.asarray(ix,np.int64)
 chosen=ix[:,0].copy()
 tie=np.flatnonzero(np.abs(d[:,1]-d[:,0])<=1e-12) if k==2 else np.empty(0,np.int64)
 for i in tie:
  near=np.asarray(tree.query_ball_point(query_xyz[i],float(d[i,0])+1e-12),np.int64)
  if len(near)>1:
   exact=np.linalg.norm(candidate_xyz[near]-query_xyz[i],axis=1)
   order=np.lexsort((candidate_ids[near],exact));chosen[i]=near[order[0]]
 exact=np.linalg.norm(candidate_xyz[chosen]-query_xyz,axis=1)
 return candidate_ids[chosen],exact
def self_test():
 off=np.array([0,2,3,6],np.int64);counts=np.diff(off);base=np.r_[0,np.cumsum(counts[:-1])]
 got=np.concatenate([np.arange(base[i],base[i]+counts[i]) for i in range(len(counts))])
 assert got.tolist()==[0,1,2,3,4,5]
 g=coo_matrix((np.array([3,4,3,4,7],np.int64),(np.array([0,0,1,2,3]),np.array([1,2,3,3,4]))),shape=(5,5),dtype=np.int64).tocsr()
 assert int(maximum_flow(g,0,4,method="dinic").flow_value)==7
 xyz=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
 ids,dist=nearest_stable(cKDTree(xyz),np.array([8,2,5]),xyz,np.array([[0.,0.,0.]]))
 assert ids.tolist()==[2] and abs(float(dist[0])-1.)<1e-15
 print("self-test PASS: CSR expansion, integer Dinic, deterministic tie-broken nearest witness")
def check_pins():
 if sha(MAN)!=EXPECTED["run_manifest"]:raise ValueError("038 run manifest SHA mismatch")
 bound=bind_named_inputs()
 for key,p in [("cut_header",CUT_HEAD),("v14_report",V14_REPORT)]:
  got=sha(p)
  if got!=EXPECTED[key]:raise ValueError(f"pinned input mismatch {key}: {got}")
 if sha(CUT_HEAD)!=EXPECTED["cut_header"]:raise ValueError("038 cut header SHA mismatch")
 if sha(CUT)!=EXPECTED["cut_gzip"]:raise ValueError("038 cut gzip SHA mismatch")
 cm=json.loads(CUT_MAN.read_text())
 if cm.get("status")!="INFEASIBLE_FULL_GRAPH_CUT" or cm.get("witness_sha256")!=EXPECTED["cut_gzip"] or cm.get("header_sha256")!=EXPECTED["cut_header"]:raise ValueError("038 full-cut manifest mismatch")
 v14=json.loads((V14/"current-envelope-point-support-report-v14.json").read_text())
 if v14.get("actual037_manifest_sha256")!=EXPECTED["old037_manifest"] or abs(float(v14.get("maximum_transport_distance_m",0))-R)>1e-12:raise ValueError("v14 support report provenance/R mismatch")
 return v14
def parse_cut():
 with gzip.open(CUT,"rt",newline="") as f:
  first=f.readline();h=fields(first)
  if h.get("schema")!="numi.h706.full-implicit-residual-cut.v1" or h.get("phase")!="minimum_flow":raise ValueError("038 witness header schema/phase mismatch")
  gh,gr=section_rows(f,"group_summary_csv_begin","group_summary_csv_end")
  gt=np.zeros((int(h["exact_group_count"]),3),np.int32);ng=0
  for r in gr:gt[int(r[0])]=[int(r[1]),int(r[2]),int(r[3])];ng+=1
  if ng!=len(gt):raise ValueError("038 group row count mismatch")
  ph,pr=section_rows(f,"parent_group_rows_csv_begin","parent_group_rows_csv_end")
  parr=[];xarr=[];iarr=[];rarr=[]
  for r in pr:
   parr.append([int(r[0]),int(r[1]),int(r[2]),int(r[3]),int(r[4]),int(r[8])])
   xarr.append([float(v) for v in r[5:8]]);iarr.append(float(r[9]));rarr.append(float(r[10]))
  ch,cr=section_rows(f,"residual_child_rows_csv_begin","residual_child_rows_csv_end")
  carr=[];cxarr=[];darr=[];allow=[];allow_by_child={}
  for r in cr:
   carr.append([int(r[0]),int(r[1]),int(r[2]),int(r[3])])
   cxarr.append([float(v) for v in r[4:7]]);darr.append(float(r[7]))
   c=int(r[0]);ids=[]
   if r[9]:
    for s in r[9].split(";"):
     q=s.split(":");g=int(q[0]);t=tuple(map(int,q[1:4]))
     if g<0 or g>=len(gt) or tuple(map(int,gt[g]))!=t:raise ValueError("child allowlist group id/tuple mismatch")
     if t[0]!=int(r[1]):raise ValueError("child allowlist material mismatch")
     ids.append(g);allow.append((c,g))
   allow_by_child[c]=np.asarray(sorted(set(ids)),np.int32)
 if len(parr)!=int(h["parent_count"]) or len(carr)!=int(h["residual_child_rows"]):raise ValueError("038 witness section row count mismatch")
 P=np.asarray(parr,np.int64);xyz=np.asarray(xarr,np.float64);init=np.asarray(iarr,np.float64);res=np.asarray(rarr,np.float64)
 C=np.asarray(carr,np.int64);cxyz=np.asarray(cxarr,np.float64);dem=np.asarray(darr,np.float64)
 return h,gt,P,xyz,init,res,C,cxyz,dem,np.asarray(allow,np.int64),allow_by_child
def crosswalk(P,xyz,init,gt):
 sm=np.loadtxt(SRC_MASS,delimiter=",",skiprows=1,usecols=(0,2),dtype=np.float64)
 lab=np.loadtxt(SRC_LABELS,delimiter=",",skiprows=1,usecols=(0,1,2,3),dtype=np.int64)
 nodes=np.loadtxt(SRC_NODES,delimiter=",",skiprows=1,usecols=(0,5,6,7),dtype=np.float64)
 tets=np.loadtxt(SRC_TETS,delimiter=",",skiprows=1,usecols=(0,3,4,5,6),dtype=np.float64)
 if len(sm)!=28632 or len(lab)!=28632 or not np.array_equal(lab[:,0],np.arange(28632)):raise ValueError("source330 row domain mismatch")
 canon=np.asarray(sorted({tuple(map(int,r[1:4])) for r in lab}),np.int32)
 if not np.array_equal(canon,gt):raise ValueError("cut exact-group registry differs from source330")
 tetxyz=nodes[:,1:4][tets[:,1:5].astype(np.int64)]
 ng=len(gt);n=len(P);gid=np.empty(n,np.int32);pts=np.empty((n,3),np.float64);im=np.empty(n,np.float64);role=np.empty(n,np.int32)
 vcar=np.empty(n,np.int32);vlocal=np.empty(n,np.int32);vglobal=np.empty(n,np.int32);vbary=np.empty((n,4),np.float64);vweight=np.empty(n,np.float64)
 packets={};cursor=0
 for mat,name in ((15,"skin"),(16,"fat")):
  with np.load(PACKET[mat],allow_pickle=False) as z:
   sp=z["source_parent_tet_rows"].astype(np.int64);gp=z["global_source330_tet_rows"].astype(np.int64);off=z["parent_offsets"].astype(np.int64)
   w=z["carrier_weights"].astype(np.float64);b=z["carrier_barycentric"].astype(np.float64);cnt=np.diff(off)
   if len(sp)!=14316 or not np.all(cnt>0) or not np.array_equal(gp,sp+(0 if mat==15 else 14316)):raise ValueError("032d role-to-global source parent crosswalk mismatch")
   loc=np.repeat(np.arange(len(sp),dtype=np.int64),cnt)
   base=np.r_[0,np.cumsum(cnt[:-1])].astype(np.int64)
   v=cursor+np.repeat(base,cnt)+(np.arange(len(w))-off[loc])
   if len(v)!=(48573 if mat==15 else 23066):raise ValueError("032d source carrier count mismatch")
   gpar=gp[loc];pt=np.einsum("ij,ijk->ik",b,tetxyz[gpar]);tu=lab[gpar,1:4]
   tids={tuple(map(int,t)):i for i,t in enumerate(gt)}
   gi=np.fromiter((tids[tuple(map(int,t))] for t in tu),np.int32,count=len(tu))
   gid[v]=gi;pts[v]=pt;im[v]=sm[gpar,1]*w;role[v]=mat
   vcar[v]=np.arange(len(w),dtype=np.int32);vlocal[v]=sp[loc].astype(np.int32);vglobal[v]=gpar.astype(np.int32);vbary[v]=b;vweight[v]=w
   packets[mat]={"vrow":v.astype(np.int64),"global_parent":gpar.astype(np.int32),"local_parent":sp[loc].astype(np.int32),
                 "bary":b,"weight":w,"target":z["carrier_target_child_rows"].astype(np.int64)}
   cursor+=len(w)
 if cursor!=n or not np.array_equal(np.sort(np.concatenate([packets[m]["vrow"] for m in (15,16)])),np.arange(n)):raise ValueError("C++ expanded carrier row mapping misses/duplicates parent rows")
 if not np.array_equal(P[:,2],role) or not np.array_equal(P[:,1],gid) or not np.array_equal(P[:,2:5],gt[gid]):raise ValueError("038 expanded parent row identity does not match packets")
 xyzerr=float(np.max(np.abs(xyz-pts)));merr=float(np.max(np.abs(init-im)))
 if xyzerr>2e-13 or merr>1e-17:raise ValueError(f"C++ expansion replay mismatch xyz={xyzerr} mass={merr}")
 return packets,gid,pts,role,vcar,vlocal,vglobal,vbary,vweight,{"max_abs_xyz_error_m":xyzerr,"max_abs_initial_mass_error_kg":merr}
def child_regions(C):
 labels=np.loadtxt(CHILD_LABELS,delimiter=",",skiprows=1,usecols=(0,1),dtype=np.int64)
 cm=labels[:,1];rows=np.flatnonzero(cm==15);inv=np.full(len(labels),-1,np.int32);inv[rows]=np.arange(len(rows),dtype=np.int32)
 with np.load(P1,allow_pickle=False) as z:
  off=z["cell_admissible_region_offsets"].astype(np.int64);ids=z["cell_admissible_region_ids"].astype(np.int64)
  if len(off)!=len(rows)+1 or int(np.asarray(z["admissibility_max_source_region_hops"]).item())!=1:raise ValueError("P1 one-hop CSR contract mismatch")
 region_keys=[];regs_by_child={}
 for c in C[:,0]:
  loc=int(inv[int(c)])
  if loc<0:raise ValueError("residual child not in Skin role-local ordering")
  rr=ids[off[loc]:off[loc+1]]
  if not len(rr) or np.any(rr[1:]<=rr[:-1]):raise ValueError("bad residual-child region CSR row")
  regs_by_child[int(c)]=rr.astype(np.int16)
  region_keys.extend(int(c)*256+int(r) for r in rr)
 return cm,inv,np.unique(np.asarray(region_keys,np.int64)),regs_by_child
def run():
 tic=time.time();OUT.mkdir(parents=True,exist_ok=True)
 v14=check_pins()
 h,gt,P,pxyz,pinit,pres,C,cxyz,dem,allowed,allow_by_child=parse_cut()
 nP=len(P);nC=int(h["child_count"]);ng=len(gt)
 if len(C)!=103978 or abs(dem.sum()-float(h["residual_child_kg"]))>1e-12:raise ValueError("038 residual demand/pin mismatch")
 nowT=mask_t(h,"child_mask_hex",nC);oldh=fields(OLD_HEAD.read_text());oldT=mask_t(oldh,"child_mask_hex",nC)
 if int(nowT.sum())!=355193 or int(oldT.sum())!=405517 or np.any(nowT&~oldT):raise ValueError("038/037 T-domain relationship changed")
 if not np.all(nowT[C[:,0]]):raise ValueError("residual child outside current T")
 packets,gid,points,role,vcar,vlocal,vglobal,vbary,vweight,cross=crosswalk(P,pxyz,pinit,gt)
 reached=P[:,5].astype(bool)
 reached_skin=reached&(role==15);reached_fat=reached&(role==16)
 if np.any(pres>0.0) and np.any((pres>0.0)&~reached):raise ValueError("positive source residual outside reached-parent mask")
 if not np.all(np.isfinite(pres)) or not np.all(np.isfinite(pinit)) or np.any(pres<0) or np.any(pinit<=0):raise ValueError("invalid source mass in cut rows")
 cm,child_local,regkeys,regs_by_child=child_regions(C)
 if np.any(C[:,1]!=15) or np.any(cm[C[:,0]]!=15):raise ValueError("current residual witness is not Skin-only")
 childrow_to_resid=np.full(nC,-1,np.int32);childrow_to_resid[C[:,0]]=np.arange(len(C),dtype=np.int32)
 source_by_gid={g:np.flatnonzero(reached_skin&(gid==g)) for g in np.unique(gid[reached_skin])}
 trees={g:cKDTree(points[src]) for g,src in source_by_gid.items()}
 source_count_by_region={};source_mass_by_region={}
 for reg in np.unique(P[reached_skin,4]):
  m=reached_skin&(P[:,4]==reg);source_count_by_region[str(int(reg))]=int(np.count_nonzero(m));source_mass_by_region[str(int(reg))]=float(pinit[m].sum())
 unmet_by_region={}
 for reg in np.unique(C[:,3]):
  m=C[:,3]==reg;unmet_by_region[str(int(reg))]={"child_rows":int(np.count_nonzero(m)),"unmet_demand_kg":float(dem[m].sum())}
 existing_by_region={};existing_cross=[]
 for g in range(ng):
  pairs=allowed[allowed[:,1]==g];src=source_by_gid.get(g)
  if not len(pairs) or src is None:continue
  cs=pairs[:,0].astype(np.int64);ci=childrow_to_resid[cs]
  valid=ci>=0
  cs=cs[valid];ci=ci[valid]
  if not len(cs):continue
  tree=trees[g]
  vrows,dist=nearest_stable(tree,src,points[src],cxyz[ci])
  reg=str(int(gt[g,2]));e=existing_by_region.setdefault(reg,{"existing_allowed_group_child_pairs":0,"existing_pairs_with_reached_point_within_R":0,"minimum_reached_point_distance_m":float("inf")})
  e["existing_allowed_group_child_pairs"]+=int(len(cs));e["existing_pairs_with_reached_point_within_R"]+=int(np.count_nonzero(dist<=R-1e-12))
  e["minimum_reached_point_distance_m"]=min(e["minimum_reached_point_distance_m"],float(dist.min(initial=np.inf)))
  for j in np.flatnonzero(dist<=R-1e-12)[:10]:
   existing_cross.append({"group_id":int(g),"child_global_row":int(cs[j]),"virtual_parent_row":int(vrows[j]),"distance_m":float(dist[j])})
 allowed_key=np.unique(allowed[:,0].astype(np.int64)*ng+allowed[:,1].astype(np.int64))
 children_by_region={}
 for c in C[:,0]:
  for reg in regs_by_child[int(c)]:children_by_region.setdefault(int(reg),[]).append(int(c))
 groups_by_region={}
 for g,t in enumerate(gt):
  if int(t[0])==15:groups_by_region.setdefault(int(t[2]),[]).append(g)
 missing_by_region={};covered=np.zeros(len(C),bool);best_dist=np.full(len(C),np.inf)
 best_gid=np.full(len(C),np.iinfo(np.int32).max,np.int32);best_v=np.full(len(C),-1,np.int32)
 reached_route_parent=np.zeros(nP,bool);route_pair_count=0
 for reg,groups in groups_by_region.items():
  children=np.asarray(children_by_region.get(reg,()),np.int32)
  if not len(children):continue
  child_resid=childrow_to_resid[children].astype(np.int64)
  for g in groups:
   src=source_by_gid.get(g)
   if src is None:continue
   full_key=children.astype(np.int64)*ng+g
   j=np.searchsorted(allowed_key,full_key);exists=j<len(allowed_key);exists[exists]=allowed_key[j[exists]]==full_key[exists]
   missing_children=children[~exists];missing_resid=child_resid[~exists]
   if not len(missing_children):continue
   e=missing_by_region.setdefault(str(reg),{"missing_onehop_group_child_pairs_tested":0,"missing_pairs_with_reached_point_within_R":0,"minimum_reached_point_distance_m":float("inf"),"covered_children":0,"covered_unmet_demand_kg":0.0})
   e["missing_onehop_group_child_pairs_tested"]+=int(len(missing_children))
   tree=trees[g];srcxyz=points[src]
   # Reverse direction gives the complete reached-source-point upper mass for this group's missing child links.
   target_tree=cKDTree(cxyz[missing_resid])
   nearest_child,point_dist=nearest_stable(target_tree,missing_children,cxyz[missing_resid],srcxyz)
   point_ok=point_dist<=R-1e-12
   reached_route_parent[src[point_ok]]=True
   for start in range(0,len(missing_children),20000):
    stop=min(start+20000,len(missing_children));children_chunk=missing_children[start:stop];resid_chunk=missing_resid[start:stop]
    vrows,dist=nearest_stable(tree,src,srcxyz,cxyz[resid_chunk])
    e["minimum_reached_point_distance_m"]=min(e["minimum_reached_point_distance_m"],float(dist.min(initial=np.inf)))
    ok=dist<=R-1e-12
    if not np.any(ok):continue
    child_ok=children_chunk[ok];resid_ok=resid_chunk[ok];v_ok=vrows[ok];d_ok=dist[ok]
    e["missing_pairs_with_reached_point_within_R"]+=int(len(child_ok))
    route_pair_count+=int(len(child_ok));covered[resid_ok]=True
    current=best_dist[resid_ok]
    better=(d_ok<current-1e-12)|((np.abs(d_ok-current)<=1e-12)&((g<best_gid[resid_ok])|((g==best_gid[resid_ok])&(v_ok<best_v[resid_ok]))))
    if np.any(better):
     rr=resid_ok[better];best_dist[rr]=d_ok[better];best_gid[rr]=g;best_v[rr]=v_ok[better]
 if not np.array_equal(covered,best_v>=0):raise ValueError("route coverage / best-witness set mismatch")
 reachable_rows=np.flatnonzero(reached_route_parent).astype(np.int32)
 reachable_mass=float(pinit[reachable_rows].sum())
 reachable_by_region={}
 for reg in np.unique(P[reachable_rows,4]) if len(reachable_rows) else ():
  m=P[reachable_rows,4]==reg;reachable_by_region[str(int(reg))]={"source_points":int(np.count_nonzero(m)),"initial_carrier_mass_kg":float(pinit[reachable_rows[m]].sum())}
 covered_demand=float(dem[covered].sum());uncovered_demand=float(dem[~covered].sum())
 uncovered_by_region={}
 for reg in np.unique(C[~covered,3]):
  m=(C[:,3]==reg)&~covered;uncovered_by_region[str(int(reg))]={"child_rows":int(np.count_nonzero(m)),"unmet_demand_kg":float(dem[m].sum())}
 for reg,e in missing_by_region.items():
  m=(C[:,3]==int(reg))&covered;e["covered_children"]=int(np.count_nonzero(m));e["covered_unmet_demand_kg"]=float(dem[m].sum())
 regset=set(unmet_by_region)|set(source_mass_by_region)|set(existing_by_region)|set(missing_by_region)|set(reachable_by_region)
 region_rows={}
 for reg in sorted(regset,key=int):
  e=existing_by_region.get(reg,{"existing_allowed_group_child_pairs":0,"existing_pairs_with_reached_point_within_R":0,"minimum_reached_point_distance_m":None})
  m=missing_by_region.get(reg,{"missing_onehop_group_child_pairs_tested":0,"missing_pairs_with_reached_point_within_R":0,"minimum_reached_point_distance_m":None,"covered_children":0,"covered_unmet_demand_kg":0.0})
  ev=e.get("minimum_reached_point_distance_m",np.inf);mv=m.get("minimum_reached_point_distance_m",np.inf)
  if ev is None or not np.isfinite(float(ev)):e["minimum_reached_point_distance_m"]=None
  if mv is None or not np.isfinite(float(mv)):m["minimum_reached_point_distance_m"]=None
  region_rows[reg]={"unmet":unmet_by_region.get(reg,{"child_rows":0,"unmet_demand_kg":0.0}),
   "reached_skin_source_points":source_count_by_region.get(reg,0),"reached_skin_initial_mass_kg":source_mass_by_region.get(reg,0.0),
   "reached_points_with_any_missing_T_route":reachable_by_region.get(reg,{"source_points":0,"initial_carrier_mass_kg":0.0}),
   "existing_allowed_routes":e,"missing_new_group_routes":m,
   "uncovered_unmet":uncovered_by_region.get(reg,{"child_rows":0,"unmet_demand_kg":0.0})}
 selected=np.flatnonzero(covered)
 witness=OUT/"reached-parent-route-witnesses.npz"
 np.savez_compressed(witness,
  child_global_row=C[selected,0].astype(np.int32),child_role_local_row=child_local[C[selected,0]].astype(np.int32),
  child_region=C[selected,3].astype(np.int32),group_id=best_gid[selected].astype(np.int32),
  virtual_parent_row=best_v[selected].astype(np.int32),packet_carrier_row=vcar[best_v[selected]].astype(np.int32),
  global_source330_tet_row=vglobal[best_v[selected]].astype(np.int32),local_source_parent_row=vlocal[best_v[selected]].astype(np.int32),
  source_barycentric=vbary[best_v[selected]].astype(np.float64),source_xyz=points[best_v[selected]].astype(np.float64),
  child_xyz=cxyz[childrow_to_resid[C[selected,0]]].astype(np.float64),distance_m=best_dist[selected].astype(np.float64),
  initial_carrier_mass_kg=pinit[best_v[selected]].astype(np.float64),carrier_weight=vweight[best_v[selected]].astype(np.float64),
  reachable_virtual_parent_rows=reachable_rows,reachable_virtual_parent_initial_mass_kg=pinit[reachable_rows].astype(np.float64))
 report={"schema":"numi.h706.current-038-reached-parent-route-census.v1","status":"REACHED_PARENT_CUT_FRONTIER_GEOMETRY_CENSUS",
  "scope":"complete exact-group missing-link enumeration for the 103,978 residual Skin child rows, using every reached Skin source carrier point and the pinned per-child one-hop region CSR at the 0.187444 m envelope. Source weights, mass, geometry, labels, P1 scores and density limits are unchanged.",
  "pins":{"current_run_manifest":pin(MAN),"transport_envelope_receipt":pin(Path(RUN_SPEC["mass_transport_envelope_receipt"]["path"])),
   "current_cut_manifest":pin(CUT_MAN),"current_cut_header":pin(CUT_HEAD),"current_cut_gzip":pin(CUT),
   "v14_support_report":pin(V14_REPORT),"skin_packet":pin(PACKET[15]),"fat_packet":pin(PACKET[16]),"skin_p1":pin(P1)},
  "cut":{"typed_support_sha256":h["typed_support_sha256"],"semantic_support_sha256":h["semantic_support_sha256"],
   "reached_virtual_parents":int(h["reached_parent_count"]),"reached_children":int(h["reached_child_count"]),
   "residual_source_kg":float(h["residual_source_kg"]),"residual_child_rows":int(len(C)),"residual_child_demand_kg":float(dem.sum()),
   "current_T_rows":int(nowT.sum()),"old037_T_rows":int(oldT.sum()),"current_T_outside_old037_T":int(np.count_nonzero(nowT&~oldT))},
  "crosswalk":{"status":"PASS","expanded_parent_rows":int(nP),"reached_skin_points":int(np.count_nonzero(reached_skin)),
   "reached_fat_points":int(np.count_nonzero(reached_fat)),"max_abs_xyz_error_m":cross["max_abs_xyz_error_m"],
   "max_abs_initial_mass_error_kg":cross["max_abs_initial_mass_error_kg"]},
  "route_screen":{"transport_envelope_m":R,"existing_allowed_group_child_pairs_checked":int(sum(x["existing_allowed_group_child_pairs"] for x in existing_by_region.values())),
   "existing_allowed_pairs_with_reached_point_within_R":int(sum(x["existing_pairs_with_reached_point_within_R"] for x in existing_by_region.values())),
   "missing_onehop_group_child_pairs_tested":int(sum(x["missing_onehop_group_child_pairs_tested"] for x in missing_by_region.values())),
   "missing_group_child_pairs_with_reached_point_within_R":route_pair_count,
   "residual_children_with_at_least_one_new_route":int(np.count_nonzero(covered)),
   "residual_children_without_new_route":int(np.count_nonzero(~covered)),
   "residual_demand_covered_by_new_routes_kg":covered_demand,"residual_demand_without_new_route_kg":uncovered_demand,
   "reached_source_point_initial_mass_upper_kg_for_new_routes":reachable_mass,
   "reached_source_points_with_at_least_one_new_route":int(len(reachable_rows)),
   "reached_source_point_mass_upper_by_region_kg":{k:v["initial_carrier_mass_kg"] for k,v in reachable_by_region.items()},
   "per_region":region_rows,"existing_pair_examples":existing_cross[:10]},
  "deterministic_best_witnesses":{"schema":"reached-parent-route-witnesses.npz","rows_one_nearest_new_route_per_covered_unmet_child":int(len(selected)),
   "selection":"minimum float64 source-point distance; ties within 1e-12 m break by exact group id then virtual parent row",
   "artifact":pin(witness)},
  "interpretation":"The reached-source mass is an optimistic upper over source carrier mass whose fixed point can connect to at least one unmet child through a newly admitted exact group-child pair. It ignores residual upstream flow bottlenecks and child competition, so it is not a feasible allocation or max-flow proof. No canonical allocator was rerun.",
  "physical_admission":False,"elapsed_seconds":time.time()-tic}
 rp=OUT/"reached-parent-route-census-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2,sort_keys=True));return 0
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if "--self-test" in sys.argv:self_test();return 0
 if "--preflight" in sys.argv:
  check_pins()
  named=RUN_SPEC["named_input_pins"]
  print(json.dumps({"status":"INPUT_PREFLIGHT_PASS","run_manifest":pin(MAN),"named_input_pins":named,
   "cut_header":pin(CUT_HEAD),"cut_manifest":pin(CUT_MAN),"v14_report":pin(V14_REPORT)},indent=2,sort_keys=True))
  return 0
 return run()
if __name__=="__main__":raise SystemExit(main())

