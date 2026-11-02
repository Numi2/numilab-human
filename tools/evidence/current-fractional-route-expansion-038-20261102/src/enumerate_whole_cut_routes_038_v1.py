#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/actual-038-cut-driven-augmentation-001")
BASE_PATH=ROOT/"screen_cut_spare_routes_038_v6.py"
spec=importlib.util.spec_from_file_location("cut038v6",BASE_PATH)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
OUT=ROOT/"whole-cut-current-R-routes-v1"
OUT.mkdir(parents=True,exist_ok=True)
ROLE_NAMES={15:"Skin",16:"Fat"}
EPS=1e-12

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def verify_all_manifest_inputs():
 named=B.RUN_SPEC["named_input_pins"]
 out={}
 for name,item in sorted(named.items()):
  p=Path(item["path"])
  if not p.is_file():raise ValueError("missing named input "+name)
  got=sha(p)
  if got!=item["sha256"] or int(item.get("bytes",p.stat().st_size))!=p.stat().st_size:
   raise ValueError("named input hash/size mismatch "+name+" got="+got)
  out[name]={"path":str(p),"sha256":got,"bytes":p.stat().st_size}
 return out

def parse_child_geometry(labels_path,node_path,ele_path,nchild):
 labels=np.loadtxt(labels_path,delimiter=",",skiprows=1,usecols=(0,1,2,3),dtype=np.int64)
 if labels.shape!=(nchild,4) or not np.array_equal(labels[:,0],np.arange(nchild)):
  raise ValueError("child label row identity mismatch")
 node_header=Path(node_path).open().readline().split()
 ele_header=Path(ele_path).open().readline().split()
 if int(node_header[0])!=311768 or int(node_header[1])!=3 or int(ele_header[0])!=nchild or int(ele_header[1])!=4:
  raise ValueError("unexpected TetGen node/element header")
 nodes=np.loadtxt(node_path,skiprows=1,usecols=(0,1,2,3),dtype=np.float64)
 ele=np.loadtxt(ele_path,skiprows=1,usecols=(0,1,2,3,4),dtype=np.int64)
 if nodes.shape!=(311768,4) or ele.shape!=(nchild,5):raise ValueError("current 433 geometry shape mismatch")
 if not np.array_equal(nodes[:,0].astype(np.int64),np.arange(1,len(nodes)+1)):
  raise ValueError("TetGen node IDs are not contiguous one-based")
 if not np.array_equal(ele[:,0],np.arange(1,nchild+1)):
  raise ValueError("TetGen element IDs are not contiguous one-based")
 conn=ele[:,1:5]
 if conn.min()<1 or conn.max()>len(nodes):raise ValueError("element references invalid node ID")
 xyz=nodes[:,1:4]
 child_xyz=xyz[conn-1].mean(axis=1)
 if not np.all(np.isfinite(child_xyz)):raise ValueError("nonfinite child centroid")
 return labels,child_xyz,nodes,ele

def make_role_maps(labels,T,p1paths,gt,ng):
 tuples={tuple(map(int,t)):i for i,t in enumerate(gt)}
 if len(tuples)!=len(gt):raise ValueError("duplicate exact group tuple")
 role={}
 region_children={}
 allowed_parts=[]
 for mat in (15,16):
  rows=np.flatnonzero(labels[:,1]==mat).astype(np.int32)
  is_t=T[rows]
  inv=np.full(len(labels),-1,np.int32);inv[rows]=np.arange(len(rows),dtype=np.int32)
  with np.load(p1paths[mat],allow_pickle=False) as z:
   off=z["cell_admissible_region_offsets"].astype(np.int64)
   reg=z["cell_admissible_region_ids"].astype(np.int32)
   goff=z["cell_group_offsets"].astype(np.int64)
   gids=z["cell_group_ids"].astype(np.int32)
   local_tuples=z["group_tuples"].astype(np.int32)
   if len(off)!=len(rows)+1 or len(goff)!=len(rows)+1:raise ValueError("role P1 child CSR domain mismatch")
   if int(np.asarray(z["admissibility_max_source_region_hops"]).item())!=1:raise ValueError("role P1 is not the pinned one-hop contract")
   loc_to_gid=np.empty(len(local_tuples),np.int32)
   for i,t in enumerate(local_tuples):
    q=tuples.get(tuple(map(int,t)))
    if q is None:raise ValueError("role P1 exact tuple absent from current group registry")
    loc_to_gid[i]=q
   child_rep=np.repeat(np.arange(len(rows),dtype=np.int32),np.diff(goff))
   child_gids=loc_to_gid[gids]
   sel=is_t[child_rep]
   if np.any(sel):
    cg=rows[child_rep[sel]].astype(np.int64)
    gg=child_gids[sel].astype(np.int64)
    allowed_parts.append(cg*ng+gg)
   # Build exact one-hop T children by source region using the authentic P1 CSR.
   reg_rep=np.repeat(np.arange(len(rows),dtype=np.int32),np.diff(off))
   reg_sel=is_t[reg_rep]
   child_reg=rows[reg_rep[reg_sel]].astype(np.int32)
   source_reg=reg[reg_sel]
   if len(child_reg):
    order=np.argsort(source_reg,kind="stable")
    sr=source_reg[order]; cr=child_reg[order]
    cuts=np.r_[0,np.flatnonzero(sr[1:]!=sr[:-1])+1,len(sr)]
    for a,b in zip(cuts[:-1],cuts[1:]):
     r=int(sr[a]);vals=np.unique(cr[a:b])
     region_children[(mat,r)]=vals.astype(np.int32)
   role[mat]={"rows":rows,"inv":inv,"off":off,"regions":reg,"t_rows":rows[is_t]}
 return role,region_children,allowed_parts

def csr_has(role,mat,child,reg):
 local=int(role[mat]["inv"][child])
 if local<0:return False
 off=role[mat]["off"]; ids=role[mat]["regions"]
 a,b=int(off[local]),int(off[local+1])
 if b<=a:return False
 row=ids[a:b]
 j=int(np.searchsorted(row,reg))
 return j<len(row) and int(row[j])==int(reg)

def append_existing_csv_pairs(path,kind,T,labels,role,gtmap,ng,R,parts,counters):
 name_to_i=None
 with open(path,newline="") as f:
  rd=csv.DictReader(f)
  required=({"source_parent_row","receiver_child_tet_row","source_material","source_donor","source_region",
             "source_to_current_child_centroid_distance_m","one_hop_source_region","one_hop_max_source_region_hops",
             "receiver_current_material","receiver_current_donor","receiver_current_region"} if kind=="centroid" else
            {"source_material","source_donor","source_region","receiver_child_tet_row","exact_projection_distance_m","projection_status"} if kind=="projection" else
            {"record_kind","role","packet_carrier_row","child_global_row","material","donor","region","provenance_mode","distance_m"})
  if not required.issubset(rd.fieldnames or []):raise ValueError(f"{kind} CSV header mismatch")
  for row in rd:
   if kind=="witness":
    if row["record_kind"]=="centroid_route_fallback":continue
    c=int(row["child_global_row"]); mat=int(row["material"]); donor=int(row["donor"]); reg=int(row["region"]); d=float(row["distance_m"])
   else:
    c=int(row["receiver_child_tet_row"]);mat=int(row["source_material"]);donor=int(row["source_donor"]);reg=int(row["source_region"])
    d=float(row["source_to_current_child_centroid_distance_m"] if kind=="centroid" else row["exact_projection_distance_m"])
   if c<0 or c>=len(T):raise ValueError(f"{kind} child row outside current domain")
   if not T[c]:continue
   if labels[c,1]!=mat:raise ValueError(f"{kind} current target material mismatch on T row")
   if kind=="centroid":
    if int(row["receiver_current_material"])!=int(labels[c,1]) or int(row["receiver_current_donor"])!=int(labels[c,2]) or int(row["receiver_current_region"])!=int(labels[c,3]):
     raise ValueError("centroid slot current receiver labels disagree with current child")
    if int(row["one_hop_source_region"])!=reg or int(row["one_hop_max_source_region_hops"])!=1:
     raise ValueError("centroid slot one-hop evidence mismatch")
   elif kind=="projection":
    if row["projection_status"]!="PASS_EXACT_CURRENT_CENTROID_SOURCE_TUPLE_PROJECTION":
     continue
    if int(row["receiver_current_material"])!=mat:raise ValueError("projection child material mismatch")
   if not np.isfinite(d):raise ValueError(f"{kind} existing T witness has nonfinite distance")
   if kind!="witness" and d>R+EPS:raise ValueError(f"{kind} existing T witness exceeds current radius")
   if kind=="witness" and row["provenance_mode"] in ("CurrentCarrierPointProjection","source_tet_canonical_quadrature") and d>R+EPS: raise ValueError("current carrier-point witness exceeds current radius")
   gid=gtmap.get((mat,donor,reg))
   if gid is None:raise ValueError(f"{kind} tuple absent from current source group registry")
   if not csr_has(role,mat,c,reg):raise ValueError(f"{kind} existing witness violates authenticated one-hop P1 region CSR")
   parts.append(np.asarray([c*ng+gid],np.int64))
   counters[kind]+=1

def child_support_keys(Bmod,h,gt,P,cut_allow_by_child,T,labels,role,region_children,manifest_inputs):
 ng=len(gt);gtmap={tuple(map(int,t)):i for i,t in enumerate(gt)}
 if len(gtmap)!=ng:raise ValueError("exact group registry duplicate")
 parts=[]
 # P1 groups from each material role.
 role_paths={15:Path(Bmod.RUN_SPEC["named_input_pins"]["p1_skin_support"]["path"]),
             16:Path(Bmod.RUN_SPEC["named_input_pins"]["p1_fat_support"]["path"])}
 for mat in (15,16):
  rows=role[mat]["rows"];sel=T[rows]
  p=role_paths[mat]
  with np.load(p,allow_pickle=False) as z:
   off=z["cell_group_offsets"].astype(np.int64); gids=z["cell_group_ids"].astype(np.int32)
   tuples=z["group_tuples"].astype(np.int32)
   l2g=np.asarray([gtmap[tuple(map(int,t))] for t in tuples],np.int32)
   local=np.repeat(np.arange(len(rows),dtype=np.int32),np.diff(off))
   ok=sel[local]
   if np.any(ok):
    child=rows[local[ok]].astype(np.int64); group=l2g[gids[ok]].astype(np.int64)
    parts.append(child*ng+group)
 counters={"centroid":0,"projection":0,"witness":0}
 ni=Bmod.RUN_SPEC["named_input_pins"]
 append_existing_csv_pairs(Path(ni["source_centroid_slots"]["path"]),"centroid",T,labels,role,gtmap,ng,Bmod.R,parts,counters)
 append_existing_csv_pairs(Path(ni["projection_evidence"]["path"]),"projection",T,labels,role,gtmap,ng,Bmod.R,parts,counters)
 for mat,name in ((15,"carrier_witness_skin"),(16,"carrier_witness_fat")):
  append_existing_csv_pairs(Path(ni[name]["path"]),"witness",T,labels,role,gtmap,ng,Bmod.R,parts,counters)
 if not parts:raise ValueError("no existing exact-group support recovered")
 keys=np.unique(np.concatenate(parts))
 # Exact parity against native-bound allowlists for each residual child row.
 mismatch=[]; checked=0
 for c,expected in cut_allow_by_child.items():
  if not T[int(c)]:raise ValueError("residual child outside T while reconciling allowlist")
  a=int(np.searchsorted(keys,int(c)*ng));b=int(np.searchsorted(keys,(int(c)+1)*ng))
  got=(keys[a:b]-int(c)*ng).astype(np.int32)
  exp=np.asarray(expected,np.int32)
  checked+=1
  if not np.array_equal(got,exp):
   mismatch.append({"child":int(c),"expected":exp[:30].tolist(),"reconstructed":got[:30].tolist(),
                    "missing":np.setdiff1d(exp,got).tolist()[:20],"extra":np.setdiff1d(got,exp).tolist()[:20]})
   if len(mismatch)>=10:break
 if mismatch:raise ValueError("reconstructed current allowed-group CSR differs from full-cut residual rows: "+json.dumps({"checked":checked,"mismatches":mismatch}))
 return keys,counters,checked

def run():
 t0=time.time()
 basepin=B.check_pins()
 named=verify_all_manifest_inputs()
 h,gt,P,pxyz,pinit,pres,C,cxyz,dem,cut_allowed,cut_allow_by_child=B.parse_cut()
 nchild=int(h["child_count"]);ng=len(gt)
 T=B.mask_t(h,"child_mask_hex",nchild)
 reached=P[:,5].astype(bool)
 packets,gid,points,roleids,vcar,vlocal,vglobal,vbary,vweight,cross=B.crosswalk(P,pxyz,pinit,gt)
 labels,childxyz,nodes,ele=parse_child_geometry(B.CHILD_LABELS,B.CHILD_NODE,B.CHILD_ELE,nchild)
 residual_c=C[:,0].astype(np.int64)
 centroid_error=np.max(np.abs(childxyz[residual_c]-cxyz)) if len(residual_c) else 0.0
 if centroid_error>2e-9:raise ValueError(f"replayed current child centroids differ from cut rows: {centroid_error}")
 if int(T.sum())!=355193 or int(np.count_nonzero(reached))!=40603:raise ValueError("current full-cut mask counts differ from pinned 038")
 role,region_children,p1parts=make_role_maps(labels,T,{15:Path(B.RUN_SPEC["named_input_pins"]["p1_skin_support"]["path"]),16:Path(B.RUN_SPEC["named_input_pins"]["p1_fat_support"]["path"])},gt,ng)
 # Keep the context-sized key arrays from make_role_maps and add the large registered proof CSVs.
 keys,csv_counts,checked=child_support_keys(B,h,gt,P,cut_allow_by_child,T,labels,role,region_children,named)
 # Make an independent explicit P1 base list too (make_role_maps returned it); parity is implicit through union and C++ residual evidence.
 if not np.all(T[C[:,0]]):raise ValueError("residual child is outside unreached T")
 if np.any((roleids!=15)&(roleids!=16)):raise ValueError("unknown source material")
 source_by_gid={}
 reached_by_gid={}
 for g in range(ng):
  src=np.flatnonzero(reached&(gid==g)).astype(np.int32)
  if len(src):source_by_gid[g]=src
 existing_pair_residual=0
 for c,gs in cut_allow_by_child.items():
  a=np.searchsorted(keys,int(c)*ng);b=np.searchsorted(keys,(int(c)+1)*ng)
  existing_pair_residual+=int(b-a)
 if existing_pair_residual<sum(len(x) for x in cut_allow_by_child.values()):raise ValueError("residual cut support parity internal count failed")
 # Search every exact source group against every one-hop admissible T child of its material.
 candidate_child=[];candidate_gid=[];candidate_vrow=[];candidate_dist=[];candidate_arc_cap=[]
 route_source=np.zeros(len(P),bool)
 pair_count_by_mat={15:0,16:0};covered_children={15:set(),16:set()}
 arc_cap_by_mat={15:0.0,16:0.0};arc_cap_by_child_region={}
 arc_cap_by_source_group={}
 pair_dist=[]; minroute=None; forbidden_existing=[]
 allT_by_mat={m:role[m]["t_rows"] for m in (15,16)}
 for g,tup in enumerate(gt):
  mat,donor,sreg=map(int,tup)
  src=source_by_gid.get(g)
  if src is None:continue
  children=region_children.get((mat,sreg))
  if children is None or not len(children):continue
  tree=cKDTree(points[src])
  # Full one-hop region set is a per-child P1 CSR, not a same-region shortcut.
  for start in range(0,len(children),20000):
   child_chunk=children[start:start+20000]
   vrows,dist=B.nearest_stable(tree,src,points[src],childxyz[child_chunk])
   ok=dist<=B.R+EPS
   if not np.any(ok):continue
   cc=child_chunk[ok];vv=vrows[ok];dd=dist[ok]
   pairkeys=cc.astype(np.int64)*ng+g
   ix=np.searchsorted(keys,pairkeys);inrng=ix<len(keys);present=np.zeros(len(ix),bool);present[inrng]=keys[ix[inrng]]==pairkeys[inrng]
   if np.any(present):
    for i in np.flatnonzero(present)[:10]:
     forbidden_existing.append({"group_id":g,"child":int(cc[i]),"virtual_parent":int(vv[i]),"distance_m":float(dd[i])})
    raise ValueError("valid reached S-to-unreached T edge already exists in reconstructed native allowlist; cut/support reconciliation is wrong: "+json.dumps(forbidden_existing))
   # Exact cut crossing capacity uses every reached source point in this group within R of this T child.
   # This is an edge-capacity increment upper, not a feasible allocation.
   caps=np.empty(len(cc),np.float64)
   for j0 in range(0,len(cc),1000):
    j1=min(len(cc),j0+1000)
    neighborhoods=tree.query_ball_point(childxyz[cc[j0:j1]],r=B.R+EPS,workers=1)
    for k,near in enumerate(neighborhoods):
     if not near:raise ValueError("nearest-inside-radius candidate has no radius-neighbor source point")
     nearidx=np.asarray(near,np.int64);pv=src[nearidx]
     route_source[pv]=True
     caps[j0+k]=float(pinit[pv].sum())
   candidate_child.append(cc.astype(np.int32));candidate_gid.append(np.full(len(cc),g,np.int32))
   candidate_vrow.append(vv.astype(np.int32));candidate_dist.append(dd.astype(np.float64));candidate_arc_cap.append(caps)
   pair_count_by_mat[mat]+=len(cc);covered_children[mat].update(map(int,cc))
   arc_cap_by_mat[mat]+=float(caps.sum())
   pair_dist.append(dd.copy())
   srkey=str(sreg)
   arc_cap_by_source_group[srkey]=arc_cap_by_source_group.get(srkey,0.0)+float(caps.sum())
   for r in np.unique(labels[cc,3]):
    sel=labels[cc,3]==r;key=f"{mat}:{int(r)}"
    arc_cap_by_child_region[key]=arc_cap_by_child_region.get(key,0.0)+float(caps[sel].sum())
   if minroute is None or float(dd.min())<minroute:minroute=float(dd.min())
 if candidate_child:
  cc=np.concatenate(candidate_child);gg=np.concatenate(candidate_gid);vv=np.concatenate(candidate_vrow);dd=np.concatenate(candidate_dist);caps=np.concatenate(candidate_arc_cap)
 else:
  cc=np.empty(0,np.int32);gg=np.empty(0,np.int32);vv=np.empty(0,np.int32);dd=np.empty(0,np.float64);caps=np.empty(0,np.float64)
 if len(cc)!=len(np.unique(cc.astype(np.int64)*ng+gg.astype(np.int64))):raise ValueError("duplicate candidate exact group-child slots")
 # The full-cut residual children must be a subset of T and absent for every candidate pair; validate reconstructed rowwise support parity.
 candidate_children=set(map(int,cc))
 all_rows=len(cc)
 source_route_rows=np.flatnonzero(route_source).astype(np.int32)
 source_route_mass=float(pinit[source_route_rows].sum())
 # Write every missing exact group-child route as a compact, replayable proposal.
 proposal=OUT/"whole-cut-crossing-routes.npz"
 np.savez_compressed(proposal,
  child_global_row=cc,group_id=gg,virtual_parent_row=vv,distance_m=dd,added_cut_arc_capacity_kg=caps,
  packet_carrier_row=vcar[vv].astype(np.int32),source_global330_tet_row=vglobal[vv].astype(np.int32),
  source_local_parent_row=vlocal[vv].astype(np.int32),source_role_material=roleids[vv].astype(np.int32),
  source_barycentric=vbary[vv].astype(np.float64),source_weight=vweight[vv].astype(np.float64),
  source_point_xyz=points[vv].astype(np.float64),child_centroid_xyz=childxyz[cc].astype(np.float64),
  child_role_local_row=np.asarray([int(role[int(labels[c,1])]["inv"][c]) for c in cc],np.int32),
  source_exact_tuple=gt[gg].astype(np.int32),
  reached_source_virtual_rows=source_route_rows,
  reached_source_point_mass_kg=pinit[source_route_rows].astype(np.float64))
 # Report per role / child-label region. These are cut-arc capacities, not flow.
 child_rows_by_mat={m:int(np.count_nonzero(T&(labels[:,1]==m))) for m in (15,16)}
 child_rows_covered={m:int(len(covered_children[m])) for m in (15,16)}
 source_rows_by_mat={m:int(np.count_nonzero(route_source&(roleids==m))) for m in (15,16)}
 source_mass_by_mat={m:float(pinit[route_source&(roleids==m)].sum()) for m in (15,16)}
 arc_cap_by_mat={str(k):v for k,v in arc_cap_by_mat.items()}
 arc_cap_by_child_region={k:v for k,v in sorted(arc_cap_by_child_region.items())}
 region_counts={}
 for mat in (15,16):
  treg=np.unique(labels[T&(labels[:,1]==mat),3])
  for reg in treg:
   mask=T&(labels[:,1]==mat)&(labels[:,3]==reg)
   cov=sum(1 for c in covered_children[mat] if labels[c,3]==reg)
   key=f"{mat}:{int(reg)}"
   region_counts[key]={"T_children":int(mask.sum()),"covered_T_children":cov,
      "new_group_child_route_count":int(np.count_nonzero((labels[cc,1]==mat)&(labels[cc,3]==reg))) if len(cc) else 0,
      "added_cut_arc_capacity_kg":arc_cap_by_child_region.get(key,0.0)}
 # Verify all current residual allowlists exactly reconstructed above.
 report={"schema":"numi.h706.current-038-whole-cut-crossing-route-census.v1",
  "status":"WHOLE_T_CROSSING_ROUTE_CENSUS_PASS",
  "pins":{"run_manifest":{"path":str(B.MAN),"sha256":sha(B.MAN)},
   "current_cut_gzip":{"path":str(B.CUT),"sha256":sha(B.CUT)},
   "current_cut_manifest":{"path":str(B.CUT_MAN),"sha256":sha(B.CUT_MAN)},
   "current_cut_header":{"path":str(B.CUT_HEAD),"sha256":sha(B.CUT_HEAD)},
   "v6_producer":{"path":str(BASE_PATH),"sha256":sha(BASE_PATH)},
   "transport_envelope_receipt":{"path":B.RUN_SPEC["mass_transport_envelope_receipt"]["path"],"sha256":B.RUN_SPEC["mass_transport_envelope_receipt"]["sha256"]},
   "named_inputs":named},
  "geometry":{"transport_envelope_m":float(B.R),"centroid_replay_max_abs_error_m":float(centroid_error),
   "current_full_T_children":int(T.sum()),"T_skin_children":child_rows_by_mat[15],"T_fat_children":child_rows_by_mat[16],
   "reached_parent_rows":int(reached.sum()),"reached_Skin_rows":int(np.count_nonzero(reached&(roleids==15))),
   "reached_Fat_rows":int(np.count_nonzero(reached&(roleids==16)))},
  "allowlist_reconstruction":{"status":"PASS_EXACT_RESIDUAL_CHILD_PARITY",
   "residual_children_checked":int(checked),"support_pair_keys_reconstructed_for_T":int(len(keys)),
   "reconstructed_residual_group_links":int(existing_pair_residual),"proof_row_counts_T_only":csv_counts,
   "sources":["P1 cell_group_ids","current source-centroid slots","authenticated exact-tuple projections","pinned current carrier witness CSVs"]},
  "crossing_candidates":{"scope":"ALL current 038 unreached T children and every reached S virtual source carrier; exact source tuple material/donor/region; target material exact; source region must be in that target's authenticated P1 one-hop CSR; source point recomputed from current 032d packet barycentrics; distance to current 433 child reference centroid <= pinned R.",
   "valid_missing_exact_group_child_pairs":int(len(cc)),
   "candidate_T_children_covered":child_rows_covered,
   "T_children_covered_by_any_missing_route":int(len(candidate_children)),
   "T_children_without_new_exact_group_child_route":int(T.sum()-len(candidate_children)),
   "reached_source_points_with_at_least_one_T_route":int(len(source_route_rows)),
   "reached_source_point_mass_upper_kg":source_route_mass,
   "reached_source_point_mass_upper_by_material_kg":{str(k):v for k,v in source_mass_by_mat.items()},
   "reached_source_point_count_by_material":{str(k):v for k,v in source_rows_by_mat.items()},
   "new_cut_crossing_arc_capacity_sum_kg":{str(k):v for k,v in arc_cap_by_mat.items()},
   "new_cut_crossing_arc_capacity_total_kg":float(caps.sum()),
   "route_distance_min_m":None if minroute is None else minroute,
   "route_distance_quantiles_m":(np.quantile(dd,[0,.25,.5,.75,.9,.99,1]).tolist() if len(dd) else []),
   "route_pairs_by_source_tuple_region":{k:float(v) for k,v in sorted(arc_cap_by_source_group.items(),key=lambda x:int(x[0]))},
   "route_pairs_by_target_material_region":region_counts,
   "proposal_npz":{"path":str(proposal),"sha256":sha(proposal),"bytes":proposal.stat().st_size}},
  "cut":{"residual_child_rows":int(len(C)),"residual_demand_kg":float(dem.sum()),
   "unreached_T_rows":int(T.sum()),"reached_child_rows":int(h["reached_child_count"]),
   "minflow_sent_kg":float(h["sent_kg"]),"minflow_required_kg":float(h["minimum_required_kg"]),
   "residual_shortfall_kg":float(h["minimum_required_kg"])-float(h["sent_kg"])},
  "interpretation":"Every emitted pair is a missing exact group-child allowlist slot with a replayable reached-side source-point witness. The reported crossing capacity is the sum of current implicit parent-to-child arc capacities over all newly enabled reached-S to unreached-T arcs, before any allocator rerun. It is an exact min-cut capacity increment under these pair routes, not a feasible-flow or mass-conservation result.",
  "physical_admission":False,"elapsed_seconds":time.time()-t0}
 rp=OUT/"whole-cut-route-census-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2,sort_keys=True))
 return 0

if __name__=="__main__":
 if "--preflight" in sys.argv:
  print(json.dumps({"base_status":B.check_pins(),"all_named_inputs":verify_all_manifest_inputs()},indent=2,sort_keys=True))
 else: raise SystemExit(run())
