#!/usr/bin/env python3
from __future__ import annotations

import csv, gzip, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
A=BASE/"current-fractional-semantic-native-1047"
RDIR=A/"actual-038-cut-driven-augmentation-001/whole-cut-current-R-routes-v1"
RUN=A/"postpass-030d-canonical-038/out-006"
RUNMAN=RUN/"run-manifest.json"
CENSUS=RDIR/"whole-cut-route-census-report.json"
NPZ=RDIR/"whole-cut-crossing-routes.npz"
PRODUCER=RDIR.parent/"enumerate_whole_cut_routes_038_v1.py"
BASEPRODUCER=RDIR.parent/"screen_cut_spare_routes_038_v6.py"
OUT=RDIR/"skin-witnesses-currentR-v15plus-whole-cut-routes-038.csv"
PART=RDIR/"skin-witnesses-currentR-v15plus-whole-cut-routes-038.csv.partial"
REPORT=RDIR/"whole-cut-route-skin-append-validation.json"
PINS={
 "run":"30611953b537a9179c22228aae7bbd250316531035fcfff5b0148ff0ba6419ba",
 "census":"d5d47d1410b0a5c0e3e91a29a3c1d51f9bb82e22647ffd82ce7c973d41726ffd",
 "npz":"6ba416a2f06261e2e002b0727b1e0497ce447ddca6db7f7466092111ef5a4009",
 "producer":"cee02a011d27e1fad13f7c66b91aefa19a9bd8233f7e7b6836146695234ca069",
 "baseproducer":"ae205ea5516771bbc24930c49ba290c1390e7f05f76684599d10f27519f10e94",
 "skin":"18256078155e0a1772bb1479644636ef4ef080a912ab98101a44f2e8e2b7140d",
 "fat":"86ddf947327c6ed692455bfac6f07a01404b6cf2ca76741be0ffc44d27cfefa1",
}
HEADER=["record_kind","role","carrier_construction","packet_carrier_row","local_parent_row",
"global_source330_tet_row","child_local_row","child_global_row","material","donor","region",
"provenance_mode","p1_group_id","projection_csv_row","bary0","bary1","bary2","bary3",
"weight","source_x_m","source_y_m","source_z_m","child_centroid_x_m","child_centroid_y_m",
"child_centroid_z_m","distance_m"]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def pin(path):
 p=Path(path).resolve()
 return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}

def cut_header_and_groups(path):
 with gzip.open(path,"rt",newline="") as f:
  fields=dict(part.split("=",1) for part in f.readline().strip().split() if "=" in part)
  if fields.get("schema")!="numi.h706.full-implicit-residual-cut.v1" or fields.get("phase")!="minimum_flow":
   raise ValueError("wrong pinned full-cut schema/phase")
  if f.readline().strip()!="group_summary_csv_begin": raise ValueError("cut group section missing")
  rd=csv.reader(f); hdr=next(rd)
  if hdr[:4]!=["group","material","donor","region"]: raise ValueError("cut group header mismatch")
  rows=[]
  for r in rd:
   if r and r[0]=="group_summary_csv_end": break
   if r: rows.append(tuple(map(int,r[:4])))
  else: raise ValueError("cut group section unterminated")
 if len(rows)!=int(fields["exact_group_count"]) or [r[0] for r in rows]!=list(range(len(rows))):
  raise ValueError("cut group domain mismatch")
 return fields,np.asarray([r[1:] for r in rows],np.int32)

def route_row(a,i):
 row=[
  "carrier_child_slot","Skin","current_source_carrier_edge_support",
  int(a["packet_carrier_row"][i]),int(a["source_local_parent_row"][i]),
  int(a["source_global330_tet_row"][i]),int(a["child_role_local_row"][i]),
  int(a["child_global_row"][i]),int(a["source_exact_tuple"][i,0]),
  int(a["source_exact_tuple"][i,1]),int(a["source_exact_tuple"][i,2]),
  "CurrentCarrierPointProjection",-1,-1,
  *[format(float(x),".17g") for x in a["source_barycentric"][i]],
  format(float(a["source_weight"][i]),".17g"),
  *[format(float(x),".17g") for x in a["source_point_xyz"][i]],
  *[format(float(x),".17g") for x in a["child_centroid_xyz"][i]],
  format(float(a["distance_m"][i]),".17g")]
 if len(row)!=26: raise AssertionError("not a 26-column row")
 return row

class DigestWriter:
 def __init__(self):
  self.digest=hashlib.sha256(); self.bytes=0
 def write(self,text):
  data=text.encode("utf-8"); self.digest.update(data); self.bytes+=len(data); return len(text)

def main():
 if PART.exists() or REPORT.exists(): raise FileExistsError("refuse overwrite existing partial/report")
 recovering_existing_output=OUT.exists()
 run=json.loads(RUNMAN.read_text()); rep=json.loads(CENSUS.read_text())
 if sha(RUNMAN)!=PINS["run"] or sha(CENSUS)!=PINS["census"] or sha(NPZ)!=PINS["npz"]:
  raise ValueError("run/census/proposal pin mismatch")
 if sha(PRODUCER)!=PINS["producer"] or sha(BASEPRODUCER)!=PINS["baseproducer"]:
  raise ValueError("route producer pin mismatch")
 if rep.get("status")!="WHOLE_T_CROSSING_ROUTE_CENSUS_PASS_REPORT_RECOVERED" or rep.get("physical_admission") is not False:
  raise ValueError("route census status is not the expected non-admitted recovered report")
 recovery=rep["report_recovery"]
 if recovery.get("producer_exit_status")!=1: raise ValueError("expected report-only source failure was not retained")
 if rep["proposal_artifact"].get("sha256")!=PINS["npz"] or rep["proposal_artifact"].get("rows")!=300860:
  raise ValueError("proposal/report pin mismatch")
 al=rep["allowlist_reconstruction"]
 if (al.get("status")!="PASS_EXACT_PARITY_BEFORE_PROPOSAL_WRITE"
     or al.get("candidate_pairs_checked_against_reconstructed_allowlist")!=300860
     or al.get("candidate_pair_already_present")!=0 or al.get("residual_children_checked")!=103978):
  raise ValueError("full prior allowlist parity evidence is incomplete")
 named=run["named_input_pins"]; pre=rep["all_22_manifest_named_inputs_preflight"]
 if set(named)!=set(pre) or len(named)!=22: raise ValueError("22-input source preflight roster mismatch")
 for k,v in named.items():
  if (v.get("path"),v.get("sha256"),v.get("bytes"))!=(pre[k].get("path"),pre[k].get("sha256"),pre[k].get("bytes")):
   raise ValueError("source preflight pin mismatch "+k)
 pskin=named["carrier_witness_skin"]; pfat=named["carrier_witness_fat"]
 if pskin["sha256"]!=PINS["skin"] or pfat["sha256"]!=PINS["fat"]: raise ValueError("v15 CSV pins changed")
 skin=Path(pskin["path"]); fat=Path(pfat["path"])
 skinpin=pin(skin)
 if skinpin["sha256"]!=PINS["skin"] or sha(fat)!=PINS["fat"]: raise ValueError("v15 CSV bytes changed")
 with skin.open("r",newline="",encoding="utf-8") as f:
  if next(csv.reader(f))!=HEADER: raise ValueError("v15 CSV header is not the exact 26-column contract")
 with skin.open("rb") as f:
  f.seek(-1,os.SEEK_END)
  if f.read(1)!=b"\n": raise ValueError("v15 CSV does not end on a record boundary")

 with np.load(NPZ,allow_pickle=False) as z: a={k:z[k] for k in z.files}
 if set(a)!=set(rep["proposal_artifact"]["arrays"]): raise ValueError("proposal NPZ key roster mismatch")
 n=300860
 vec=["child_global_row","group_id","virtual_parent_row","distance_m","added_cut_arc_capacity_kg",
 "packet_carrier_row","source_global330_tet_row","source_local_parent_row","source_role_material",
 "source_weight","child_role_local_row"]
 if any(a[k].shape!=(n,) for k in vec): raise ValueError("route vector shape mismatch")
 if a["source_barycentric"].shape!=(n,4) or a["source_point_xyz"].shape!=(n,3) or a["child_centroid_xyz"].shape!=(n,3) or a["source_exact_tuple"].shape!=(n,3):
  raise ValueError("route matrix shape mismatch")
 if any(v.dtype.kind in "fc" and not np.all(np.isfinite(v)) for v in a.values()):
  raise ValueError("route NPZ has nonfinite numeric values")
 if np.any(a["source_weight"]<=0) or np.any(a["distance_m"]<0) or np.any(a["added_cut_arc_capacity_kg"]<0):
  raise ValueError("route includes invalid weight/distance/capacity")

 cut=Path(rep["pins"]["cut_gzip"]["path"])
 ch,cgroups=cut_header_and_groups(cut)
 if len(cgroups)!=int(ch["exact_group_count"]): raise ValueError("group registry mismatch")
 ng=len(cgroups); child=a["child_global_row"].astype(np.int64); gid=a["group_id"].astype(np.int64)
 if np.any(gid<0) or np.any(gid>=ng): raise ValueError("group ID outside full-cut registry")
 if not np.array_equal(a["source_exact_tuple"],cgroups[gid]): raise ValueError("group ID to exact tuple crosswalk mismatch")
 pair=child*ng+gid
 if len(np.unique(pair))!=n: raise ValueError("duplicate route group-child pair")
 if rep["cut"]["T_skin_children"]!=355193 or rep["cut"]["T_fat_children"]!=0: raise ValueError("cut T role roster changed")

 src_lab_path=Path(named["source330_labels"]["path"])
 sl=np.loadtxt(src_lab_path,delimiter=",",skiprows=1,usecols=(0,1,2,3),dtype=np.int64)
 if not np.array_equal(sl[:,0],np.arange(len(sl))): raise ValueError("source330 row identity mismatch")
 canonical=np.asarray(sorted({tuple(map(int,r[1:4])) for r in sl}),np.int32)
 if not np.array_equal(canonical,cgroups): raise ValueError("source330 group registry differs from full cut")
 sg=a["source_global330_tet_row"].astype(np.int64)
 if np.any(sg<0) or np.any(sg>=len(sl)) or not np.array_equal(a["source_exact_tuple"],sl[sg,1:4].astype(np.int32)):
  raise ValueError("route source tuple does not replay from source330 labels")
 if not np.all(a["source_role_material"]==15) or not np.all(a["source_exact_tuple"][:,0]==15):
  raise ValueError("route contains non-Skin source role")
 if not np.array_equal(a["source_local_parent_row"],a["source_global330_tet_row"]):
  raise ValueError("Skin local/global parent row mismatch")

 with np.load(Path(named["carrier_skin_packet"]["path"]),allow_pickle=False) as z:
  pp=z["source_parent_tet_rows"].astype(np.int64)
  pg=z["global_source330_tet_rows"].astype(np.int64)
  po=z["parent_offsets"].astype(np.int64)
  pb=z["carrier_barycentric"]; pw=z["carrier_weights"]
 if (po.shape!=(len(pp)+1,) or po[0]!=0 or po[-1]!=len(pw)
     or np.any(np.diff(po)<0) or pg.shape!=pp.shape or pb.shape!=(len(pw),4)):
  raise ValueError("032d packet parent/carrier CSR malformed")
 point_parent_index=np.repeat(np.arange(len(pp),dtype=np.int64),np.diff(po))
 if point_parent_index.shape!=(len(pw),):
  raise ValueError("032d packet point-to-parent expansion malformed")
 carrier=a["packet_carrier_row"].astype(np.int64); virtual=a["virtual_parent_row"].astype(np.int64)
 if np.any(carrier<0) or np.any(carrier>=len(pw)) or not np.array_equal(virtual,carrier):
  raise ValueError("virtual parent/032d packet row crosswalk mismatch")
 source_parent_index=point_parent_index[carrier]
 if (not np.array_equal(a["source_local_parent_row"],pp[source_parent_index])
     or not np.array_equal(a["source_global330_tet_row"],pg[source_parent_index])
     or not np.array_equal(a["source_barycentric"],pb[carrier])
     or not np.array_equal(a["source_weight"],pw[carrier])):
  raise ValueError("source parent/bary/weight does not replay from pinned 032d packet")
 reached=a["reached_source_virtual_rows"].astype(np.int64); rmass=a["reached_source_point_mass_kg"].astype(np.float64)
 if reached.shape!=(2482,) or rmass.shape!=(2482,) or len(np.unique(reached))!=len(reached) or not np.all(np.isin(virtual,reached)):
  raise ValueError("route virtual parents are not in the unique reached-source roster")
 sm=np.loadtxt(Path(named["source330_mass"]["path"]),delimiter=",",skiprows=1,usecols=(0,2),dtype=np.float64)
 rerr=float(np.max(np.abs(sm[pg[point_parent_index[reached]],1]*pw[reached]-rmass)))
 if rerr>1e-17: raise ValueError("reached-source mass does not replay from exact 032d weight")

 sn=np.loadtxt(Path(named["source330_nodes"]["path"]),delimiter=",",skiprows=1,usecols=(0,5,6,7),dtype=np.float64)
 st=np.loadtxt(Path(named["source330_tets"]["path"]),delimiter=",",skiprows=1,usecols=(0,3,4,5,6),dtype=np.float64)
 if not np.array_equal(st[:,0].astype(np.int64),np.arange(len(st))): raise ValueError("source tet identity mismatch")
 tn=st[sg,1:5].astype(np.int64)
 if tn.min()<0 or tn.max()>=len(sn): raise ValueError("source tet node index outside source domain")
 source_xyz=np.einsum("ni,nij->nj",a["source_barycentric"],sn[tn,1:4])
 source_error=float(np.max(np.abs(source_xyz-a["source_point_xyz"])))
 if source_error>2e-13: raise ValueError("source point fails exact 032d F64 barycentric replay")

 cl=np.loadtxt(Path(named["child_skin_labels"]["path"]),delimiter=",",skiprows=1,usecols=(0,1,2,3),dtype=np.int64)
 if not np.array_equal(cl[:,0],np.arange(len(cl))): raise ValueError("current child labels row identity mismatch")
 c_local=a["child_role_local_row"].astype(np.int64)
 skin_rows=np.flatnonzero(cl[:,1]==15).astype(np.int64)
 if np.any(child<0) or np.any(child>=len(cl)) or np.any(c_local<0) or np.any(c_local>=len(skin_rows)):
  raise ValueError("route child row outside current Skin domain")
 if not np.array_equal(skin_rows[c_local],child) or not np.all(cl[child,1]==15):
  raise ValueError("route child global/role-local/material crosswalk mismatch")

 mask_header, _=cut_header_and_groups(cut)
 mask=np.unpackbits(np.frombuffer(bytes.fromhex(mask_header["child_mask_hex"]),dtype=np.uint8),bitorder="little")[:len(cl)].astype(bool)
 tmask=~mask
 if int(tmask.sum())!=355193 or not np.all(tmask[child]): raise ValueError("route child is not in current unreached T")
 if np.any(tmask & (cl[:,1]==16)): raise ValueError("current T unexpectedly contains Fat children")

 node_path=Path(named["child_skin_node"]["path"]); ele_path=Path(named["child_skin_elements"]["path"])
 with node_path.open() as f: nh=[int(x) for x in f.readline().split()]
 with ele_path.open() as f: eh=[int(x) for x in f.readline().split()]
 nodes=np.loadtxt(node_path,skiprows=1,usecols=(0,1,2,3),dtype=np.float64)
 elems=np.loadtxt(ele_path,skiprows=1,usecols=(0,1,2,3,4),dtype=np.int64)
 if nh[:2]!=[311768,3] or eh[:2]!=[len(cl),4] or nodes.shape!=(311768,4) or elems.shape!=(len(cl),5):
  raise ValueError("current 433 TetGen dimensions changed")
 if (not np.array_equal(nodes[:,0].astype(np.int64),np.arange(1,len(nodes)+1))
     or not np.array_equal(elems[:,0],np.arange(1,len(cl)+1))):
  raise ValueError("current 433 TetGen IDs changed")
 unique_child=np.unique(child); conn=elems[unique_child,1:5]
 if conn.min()<1 or conn.max()>len(nodes): raise ValueError("current child tet has invalid node ID")
 cent=nodes[conn-1,1:4].mean(axis=1)
 ri=np.searchsorted(unique_child,child)
 route_cent=a["child_centroid_xyz"]
 centroid_error=float(np.max(np.abs(cent[ri]-route_cent)))
 if centroid_error>2e-12: raise ValueError("child centroid fails pinned 433 geometry replay")

 with np.load(Path(named["p1_skin_support"]["path"]),allow_pickle=False) as z:
  ro=z["cell_admissible_region_offsets"].astype(np.int64); rids=z["cell_admissible_region_ids"].astype(np.int32)
  go=z["cell_group_offsets"].astype(np.int64); gids_local=z["cell_group_ids"].astype(np.int32)
  gt_local=z["group_tuples"].astype(np.int32)
  onehop=int(np.asarray(z["admissibility_max_source_region_hops"]).item())
 if onehop!=1 or len(ro)!=len(skin_rows)+1 or len(go)!=len(skin_rows)+1:
  raise ValueError("P1 one-hop CSR domain mismatch")
 if np.any(np.diff(ro)<0) or np.any(np.diff(go)<0): raise ValueError("P1 CSR offset order invalid")
 counts=np.diff(ro); row=np.repeat(np.arange(len(counts),dtype=np.int64),counts)
 within=np.arange(len(rids),dtype=np.int64)-np.repeat(ro[:-1],counts)
 internal=within[1:]>0
 if np.any(rids[1:][internal]<=rids[:-1][internal]): raise ValueError("P1 region rows not strictly sorted")
 stride=int(max(rids.max(initial=0),a["source_exact_tuple"][:,2].max(initial=0)))+1
 allowed_region=np.unique(row*stride+rids.astype(np.int64))
 requested_region=c_local*stride+a["source_exact_tuple"][:,2].astype(np.int64)
 ix=np.searchsorted(allowed_region,requested_region)
 if np.any(ix>=len(allowed_region)) or not np.all(allowed_region[ix]==requested_region):
  raise ValueError("route region fails exact one-hop P1 CSR")

 tuple_to_group={tuple(map(int,t)):i for i,t in enumerate(cgroups)}
 local_to_global=np.asarray([tuple_to_group.get(tuple(map(int,t)),-1) for t in gt_local],np.int32)
 if np.any(local_to_global<0): raise ValueError("P1 group absent from cut registry")
 gc=np.diff(go); grow=np.repeat(np.arange(len(gc),dtype=np.int64),gc)
 existing=np.unique(skin_rows[grow]*ng+local_to_global[gids_local].astype(np.int64))
 route_key=child*ng+gid
 ix=np.searchsorted(existing,route_key)
 present=(ix<len(existing)) & (existing[np.minimum(ix,len(existing)-1)]==route_key)
 if np.any(present): raise ValueError("route exact group-child pair already exists in P1")

 envref=run["mass_transport_envelope_receipt"]; env=json.loads(Path(envref["path"]).read_text())
 envelope=env.get("mass_transport_envelope",{}); radius=float(envelope.get("predeclared_transport_envelope_m",0.0))
 if envelope.get("schema")!="numi.human.mass-transport-envelope.v1" or radius!=0.187444:
  raise ValueError("pinned 1058 envelope schema/cap mismatch")
 if float(rep["cut"]["transport_envelope_m"])!=radius: raise ValueError("census/current envelope mismatch")
 argv=run["argv"]; r_arg=float(argv[argv.index("--maximum-transport-distance-m")+1])
 if r_arg!=radius: raise ValueError("out-006 CLI R mismatch")
 dist=np.linalg.norm(a["source_point_xyz"]-route_cent,axis=1)
 dist_error=float(np.max(np.abs(dist-a["distance_m"])))
 if dist_error>2e-12 or float(dist.max())>radius+1e-12: raise ValueError("route distance/R replay failed")
 if abs(float(dist.max())-float(rep["whole_T_routes"]["distance_m_max"]))>2e-12:
  raise ValueError("route maximum distance differs from recovered census summary")

 if PART.exists() or REPORT.exists(): raise FileExistsError("refusing overwrite existing partial/report")
 if not recovering_existing_output:
  cp=subprocess.run(["cp","-c",str(skin),str(PART)],capture_output=True,text=True)
  if cp.returncode: raise RuntimeError("APFS clonefile failed: "+cp.stderr.strip())
  with PART.open("a",newline="",encoding="utf-8") as f:
   wr=csv.writer(f,lineterminator="\n")
   for i in range(n): wr.writerow(route_row(a,i))
   f.flush(); os.fsync(f.fileno())
 append_target=OUT if recovering_existing_output else PART
 if append_target.stat().st_size<=skin.stat().st_size: raise ValueError("append output did not increase CSV size")
 prefix=hashlib.sha256(); suffix=hashlib.sha256(); full=hashlib.sha256()
 source_bytes=skin.stat().st_size; position=0
 with append_target.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):
   full.update(block); end=position+len(block)
   if position<source_bytes:
    take=min(len(block),source_bytes-position); prefix.update(block[:take])
   if end>source_bytes:
    first=max(0,source_bytes-position); suffix.update(block[first:])
   position=end
 if position<=source_bytes or prefix.hexdigest()!=skinpin["sha256"]:
  raise ValueError("v15 byte prefix changed in output")
 expected=DigestWriter(); expected_writer=csv.writer(expected,lineterminator="\n")
 for i in range(n): expected_writer.writerow(route_row(a,i))
 appended_bytes=position-source_bytes
 if (appended_bytes!=expected.bytes or suffix.hexdigest()!=expected.digest.hexdigest()):
  raise ValueError("appended output rows do not exactly serialize the pinned route NPZ")
 outsha=full.hexdigest(); outbytes=append_target.stat().st_size
 if not recovering_existing_output: os.replace(PART,OUT)
 producer_sha=sha(Path(__file__))
 result={
  "schema":"numi.h706.current-038-whole-cut-skin-route-append-validation.v1",
  "status":"PASS_ALL_ROUTE_SLOTS_REPLAYED_AND_APPENDED",
  "physical_admission":False,"native_steps":0,
  "census_recovery":recovery,
  "prior_append_attempt_recovery":{
   "source":{"path":"append_skin_witnesses_from_whole_cut_routes_038c.py","sha256":"e3694a708ac4ce1a91b2d3574ffb2dbb66ad69330d354daa45b1af9492768abd"},
   "prior_exception":"KeyError p1_skin during receipt assembly after output CSV replacement",
   "existing_output_reused_without_reappend":recovering_existing_output,
   "appended_suffix_replayed_byte_exactly":True},
  "pins":{
   "out006_run_manifest":skinpin if False else {"path":str(RUNMAN.resolve()),"sha256":PINS["run"],"bytes":RUNMAN.stat().st_size},
   "route_census_report":{"path":str(CENSUS.resolve()),"sha256":PINS["census"],"bytes":CENSUS.stat().st_size},
   "route_npz":{"path":str(NPZ.resolve()),"sha256":PINS["npz"],"bytes":NPZ.stat().st_size},
   "route_producer":{"path":str(PRODUCER.resolve()),"sha256":PINS["producer"],"bytes":PRODUCER.stat().st_size},
   "base_route_producer":{"path":str(BASEPRODUCER.resolve()),"sha256":PINS["baseproducer"],"bytes":BASEPRODUCER.stat().st_size},
   "skin_v15_prefix":skinpin,
   "fat_v15_unchanged":{"path":str(fat.resolve()),"sha256":PINS["fat"],"bytes":fat.stat().st_size},
   "append_producer":{"path":str(Path(__file__).resolve()),"sha256":producer_sha,"bytes":Path(__file__).stat().st_size},
   "p1_skin_support":dict(named["p1_skin_support"]),"skin_packet_032d":dict(named["carrier_skin_packet"]),
   "source330_labels":dict(named["source330_labels"]),"source330_nodes":dict(named["source330_nodes"]),
   "source330_tets":dict(named["source330_tets"]),"child_skin_labels":dict(named["child_skin_labels"]),
   "child_skin_node":dict(named["child_skin_node"]),"child_skin_elements":dict(named["child_skin_elements"]),
   "envelope_receipt":dict(envref)},
  "validation":{
   "route_pairs":n,"unique_group_child_pairs":int(len(np.unique(pair))),
   "all_sources_skin":True,"all_targets_skin":True,"all_targets_in_current_T":True,
   "032d_packet_crosswalk_rows":n,"source_barycentric_exact_packet_matches":n,
   "source_point_replay_max_abs_error_m":source_error,"child_role_local_crosswalk_rows":n,
   "child_centroid_replay_max_abs_error_m":centroid_error,"one_hop_P1_region_checks":n,
   "P1_duplicate_group_child_pairs":0,"prior_full_allowlist_pairs_checked":al["candidate_pairs_checked_against_reconstructed_allowlist"],
   "prior_full_allowlist_pair_hits":al["candidate_pair_already_present"],"residual_children_replayed":al["residual_children_checked"],
   "reached_source_virtual_rows":len(reached),"reached_source_mass_max_abs_error_kg":rerr,
   "route_distance_replay_max_abs_error_m":dist_error,"maximum_route_distance_m":float(dist.max()),
   "predeclared_R_m":radius,"T_skin_children":int(rep["cut"]["T_skin_children"]),
   "T_fat_children":int(rep["cut"]["T_fat_children"]),
   "routed_T_skin_children":int(rep["whole_T_routes"]["T_children_covered_by_any_new_route"]),
   "unrouted_T_skin_children":int(rep["whole_T_routes"]["T_children_without_any_new_route"]),
   "new_cut_arc_capacity_kg":float(rep["whole_T_routes"]["new_S_to_T_implicit_arc_capacity_sum_kg"]),
   "capacity_interpretation":"cut-arc capacity sum only, not a feasible flow or mass-conservation result"},
  "output_csv":{"path":str(OUT.resolve()),"sha256":outsha,"bytes":outbytes,
   "source_prefix_bytes":skin.stat().st_size,"source_prefix_sha256":skinpin["sha256"],
   "appended_rows":n,"appended_suffix_bytes":appended_bytes,
   "appended_suffix_sha256":suffix.hexdigest(),"header_columns":26}}
 temp=REPORT.with_suffix(".json.tmp")
 temp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 os.replace(temp,REPORT)
 print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__": main()
