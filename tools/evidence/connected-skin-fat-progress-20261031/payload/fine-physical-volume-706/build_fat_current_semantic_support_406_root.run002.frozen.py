"""Build current Fat node/cell semantic support by the pinned 141/230 owner.
Exact coordinate matches reuse the frozen 141 per-node support; only unmatched
current points are queried with the same full-tie same-material 230 owner.
No scalar nearest-label assignment or mass/physical admission is made.
"""
from pathlib import Path
import numpy as np, json, hashlib, importlib.util, sys, time, math
from collections import Counter
H=Path(__file__).resolve().parent
RUNNER=Path(__file__).resolve()
OUT=H/"fat-current-semantic-support-406-root-run002"
VOL=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
BASEVOL=H/"combined-D309-I399-volume-404-root/fat-D309-I399-full-domain-volume-proposal.npz"
OLDVOL=H/"fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz"
OLDSUPPORT=H/"fat-current-semantic-support-141-root/fat-current-semantic-support.npz"
OLDRECEIPT=H/"fat-current-semantic-support-141-root/receipt.json"
VOLREPORT=H/"current-D309-I399-target-406-root/target397/report.json"
BASE=H.parent/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
OWNER=H/"current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py"
PINS={
 VOL:"5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211",
 BASEVOL:"c7715bf911a9bfa99ce712c499233827d0d447fb2f010dc5891e1ef4c5dcf054",
 OLDVOL:"7ab13a433e0cb08350c901388c0c86712269e0cf66ce70df5fe33de248357e76",
 OLDSUPPORT:"f3081e59fd88a88b48aa3d9caa8153609f6c24fcc0aac32d069c634ddad12b2b",
 OLDRECEIPT:"36a44948aa2337d470dead595d4a17ac00dfb80b0744ff947760a3f914498440",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imp(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def exact_keys(a):
 a=np.ascontiguousarray(a)
 if a.dtype!=np.float32 or a.ndim!=2 or a.shape[1]!=3: raise ValueError("point key domain must be float32 Nx3")
 return a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).reshape(-1)
def main():
 if OUT.exists(): raise FileExistsError(str(OUT))
 OUT.mkdir()
 start=time.monotonic()
 pins={str(p):sha(p) for p in (RUNNER,VOL,VOLREPORT,BASEVOL,OLDVOL,OLDSUPPORT,OLDRECEIPT,BASE,OWNER)}
 for p,s in PINS.items():
  if sha(p)!=s: raise ValueError("frozen input hash mismatch: "+str(p))
 oldreceipt=json.loads(OLDRECEIPT.read_text())
 if oldreceipt["output"]["sha256"]!=sha(OLDSUPPORT): raise ValueError("141 receipt/output mismatch")
 if oldreceipt["inputs"][str(OLDVOL)]!=sha(OLDVOL): raise ValueError("141 volume input mismatch")
 sys.path.insert(0,str(H/"downstream-003-current-typed-228/native-domain-217-source-005"))
 reg=imp("regFat406",OWNER)
 parent,index,mi,pm=reg.load_parent_current(BASE)
 with np.load(VOL) as z:
  points=z["points_world_m"].copy(); tets=z["tetrahedra"].copy(); mats=z["material_indices"].copy()
  added=z["fat_boundary_rebase_377_component_added_point_global_rows"].copy()
 with np.load(BASEVOL) as z: basepoints=z["points_world_m"].copy(); source_output_to_input=z["source_5bf_output_node_to_input_node"].copy()
 with np.load(OLDVOL) as z: oldpoints=z["points_world_m"].copy()
 with np.load(OLDSUPPORT) as z: old={k:z[k] for k in z.files}
 if len(source_output_to_input)!=253199 or len(points)<=len(source_output_to_input): raise ValueError("404/406 prefix row-map domain mismatch")
 if not np.array_equal(points[:len(source_output_to_input)].view("u4"),basepoints[source_output_to_input].view("u4")):
  raise ValueError("406 point prefix differs from pinned 404 source through source_5bf_output_node_to_input_node")
 if not np.array_equal(added,np.arange(len(source_output_to_input),len(points),dtype=added.dtype)):
  raise ValueError("explicit 377 added-node ancestry is not the appended 21-row suffix")
 if len(added)!=21: raise ValueError("expected 21 explicit current cavity construction nodes")
 if not np.all(mats==16): raise ValueError("current Fat volume contains non-Fat material IDs")
 if len(oldpoints)!=len(old["node_group_offsets"])-1: raise ValueError("141 node domain mismatch")
 if not np.array_equal(old["group_tuples"],np.array(sorted({tuple(int(parent[k][pt]) for k in ("materials","donors","regions")) for pt in np.flatnonzero(parent["materials"]==16)}),np.int32)):
  raise ValueError("141 source group domain changed")
 # Copy the exact 141 support only when the complete float32 point bytes match.
 oldkeys=exact_keys(oldpoints); curkeys=exact_keys(points)
 if len(np.unique(oldkeys))!=len(oldkeys): raise ValueError("141 exact point coordinates are not unique")
 ord_old=np.argsort(oldkeys,kind="stable"); sorted_old=oldkeys[ord_old]
 pos=np.searchsorted(sorted_old,curkeys); valid=pos<len(sorted_old)
 valid &= sorted_old[np.minimum(pos,len(sorted_old)-1)]==curkeys
 current_to_old=np.full(len(points),-1,np.int64);current_to_old[valid]=ord_old[pos[valid]]
 node_support=[{} for _ in range(len(points))]
 chosen=np.full(len(points),-1,np.int32); bary=np.zeros((len(points),4),np.float64)
 residual=np.zeros((len(points),3),np.float64);dist=np.full(len(points),np.nan,np.float64);origin=np.zeros(len(points),np.uint8)
 oldg=old["group_tuples"]
 oldno=old["node_group_offsets"]; oldids=old["node_group_ids"]
 oldpo=old["node_group_parent_tet_offsets"]; oldpr=old["node_group_parent_tet_rows"]
 for i in np.flatnonzero(valid):
  oi=int(current_to_old[i]);chosen[i]=old["node_parent_chosen_tet_row"][oi]
  bary[i]=old["node_parent_barycentric"][oi];residual[i]=old["node_parent_residual_world_m"][oi]
  dist[i]=old["node_parent_registration_distance_m"][oi];origin[i]=old["node_origin_kind"][oi]
  for slot in range(int(oldno[oi]),int(oldno[oi+1])):
   g=tuple(map(int,oldg[int(oldids[slot])]))
   node_support[i][g]=set(map(int,oldpr[int(oldpo[slot]):int(oldpo[slot+1])]))
 if np.any(valid):
  if np.nanmax(dist[valid])>.02: raise ValueError("reused 141 support exceeds owner radius")
 missing=np.flatnonzero(~valid)
 exact_added_match=bool(np.any(valid[added]))
 if exact_added_match: raise ValueError("cavity-added point unexpectedly aliases an old support point")
 def put(i,tr,b,res,d,ties):
  ties=list(map(int,ties))
  if tr not in ties or len(ties)!=len(set(ties)): raise ValueError("bad canonical tied source parent support")
  if not np.isfinite(d) or d>.02 or abs(np.linalg.norm(res)-d)>1e-10: raise ValueError("current node query distance outside 230 bound")
  q=parent["points"][parent["tets"][tr]]
  if abs(np.sum(b)-1)>1e-8 or np.min(b)<-1e-8 or np.linalg.norm(b@q+res-points[i].astype(float))>1e-10: raise ValueError("230 source-point reconstruction failed")
  for pt in ties:
   g=tuple(int(parent[k][pt]) for k in ("materials","donors","regions"))
   if g[0]!=16: raise ValueError("canonical tie crosses material")
   node_support[i].setdefault(g,set()).add(pt)
  chosen[i]=tr;bary[i]=b;residual[i]=res;dist[i]=d;origin[i]=2
 fresh=[]
 qreport=[]
 for j,i in enumerate(missing):
  q=reg.closest_same_material(parent,index,mi,16,points[i].astype(float),.02,1e-8)
  put(int(i),int(q["tet"]),np.asarray(q["bary"],np.float64),np.asarray(q["residual"],np.float64),float(q["distance"]),q["ties"])
  fresh.append((int(i),int(q["tet"]),np.asarray(q["bary"],np.float64),np.asarray(q["residual"],np.float64),float(q["distance"]),np.asarray(q["ties"],np.int32)))
  if j%512==511 or j+1==len(missing):
   progress={"stage":"CANONICAL_230_QUERY_UNMATCHED_CURRENT_POINTS","queried":j+1,"total":len(missing),"reused_exact_point_rows":int(valid.sum()),"elapsed_s":time.monotonic()-start,"physical_admission":False,"native_steps":0}
   (OUT/"progress.json").write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n");print(json.dumps(progress),flush=True)
 if np.any(chosen<0) or any(not x for x in node_support): raise ValueError("current point lacks canonical source support")
 groups=oldg.copy();gids={tuple(map(int,g)):i for i,g in enumerate(groups)}
 ngoff=[0];ng=[];npoff=[0];npr=[]
 for supports in node_support:
  for g in sorted(supports):
   if g not in gids: raise ValueError("new canonical query introduced unregistered group")
   ng.append(gids[g]);npr.extend(sorted(supports[g]));npoff.append(len(npr))
  ngoff.append(len(ng))
 coff=[0];cg=[];num=[];den=[];cpoff=[0];cpr=[];cfoff=[0];cfr=[]
 for i,tet in enumerate(tets):
  supports=[node_support[int(v)] for v in tet]
  scale=math.lcm(*(len(s) for s in supports)); scores=Counter()
  for s in supports:
   for g in s:scores[g]+=scale//len(s)
  if sum(scores.values())!=4*scale: raise ValueError("P1 partition does not sum to one")
  for g in sorted(scores):
   cg.append(gids[g]);num.append(scores[g]);den.append(4*scale)
   parents=sorted(set().union(*(s.get(g,set()) for s in supports)))
   faces=sorted(set(int(parent["faces"][pt]) for pt in parents))
   cpr.extend(parents);cpoff.append(len(cpr));cfr.extend(faces);cfoff.append(len(cfr))
  coff.append(len(cg))
  if i and i%100000==0: print(json.dumps({"stage":"REBUILD_CURRENT_CELL_P1_GROUPS","cells":i,"total":len(tets)}),flush=True)
 dest=OUT/"fat-current-semantic-support.npz"
 np.savez_compressed(dest,node_group_offsets=np.array(ngoff,np.int64),node_group_ids=np.array(ng,np.int32),node_group_parent_tet_offsets=np.array(npoff,np.int64),node_group_parent_tet_rows=np.array(npr,np.int32),node_origin_kind=origin,node_parent_registration_distance_m=dist,group_tuples=groups,cell_group_offsets=np.array(coff,np.int64),cell_group_ids=np.array(cg,np.int32),cell_group_p1_score_numerators=np.array(num,np.int64),cell_group_p1_score_denominators=np.array(den,np.int64),cell_group_parent_tet_offsets=np.array(cpoff,np.int64),cell_group_parent_tet_rows=np.array(cpr,np.int32),cell_group_stable_face_offsets=np.array(cfoff,np.int64),cell_group_stable_face_rows=np.array(cfr,np.uint32),node_parent_chosen_tet_row=chosen,node_parent_barycentric=bary,node_parent_residual_world_m=residual,volume_sha256=np.array(sha(VOL)))
 with np.load(dest) as z:
  if len(z["node_group_offsets"])!=len(points)+1 or len(z["cell_group_offsets"])!=len(tets)+1: raise ValueError("output domain mismatch")
  if not np.all(np.diff(z["cell_group_offsets"])>0): raise ValueError("unsupported current cells")
  if not np.all(z["cell_group_p1_score_numerators"]*1.0/z["cell_group_p1_score_denominators"]>0): raise ValueError("nonpositive P1 preference")
 cross=OUT/"exact-point-crosswalk.npz"
 np.savez_compressed(cross,current_output_to_141_node_row=current_to_old,explicit_added_point_global_rows=added)
 report={"schema":"numi.human.current-fat-semantic-support-rebuild.v1","status":"PASS_SOURCE_ONLY_COMPLETE_CURRENT_FAT_SEMANTIC_SUPPORT","physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN","policy":"The 406 point prefix is verified bitwise against the pinned 404 source through its explicit output-to-input row map; 21 appended rows are pinned to 377 cavity ancestry. Reuse per-node 141 support only for exact Float32 coordinate matches. Resolve every unmatched current point with the identical 230 exact same-material projection owner, retaining all tied parent-tet semantic groups; no scalar nearest-label override. Recompute every current cell P1 group from its actual four-node support. This rebuild does not qualify geometry or mass transport.","inputs":pins,"verified_point_prefix_map":{"field":"source_5bf_output_node_to_input_node","domain_rows":int(len(source_output_to_input)),"source_base_candidate_sha256":pins[str(BASEVOL)],"added_377_point_rows":added.tolist()},"runner":{"path":str(RUNNER),"sha256":sha(RUNNER)},"owner_141_script":{"path":str(H/"build_fat_current_semantic_support_141_root.py"),"sha256":sha(H/"build_fat_current_semantic_support_141_root.py")},"owner_230_script":{"path":str(OWNER),"sha256":sha(OWNER)},"source330":pm,"counts":{"current_points":len(points),"current_cells":len(tets),"explicit_added_377_cavity_points":len(added),"verified_unchanged_404_point_prefix_rows":int(len(source_output_to_input)),"exact_coordinate_reused_nodes":int(valid.sum()),"unmatched_nodes_queried_with_230":int(len(missing)),"unmatched_explicit_cavity_points_queried":int(np.count_nonzero(~valid[added])),"old_support_nodes_not_present_current":int(len(oldpoints)-np.unique(current_to_old[current_to_old>=0]).size),"source_groups":len(groups),"current_cell_group_slots":len(cg),"unsupported_nodes":0,"unsupported_cells":0},"limits":{"same_material":16,"node_projection_distance_m_max":.02,"tie_tolerance_m":1e-8,"mass_region_policy":"separate143authenticated1hop","mass_transport_radius_m":.019353,"mass_transport_NOT_evaluated":True},"output":{"path":str(dest),"sha256":sha(dest),"size_bytes":dest.stat().st_size},"crosswalk":{"path":str(cross),"sha256":sha(cross),"size_bytes":cross.stat().st_size},"elapsed_s":time.monotonic()-start,"maximum_node_projection_distance_m":float(np.max(dist))}
 (OUT/"receipt.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report),flush=True)
if __name__=="__main__": main()
