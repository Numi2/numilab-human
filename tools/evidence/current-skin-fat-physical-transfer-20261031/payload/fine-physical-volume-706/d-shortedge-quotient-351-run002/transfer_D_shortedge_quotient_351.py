from pathlib import Path
import json,hashlib,time,importlib.util,sys
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE=H/"fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz"
BASE_SHA="d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f"
SURFACE=H/"d-shortedge-quotient-351-run002/identity-seed-D.npz"
SURFACE_SHA="7cab78eefc85112eda3c9f13df9839ecaadc58d82ac00f20b828d11d0ca8b535"
SURFACE_OUT=H/"d-shortedge-quotient-351-run002/surface-run001/candidate-D-short-edge-collapse-raw.npz"
SURFACE_OUT_SHA="29f9ec52652a4a942f8dca1fba9765b502f98d7271e9ddef91731f7e5b72f67f"
REPORT=H/"d-shortedge-quotient-351-run002/surface-run001/report.json"
REPORT_SHA="b549dd67254f0e44ae69c28f69545c5847235ca12dbdf69b98f7d3295d77156b"
LEDGER=H/"d-shortedge-quotient-351-run002/surface-run001/accepted-operations.jsonl"
PREFLIGHT=H/"d-shortedge-quotient-351-run002/preflight.json"
PREFLIGHT_SHA="43cded6ce3f6e093cbd05f7c678c079c54909f3ea7658fb061f59efa0837d7a8"
AUTH=H/"d-shortedge-quotient-351-run002/allowed-D-edge-map.npz"
AUTH_SHA="ea28e81a91b27b11997668b496acd7fce63ccc19ad97b1d721cb7e363d523445"
CORE=H/"transfer_surface_contractions_to_volume_209_root.py"
def sha(p):return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
q=mod("surface_volume_quotient_209",CORE)
m=q.m
def run(out):
 out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
 pins={str(p):h for p,h in ((BASE,BASE_SHA),(SURFACE,SURFACE_SHA),(SURFACE_OUT,SURFACE_OUT_SHA),(REPORT,REPORT_SHA),(PREFLIGHT,PREFLIGHT_SHA),(AUTH,AUTH_SHA),(CORE,sha(CORE)))}
 for p,h in list(pins.items()):
  if sha(Path(p))!=h:raise ValueError(f"input SHA mismatch {p}")
 rep=json.loads(REPORT.read_text())
 if rep.get("status")!="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL":raise ValueError("surface proposal status")
 if rep.get("joint_changed_vs_complete_OID_walls_all185",{}).get("pair_count")!=0:raise ValueError("surface exact context failed")
 if rep.get("joint_changed_vs_complete_OID_walls_all185",{}).get("new_face_self_pairs")!=0:raise ValueError("surface self intersection")
 with np.load(BASE,allow_pickle=False) as z:
  p=np.asarray(z["points_world_m"],np.float32);t=np.asarray(z["tetrahedra"],np.int64).copy();mat=np.asarray(z["material_indices"],np.int32).copy()
 with np.load(SURFACE,allow_pickle=False) as z:
  sp=np.asarray(z["points_world_m_f32"],np.float32);sf=np.asarray(z["faces"],np.int64)
 ops=[json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
 if len(ops)!=307 or any(x.get("role")!="D" for x in ops):raise ValueError("unexpected surface operation census")
 with np.load(AUTH,allow_pickle=False) as z:
  allowed={(int(a),int(b)) for a,b in zip(np.asarray(z["D_allowed_collapsed_point_rows"],np.int64),np.asarray(z["D_allowed_retained_point_rows"],np.int64))}
 seen_surface=set()
 for op in ops:
  k=(int(op["collapsed_input_point_row"]),int(op["retained_input_point_row"]))
  if k not in allowed:raise ValueError("surface op orientation not in 351 authorized map")
  used=set(map(int,op["affected_parent_face_rows"]))
  if seen_surface&used:raise ValueError("surface operations are not star-disjoint")
  seen_surface|=used
 b=q.boundary(t);look={q.key(x):i for i,x in enumerate(p)}
 if len(look)!=len(p):raise ValueError("ambiguous F32 point identity in source volume")
 v={"path":BASE,"p":p,"t":t,"mat":mat,"base_boundary":b,"boundary_keys":{tuple(sorted(map(int,f))) for f in b},"look":look,"active":np.ones(len(t),bool),"map":np.arange(len(p),dtype=np.int64),"accepted":[],"used_cell_stars":[]}
 before_det,before_q=m.measurements(p,t)
 if np.any(before_det<=0):raise ValueError("base volume has nonpositive F32 signed cells")
 before_good=m.passes(before_det,before_q)
 operation_results=[];rejected=[]
 surface_point_hits=np.fromiter((look.get(q.key(x),-1) for x in sp),np.int64,count=len(sp))
 if np.count_nonzero(surface_point_hits>=0)!=120635:raise ValueError("D surface/volume coordinate crosswalk count changed")
 for oi,op in enumerate(ops):
  result,why=q.try_op(v,sp,sf,op)
  if result is None:
   rejected.append({"ledger_row":oi,"patch_id":op.get("patch_id"),"reason":why})
   continue
  rows=result["rows"];deleted=result["deleted"]
  if any(np.intersect1d(rows,prior,assume_unique=False).size for prior in v["used_cell_stars"]):
   rejected.append({"ledger_row":oi,"patch_id":op["patch_id"],"reason":"overlaps_prior_volume_star"});continue
  v["used_cell_stars"].append(rows.copy())
  v["t"][rows]=result["new"]
  v["active"][rows[deleted]]=False
  v["map"][result["retired_volume_node"]]=result["kept_volume_node"]
  v["accepted"].append((oi,op,result))
  operation_results.append({"ledger_row":oi,"patch_id":op["patch_id"],"drop_D_row":int(op["collapsed_input_point_row"]),"keep_D_row":int(op["retained_input_point_row"]),"drop_volume_node":int(result["retired_volume_node"]),"keep_volume_node":int(result["kept_volume_node"]),"touched_input_cell_rows":rows.astype(int).tolist(),"removed_input_cell_rows":rows[deleted].astype(int).tolist(),"removed_cell_count":result["removed_cells"],"local_quality_bad_before":result["bad_before"],"local_quality_bad_after":result["bad_after"],"removed_volume_m3":float(np.sum(m.measurements(p,result["old"])[0][deleted])/6.0)})
 t2=v["t"][v["active"]];mat2=v["mat"][v["active"]]
 uncompact_b=q.boundary(t2)
 mapped_expected=v["map"][v["base_boundary"]]
 srt=np.sort(mapped_expected,axis=1);mapped_expected=mapped_expected[~np.any(srt[:,1:]==srt[:,:-1],axis=1)]
 boundary_equal=np.array_equal(q.boundary_identity(mapped_expected),q.boundary_identity(uncompact_b))
 if not boundary_equal:raise ValueError("volume boundary does not equal exact quotient of the d4906 boundary")
 used=np.unique(t2);compact=np.full(len(p),-1,np.int64);compact[used]=np.arange(len(used),dtype=np.int64)
 out_t=compact[t2];out_p=p[used];out_b=compact[uncompact_b]
 det,qb=m.measurements(out_p,out_t)
 if np.any(det<=0):raise ValueError("quotient output has nonpositive F32 signed cell")
 exactpoints=m.integer_points(out_p)
 exact_bad=[]
 for i,tet in enumerate(out_t):
  if m.exact_det(exactpoints,tet)<=0:exact_bad.append(i)
 if exact_bad:raise ValueError(f"exact integer F32 orientation failed in output cells {exact_bad[:12]}")
 after_good=m.passes(det,qb)
 if np.count_nonzero(~after_good)>np.count_nonzero(~before_good):raise ValueError("global native geometry quality regression")
 retained_rows=np.flatnonzero(v["active"])
 regression=retained_rows[before_good[retained_rows]&~after_good]
 if len(regression):raise ValueError(f"formerly-good cells regressed: {regression[:12].tolist()}")
 in_node_to_out=compact[v["map"]]
 input_cell_to_out=np.full(len(t),-1,np.int64);input_cell_to_out[retained_rows]=np.arange(len(retained_rows),dtype=np.int64)
 out_npz=out/"fat-D-shortedge-quotient-volume-proposal.npz"
 np.savez_compressed(out_npz,points_world_m=out_p,tetrahedra=out_t,material_indices=mat2,boundary_faces=out_b,retained_input_cell_rows=retained_rows,input_cell_to_output_cell=input_cell_to_out,output_cell_to_input_cell=retained_rows,input_node_to_output_node=in_node_to_out,output_node_to_input_node=used,source_surface_contraction_ledger_rows=np.asarray([x[0] for x in v["accepted"]],np.int64))
 total_before=np.bincount(mat,weights=before_det/6,minlength=int(mat.max())+1)
 total_after=np.bincount(mat2,weights=det/6,minlength=int(mat.max())+1)
 mat_report={}
 for i in range(max(len(total_before),len(total_after))):
  a=float(total_before[i]) if i<len(total_before) else 0.;c=float(total_after[i]) if i<len(total_after) else 0.
  if a or c:mat_report[str(i)]={"input_volume_m3":a,"output_volume_m3":c,"delta_m3":c-a}
 outD=np.load(SURFACE_OUT,allow_pickle=False);dfaces=np.asarray(outD["faces"],np.int64)
 bkeys={tuple(sorted(map(int,f))) for f in out_b};d_face_count=0;present=0;unresolved_point_faces=0
 for tri in dfaces:
  mtri=[]
  for sr in tri:
   vr=int(surface_point_hits[int(sr)])
   if vr<0:mtri=[];break
   mapped_node=int(v["map"][vr])
   if mapped_node<0 or compact[mapped_node]<0:mtri=[];break
   mtri.append(int(compact[mapped_node]))
  if len(mtri)!=3:unresolved_point_faces+=1;continue
  if len(set(mtri))<3:continue
  d_face_count+=1
  if tuple(sorted(mtri)) in bkeys:present+=1
 outD.close()
 report={"schema":"h706.d-shortedge-351-volume-quotient.v1","status":"PASS_D_ONLY_POSITIVE_VOLUME_QUOTIENT_PARTIAL_BOUNDARY","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","source_pins":pins,"surface":{"report_path":str(REPORT),"report_sha256":REPORT_SHA,"raw_D_path":str(SURFACE_OUT),"raw_D_sha256":SURFACE_OUT_SHA,"authorized_edge_map_path":str(AUTH),"authorized_edge_map_sha256":AUTH_SHA,"accepted_surface_operation_count":len(ops),"accepted_volume_operation_count":len(operation_results),"rejected_volume_operation_count":len(rejected),"rejected":rejected,"operations":operation_results},"volume":{"input_path":str(BASE),"input_sha256":BASE_SHA,"output_path":str(out_npz),"output_sha256":sha(out_npz),"input_nodes":len(p),"output_nodes":len(out_p),"input_cells":len(t),"output_cells":len(out_t),"retired_cells":int(len(t)-len(out_t)),"contracted_volume_nodes":len(v["accepted"]),"boundary_equals_exact_transformed_input_boundary":boundary_equal,"input_boundary_face_count":len(b),"output_boundary_face_count":len(out_b),"input_positive_f32_cells":int(np.count_nonzero(before_det>0)),"output_positive_f32_cells":int(np.count_nonzero(det>0)),"output_exact_integer_positive_cells":len(out_t),"input_native_quality_failures":int(np.count_nonzero(~before_good)),"output_native_quality_failures":int(np.count_nonzero(~after_good)),"formerly_good_retained_cell_regressions":len(regression),"minimum_output_det6_m3":float(det.min()),"minimum_output_tet_volume_m3":float((det/6).min()),"minimum_output_joint_native_quality":float(qb.min()),"total_physical_volume_by_material_m3":mat_report,"input_to_output_node_map_sha256":hashlib.sha256(in_node_to_out.tobytes()).hexdigest(),"output_to_input_node_map_sha256":hashlib.sha256(used.tobytes()).hexdigest(),"input_to_output_cell_map_sha256":hashlib.sha256(input_cell_to_out.tobytes()).hexdigest(),"output_to_input_cell_map_sha256":hashlib.sha256(retained_rows.tobytes()).hexdigest()},"target_boundary_comparison":{"mapped_current_D_faces":d_face_count,"mapped_current_D_faces_present_in_quotient_volume_boundary":present,"D_output_faces_with_unmapped_surface_points":unresolved_point_faces,"interpretation":"partial current-boundary comparison; pending cavities and source splits remain outside this D-only quotient"},"limitations":["D-only topology quotient of d4906; it does not merge 352 I-side Fat operations or pending cavity remeshes.","D current target boundary remains partial relative to the whole Fat volume.","No mass mapping, native installation, or physical admission is claimed."],"elapsed_s":time.monotonic()-tic}
 (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":report["status"],"output":str(out_npz),"output_sha256":sha(out_npz),"report":str(out/"report.json"),"report_sha256":sha(out/"report.json"),"surface_accepts":len(ops),"volume_accepts":len(operation_results),"volume_rejects":len(rejected),"cells_before_after":[len(t),len(out_t)],"nodes_before_after":[len(p),len(out_p)],"boundary_exact":boundary_equal,"qfail_before_after":[int(np.count_nonzero(~before_good)),int(np.count_nonzero(~after_good))],"formerly_good_regressions":len(regression),"current_D_faces_present":present,"current_D_faces_mapped":d_face_count,"unmapped_D_faces":unresolved_point_faces,"elapsed_s":report["elapsed_s"]},indent=2))
if __name__=="__main__":
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);run(ap.parse_args().out)
