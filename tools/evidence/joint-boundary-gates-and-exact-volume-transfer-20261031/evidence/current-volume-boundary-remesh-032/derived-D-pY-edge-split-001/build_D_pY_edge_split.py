from pathlib import Path
from fractions import Fraction
import hashlib,json,importlib.util,sys
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"current-volume-boundary-remesh-032"
OUT=R/"derived-D-pY-edge-split-001"
D=H/"fixed-boundary-quality-topology-196/coordinate-six-short-edge-repair-030-run001/join-current-D030-plus-227-coordinates-001/candidate-D-current030-plus-227-coordinates-typed-raw.npz"
MAP=H/"current-volume-boundary-preflight-031/fat-current-desired-boundary-map.npz"
VOL=R/"fat026-current-target-input.npz"
PYREPORT=R/"fat-targetset-3c585-pY-diagnostic-001/run-002/report.json"
PYRUN=R/"fat-targetset-3c585-pY-diagnostic-001/run-002/run-result.json"
PYDIR=R/"fat-targetset-3c585-pY-diagnostic-001/run-002"
PYTABLES=[PYDIR/"cavity.1.node",PYDIR/"cavity.1.ele",PYDIR/"cavity.1.face",PYDIR/"cavity.1.edge"]

# Hash-pinned inputs from the recorded exact D mapping and pY-only diagnostic.
PIN={str(D):"f51ecedbd5b1435cc3fbb9fb580c986f2afbd82512e6005774c9654bb93a09f8",str(MAP):"96a37802391af69e6979eef327c002e915d179dcb8160be770c708d8743f55d0",str(VOL):"ee63458bda14096afc74f4314d047a2c876ccbcaac309767e00cc7e789a75bb9",str(PYREPORT):"bd8eeb0eacd08dab50e89dc86a1add9385891856d66005b9a6d4b129168e808e",str(PYRUN):""}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
for p,want in PIN.items():
 got=sha(p)
 if want and got!=want:raise ValueError(f"pinned input mismatch {p}: {got}")

x=loadnpz(D);m=loadnpz(MAP);v=loadnpz(VOL);pr=json.loads(PYREPORT.read_text());run=json.loads(PYRUN.read_text())
if pr["status"]!="DIAGNOSTIC_ONLY" or pr["physical_admission"] or pr["native_steps"]!=0:raise ValueError("pY run is not diagnostic-only")
if run["return_code"]!=0 or run["status"]!="TETGEN_COMPLETED_DIAGNOSTIC_ONLY":raise ValueError("pY TetGen input is not a successful diagnostic")
faces=np.asarray(x["faces"],np.int64);nface=len(faces);pts=np.asarray(x["points_world_m"]);p32=np.asarray(x["points_world_m_f32"],np.float32)
if pts.dtype!=np.float64 or p32.shape!=pts.shape:raise ValueError("unexpected current D point dtype/shape")
if not np.array_equal(pts.astype(np.float32).view(np.uint32),p32.view(np.uint32)):raise ValueError("D F64/F32 candidate points disagree")
# Verify the D vertex and face bridge by registered point ancestry, not row-number assumptions.
volrows=np.asarray(m["D_surface_point_to_volume_node_rows"],np.int64)
if len(volrows)!=len(pts) or len(np.unique(volrows))!=len(volrows):raise ValueError("D point-to-volume map is not complete and unique")
edge_global=np.asarray(pr["boundary_Steiner_details"][0]["source_edge_global_point_rows"],np.int64)
edge_t=float(pr["boundary_Steiner_details"][0]["projection_t"])
edge_rows=[]
for g in edge_global:
 rows=np.flatnonzero(volrows==g)
 if len(rows)!=1:raise ValueError(f"source endpoint global point {int(g)} not uniquely mapped to D")
 r=int(rows[0]);edge_rows.append(r)
 if not np.array_equal(p32[r].view(np.uint32),v["points_world_m"][int(g)].view(np.uint32)):raise ValueError(f"D endpoint row {r} not F32 exact to volume point {int(g)}")
# Source D faces are pinned through the preflight role/source rows and actual triangles.
parent_D_rows=[];target_rows=[]
for drow in pr["boundary_Steiner_details"][0]["incident_source_shell_faces"]:
 srcface=int(drow["global_source_point_rows"] and drow["packet_shell_face_row"])
 # The analyzer mapped each exact source shell face to a current and desired volume boundary row.
 tr=int(drow["desired_boundary_face_row"])
 if tr<0:raise ValueError("D source face has no current desired target row")
 if int(m["target_face_role_codes"][tr])!=4:raise ValueError("pY split source face is not D role code 4")
 if int(m["target_face_orientation_multiplier"][tr])!=-1:raise ValueError("D/Fat orientation relationship changed")
 srow=int(m["target_face_source_rows"][tr])
 if srow<0 or srow>=nface:raise ValueError("D target source face row outside current surface")
 expected_global=np.asarray(drow["global_source_point_rows"],np.int64)
 got_global=volrows[faces[srow]]
 if set(map(int,got_global))!=set(map(int,expected_global)):raise ValueError("source face row does not map to the authenticated D triangle")
 parent_D_rows.append(srow);target_rows.append(tr)
if len(set(parent_D_rows))!=2 or set(edge_rows)-set(faces[parent_D_rows].ravel()):raise ValueError("D source edge is not incident to the two authenticated D faces")
parents=np.asarray(parent_D_rows,np.int64)
# Read pY tables and verify the four Steiner boundary facets against the source edge faces.
def readtable(path):
 with path.open() as f:n=int(f.readline().split()[0])
 return np.atleast_2d(np.loadtxt(path,skiprows=1,max_rows=n))
nt=readtable(PYTABLES[0]);et=readtable(PYTABLES[1]);ft=readtable(PYTABLES[2])
nodeids=nt[:,0].astype(np.int64);nodexyz=nt[:,1:4].astype(np.float32);idrow={int(q):i for i,q in enumerate(nodeids)}
tri_ids=et[:,1:5].astype(np.int64);ot=np.asarray([[idrow[int(q)] for q in row]for row in tri_ids],np.int64)
new_node_id=int(pr["boundary_Steiner_details"][0]["tetgen_node_id"]);newnode_rows=np.flatnonzero(nodeids==new_node_id)
if len(newnode_rows)!=1:raise ValueError("pY boundary Steiner node ID is not unique")
newnode=int(newnode_rows[0]);newp=nodexyz[newnode]
if not np.array_equal(newp.view(np.uint32),np.asarray(pr["boundary_Steiner_details"][0]["actual_f32_world_m"],np.float32).view(np.uint32)):raise ValueError("pY table boundary point differs from analyzed actual F32 point")
if newnode not in set(range(len(nodexyz))):raise ValueError("invalid pY node row")
if np.any(np.all(p32.view(np.uint32)==newp.view(np.uint32),axis=1)):raise ValueError("new D split point already exists in D surface")
# D point coordinate -> local row. Need unique matches only for the four pY subfaces.
coord_index={tuple(map(int,row)):i for i,row in enumerate(p32.view(np.uint32))}
if len(coord_index)!=len(p32):raise ValueError("current D F32 point coordinates are not unique")
# Exact integer lattice coordinate helpers for orientation comparison.
spec=importlib.util.spec_from_file_location("cavity218_for_D_split",H/"remesh_complete_surface_cavities_218_root.py")
cav=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cav;spec.loader.exec_module(cav)
output_boundary_local=cav.m.boundary(ot)
expected_new_point_local=len(pts)
Dsplit={int(parents[0]):[[int(faces[parents[0],0]),edge_rows[0],expected_new_point_local],[int(faces[parents[0],0]),expected_new_point_local,edge_rows[1]]],int(parents[1]):[[int(faces[parents[1],0]),edge_rows[1],expected_new_point_local],[int(faces[parents[1],0]),expected_new_point_local,edge_rows[0]]]}
expected_global_sets={tuple(sorted(map(int,volrows[np.asarray(tri,np.int64)[np.asarray(tri,np.int64)<len(pts)]]))) for tri in []}
# Build four oriented volume boundary triangles and map their source vertices to D local rows.
vol_split=[]
for tri in output_boundary_local:
 if newnode not in set(map(int,tri)):continue
 mapped=[]
 for rr in tri:
  rr=int(rr)
  if rr==newnode:mapped.append(expected_new_point_local);continue
  key=tuple(map(int,nodexyz[rr].view(np.uint32)))
  dr=coord_index.get(key)
  if dr is None:raise ValueError("pY split boundary face has a vertex outside current D point domain")
  mapped.append(int(dr))
 vol_split.append(mapped)
if len(vol_split)!=4:raise ValueError(f"expected four pY output boundary subfaces around D edge, got {len(vol_split)}")
Dsplit_list=[tri for row in sorted(Dsplit) for tri in Dsplit[row]]
def cyclic(t):
 t=tuple(map(int,t));return min((t,(t[1],t[2],t[0]),(t[2],t[0],t[1])))
def reverse_cyclic(t):return cyclic((t[0],t[2],t[1]))
if {tuple(sorted(t)) for t in vol_split}!={tuple(sorted(t)) for t in Dsplit_list}:raise ValueError("pY boundary faces do not equal the proposed D split face set")
for tri in vol_split:
 if reverse_cyclic(tri) not in {cyclic(q) for q in Dsplit_list}:raise ValueError("pY cavity boundary orientation is not opposite current D orientation")
# Construct the current D derivative: preserve all points/faces except the declared edge split.
newfaces=faces.copy()
newrows=[int(parents[0]),int(parents[1]),nface,nface+1]
newfaces[int(parents[0])]=np.asarray(Dsplit[int(parents[0])][0],np.int64)
newfaces[int(parents[1])]=np.asarray(Dsplit[int(parents[1])][0],np.int64)
newfaces=np.vstack([newfaces,np.asarray([Dsplit[int(parents[0])][1],Dsplit[int(parents[1])][1]],np.int64)])
newpoints=np.vstack([pts,newp.astype(np.float64)])
newpoints32=np.vstack([p32,newp])
if newfaces.shape!=(nface+2,3):raise ValueError("unexpected D split face count")
# Build full-patch 7-field parent union from both retired parent faces, flattening inherited unions.
lanes=["layer","body","bone","face_id","original_id","identity0","identity1"]
parenttuples=[]
for r in parents:
 lo,hi=map(int,x["quality_star_patch_union_offsets"][r:r+2])
 if hi>lo:
  parenttuples.extend(tuple(int(x[f"quality_star_patch_union_face_source_{lane}"][j])for lane in lanes)for j in range(lo,hi))
 else:
  parenttuples.append(tuple(int(x[f"face_source_{lane}"][r])for lane in lanes))
union=sorted(set(parenttuples))
if len(union)<2:raise ValueError("two-parent split did not retain both typed source identities")
union_payload=json.dumps([list(t)for t in union],sort_keys=True,separators=(",",":"),ensure_ascii=True)
union_sha=hashlib.sha256(union_payload.encode("utf-8")).hexdigest()
# Flatten and preserve complete historical face ancestry from both retired parents.
ancestry_fields=("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")
old_ancestry_offsets=np.asarray(x["face_ancestry_offsets"],np.int64)
ancestry_union=[]
for rr in parents:
 alo,ahi=map(int,old_ancestry_offsets[int(rr):int(rr)+2])
 ancestry_union.extend(tuple(int(x[k][j]) for k in ancestry_fields) for j in range(alo,ahi))
ancestry_union=sorted(set(ancestry_union))
if not ancestry_union:raise ValueError("split parent faces have no authenticated face ancestry")
ancestry_payload=json.dumps([list(t) for t in ancestry_union],sort_keys=True,separators=(",",":"),ensure_ascii=True)
ancestry_sha=hashlib.sha256(ancestry_payload.encode("utf-8")).hexdigest()
# Both parents have a uniform physical stratum; keep those fields as the candidate-native descriptors.
for lane in ["layer","body","bone"]:
 vals={t[lanes.index(lane)]for t in union}
 if len(vals)!=1:raise ValueError(f"derived D patch has mixed {lane} descriptors; refusing single group type")
stratum={lane:union[0][lanes.index(lane)]for lane in ["layer","body","bone"]}
# Use a candidate-local unused ID for this generated patch and document its registry addition separately.
all_orig=np.concatenate([x["face_source_original_id"],x["quality_star_patch_union_face_source_original_id"]])
synthetic_id=int(np.max(all_orig))+1
if synthetic_id in set(map(int,all_orig)):raise ValueError("synthetic source ID collision")
patch_id="D-pY-edge-split-3c585-001"
# Derived identity tuple semantics: no single source face, row-unique output identity, patch ordinal.
# Extend the per-face typed columns for the two appended triangles before updating all four changed rows.
for lane in lanes:
 field=f"face_source_{lane}"
 old=np.asarray(x[field])
 x[field]=np.r_[old,np.zeros(2,dtype=old.dtype)]
for row in newrows:
 for lane,val in [("layer",stratum["layer"]),("body",stratum["body"]),("bone",stratum["bone"]),("face_id",-1),("original_id",synthetic_id),("identity0",int(row)),("identity1",0)]:
  x[f"face_source_{lane}"][row]=val
# Add the four faces as complete patch-union support, rebuilding the standard CSR.
newrowset=set(newrows);offsets=np.zeros(len(newfaces)+1,np.int64);rebuilt={lane:[] for lane in lanes}
ancestry_offsets=np.zeros(len(newfaces)+1,np.int64);ancestry_rebuilt={lane:[] for lane in ancestry_fields}
for r in range(len(newfaces)):
 if r in newrowset:
  tuples=union; ancestry=ancestry_union
 else:
  lo,hi=map(int,x["quality_star_patch_union_offsets"][r:r+2])
  tuples=[tuple(int(x[f"quality_star_patch_union_face_source_{lane}"][j])for lane in lanes)for j in range(lo,hi)]
  alo,ahi=map(int,old_ancestry_offsets[r:r+2])
  ancestry=[tuple(int(x[k][j]) for k in ancestry_fields) for j in range(alo,ahi)]
 for t in tuples:
  for j,lane in enumerate(lanes):rebuilt[lane].append(int(t[j]))
 offsets[r+1]=offsets[r]+len(tuples)
 for t in ancestry:
  for j,lane in enumerate(ancestry_fields):ancestry_rebuilt[lane].append(int(t[j]))
 ancestry_offsets[r+1]=ancestry_offsets[r]+len(ancestry)
x["face_ancestry_offsets"]=ancestry_offsets
for lane in ancestry_fields:x[lane]=np.asarray(ancestry_rebuilt[lane],np.int64)
x["points_world_m"]=newpoints;x["points_world_m_f32"]=newpoints32;x["faces"]=newfaces
x["quality_star_point_output_to_input_row"]=np.r_[np.asarray(x["quality_star_point_output_to_input_row"],np.int64),np.int64(-1)]
face_map=np.asarray(x["quality_star_face_output_to_input_row"],np.int64).copy()
face_map[parents]=-1
x["quality_star_face_output_to_input_row"]=np.r_[face_map,np.asarray([-1,-1],np.int64)]
# Exact immediate-parent maps for the existing changed-D auditor: all parent points survive once; the one appended point has no parent row.
x["point_output_to_parent_rows"]=np.r_[np.arange(len(p32),dtype=np.int64),np.int64(-1)]
parent_face_map=np.r_[np.arange(nface,dtype=np.int64),np.asarray([-1,-1],np.int64)]
parent_face_map[parents]=-1
x["face_output_to_parent_rows"]=parent_face_map
x["retired_parent_face_rows"]=parents.copy()
x["changed_parent_face_rows"]=np.asarray(newrows,dtype=np.int64)
x["immediate_parent_candidate_sha256"]=np.asarray([sha(D)],dtype="U64")
x["quality_star_patch_union_offsets"]=offsets
for lane in lanes:x[f"quality_star_patch_union_face_source_{lane}"]=np.asarray(rebuilt[lane],np.int64)
# The schema addition is intentionally separately declared because parent candidate did not carry a registry field.
operation={"schema":"numi.D-current-boundary-split-operation.v1","status":"DERIVED_LOCAL_BOUNDARY_SPLIT_PROPOSAL","physical_admission":False,"native_steps":0,"operation":patch_id,"derived_operation":"pY_boundary_Steiner_edge_split","role":"D","parent_candidate":{"path":str(D),"sha256":sha(D)},"parent_face_rows":parents.tolist(),"retired_parent_typed_union":union,"parent_union_tuple_count":len(union),"parent_union_tuple_sha256":union_sha,"parent_support_semantics":"Every generated D split face carries the complete deduplicated typed union of both retired edge-adjacent source faces. This is patch-level source support, not a per-triangle overlap or single-face ancestry claim.","face_ancestry_parent_union":ancestry_union,"face_ancestry_union_sha256":ancestry_sha,"synthetic_original_id":synthetic_id,"new_output_face_rows":newrows,"new_output_faces_D_row_domain":newfaces[newrows].tolist(),"new_output_face_source_tuple_by_row":{str(row):[int(x[f"face_source_{lane}"][row])for lane in lanes]for row in newrows},"point":{"output_D_point_row":int(expected_new_point_local),"pY_node_id":new_node_id,"pY_node_row":newnode,"world_f32_m":newp.tolist(),"world_f64_widened_m":newp.astype(np.float64).tolist(),"source_D_edge_rows":edge_rows,"source_global_volume_edge_rows":edge_global.tolist(),"barycentric_from_first_global_endpoint":edge_t,"actual_f32_edge_distance_m":float(pr["boundary_Steiner_details"][0]["actual_f32_to_exact_edge_distance_m"]),"source_faces_D_rows":parents.tolist()},"current_volume_target_face_rows":target_rows,"current_volume_target_face_roles":[int(m["target_face_role_codes"][r])for r in target_rows],"current_volume_target_face_orientation_multipliers":[int(m["target_face_orientation_multiplier"][r])for r in target_rows],"pY_diagnostic":{"report_path":str(PYREPORT),"report_sha256":sha(PYREPORT),"run_result_path":str(PYRUN),"run_result_sha256":sha(PYRUN),"poly_sha256":run["poly_sha256"],"output_node_sha256":run["output_files"]["cavity.1.node"]["sha256"],"output_ele_sha256":run["output_files"]["cavity.1.ele"]["sha256"],"output_face_sha256":run["output_files"]["cavity.1.face"]["sha256"],"pY_boundary_faces_mapped_to_D_patch":len(vol_split)},"checks":{"endpoint_F32_source_map_exact":True,"two_parent_faces_are_D_role":True,"D_point_rows_not_shared_with_I":True,"pY_faces_equal_D_split_unordered":True,"pY_boundary_orientation_opposes_D":True,"F32_point_coordinates_unchanged_outside_appended_point":True},"source_identity_schema_addition":{"key":str(synthetic_id),"role":"DERIVED_PATCH","derived_operation":"pY_boundary_Steiner_edge_split","derived_patch_id":patch_id,"derived_patch_ordinal":0,"face_count":4,"identity_face_id_key":"face_source_face_id","identity_keys":["face_source_identity0","face_source_identity1"],"parent_union_tuple_count":len(union),"parent_union_tuple_sha256":union_sha,"complete_typed_parent_union":union,"parent_face_row_domain":"current_D030_face_rows","parent_support_semantics":"Patch-level union; no per-triangle overlap claim.","source_lineage":{"parent_candidate_path":str(D),"parent_candidate_sha256":sha(D),"parent_face_rows":parents.tolist(),"source_global_volume_point_rows":sorted(set(map(int,edge_global)))}},"schema_registry_handling":"Parent current-D030 NPZ has no face_source_identity_schema_by_original_id_json. This derivative carries an explicit addition object and operation receipt; a downstream complete registry must merge this entry without treating it as a complete registry."}
op=OUT/(patch_id+".json");op.write_text(json.dumps(operation,indent=2,sort_keys=True)+"\n");opsha=sha(op)
# Keep output->input rows honest: changed/derived faces and new point have no single exact identity.
x["derived_boundary_split_operation_json_path"]=np.asarray([str(op)],dtype="U")
x["derived_boundary_split_operation_json_sha256"]=np.asarray([opsha],dtype="U64")
x["derived_boundary_split_parent_face_rows"]=parents
x["derived_boundary_split_output_face_rows"]=np.asarray(newrows,np.int64)
x["derived_boundary_split_patch_parent_union_tuples"]=np.asarray(union,np.int64)
x["derived_boundary_split_patch_parent_union_sha256"]=np.asarray([union_sha],dtype="U64")
x["derived_boundary_split_face_ancestry_union_tuples"]=np.asarray(ancestry_union,np.int64)
x["derived_boundary_split_face_ancestry_union_sha256"]=np.asarray([ancestry_sha],dtype="U64")
x["derived_boundary_split_point_row"]=np.asarray(expected_new_point_local,np.int64)
x["derived_boundary_split_point_source_D_edge_rows"]=np.asarray(edge_rows,np.int64)
x["derived_boundary_split_point_source_global_volume_edge_rows"]=edge_global
x["derived_boundary_split_point_barycentric"]=np.asarray([1.0-edge_t,edge_t],np.float64)
# Local closed surface/orientation checks for the D candidate (not the global exact collision gate).
if len({tuple(sorted(map(int,f)))for f in newfaces})!=len(newfaces):raise ValueError("duplicate D face after split")
edge_dir={}
for fi,f in enumerate(newfaces):
 for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
  k=tuple(sorted((a,b)));edge_dir.setdefault(k,[]).append((a,b,fi))
edge_bad=[(k,vv)for k,vv in edge_dir.items()if len(vv)!=2 or vv[0][0]==vv[1][0]]
def directed_boundary_signature(F):
 dd={}
 for ff in F:
  for a,b in ((int(ff[0]),int(ff[1])),(int(ff[1]),int(ff[2])),(int(ff[2]),int(ff[0]))):
   kk=tuple(sorted((a,b)));dd.setdefault(kk,[]).append((a,b))
 return {k:tuple(sorted(v)) for k,v in dd.items() if len(v)!=2 or v[0][0]==v[1][0]}
edge_bad_before=directed_boundary_signature(faces)
edge_bad_after=directed_boundary_signature(newfaces)
if edge_bad_before!=edge_bad_after:raise ValueError(f"D split changes existing directed boundary: before={len(edge_bad_before)} after={len(edge_bad_after)}")
# Positive orientation and nondegeneracy of each new face relative to its parent normal.
xyz=newpoints32.astype(np.float64);orientation=[];quality=[]
for r in newrows:
 parent=int(parents[0] if r in Dsplit[int(parents[0])] else parents[1])
 old=faces[parent];n0=np.cross(p32[old[1]].astype(np.float64)-p32[old[0]],p32[old[2]].astype(np.float64)-p32[old[0]])
 tri=newfaces[r];n1=np.cross(xyz[tri[1]]-xyz[tri[0]],xyz[tri[2]]-xyz[tri[0]])
 dot=float(np.dot(n0,n1));orientation.append(dot>0)
 lens=[float(np.linalg.norm(xyz[tri[(i+1)%3]]-xyz[tri[i]]))for i in range(3)];area2=float(np.linalg.norm(n1));qbound=area2/(max(lens)**2)
 quality.append({"face_row":r,"double_area_m2":area2,"max_edge_m":max(lens),"normalized_surface_bound":qbound,"parent_normal_dot":dot})
if not all(orientation):raise ValueError("split face reverses an input parent normal")
if np.any(newpoints32[-1].view(np.uint32)!=newp.view(np.uint32)):raise ValueError("new D point changed F32 bits")
# Save NPZ and full scope receipt.
outnpz=OUT/"candidate-D-pY-boundary-split.npz"
np.savez_compressed(outnpz,**x)
report={"schema":"numi.D-current-pY-boundary-split-preparation.v1","status":"PASS_LOCAL_D_BOUNDARY_SPLIT_CONSTRUCTION_ONLY","physical_admission":False,"native_steps":0,"input_D_path":str(D),"input_D_sha256":sha(D),"target_map_path":str(MAP),"target_map_sha256":sha(MAP),"volume_input_path":str(VOL),"volume_input_sha256":sha(VOL),"candidate_path":str(outnpz),"candidate_sha256":sha(outnpz),"operation_json_path":str(op),"operation_json_sha256":opsha,"source_identity_schema_addition":operation["source_identity_schema_addition"],"parent_D_face_rows":parents.tolist(),"new_output_D_face_rows":newrows,"new_point_D_row":expected_new_point_local,"D_points_before_after":[len(pts),len(newpoints)],"D_faces_before_after":[len(faces),len(newfaces)],"local_directed_surface_edges_bad":len(edge_bad_after),"directed_boundary_edges_before_after":[len(edge_bad_before),len(edge_bad_after)],"directed_boundary_signature_unchanged":edge_bad_before==edge_bad_after,"local_duplicate_faces":0,"local_new_face_parent_normal_positive":orientation,"local_new_face_quality":quality,"pY_exact_target_face_rows":target_rows,"pY_fat_boundary_orientation_is_opposite_D":True,"pY_boundary_Steiner_distance_m":operation["point"]["actual_f32_edge_distance_m"],"all_other_point_coordinates_bitwise_unchanged":bool(np.array_equal(newpoints[:-1].view(np.uint64),pts.view(np.uint64))),"all_other_face_rows_unchanged_except_two_retired_faces_and_two_appended_faces":True,"full_D_self_and_D_vs_I/O/all185_exact_gates":"NOT RUN","point_anatomical_clearance_after_move":"NOT RUN","surface_quality_gate":"Local metrics only; no global qualification","scope":"A local current-D boundary-split proposal conforming to the captured pY TetGen diagnostic. Not a volume composition or physical admission."}
reportp=OUT/"report.json";reportp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report_path":str(reportp),"report_sha256":sha(reportp),"candidate_path":str(outnpz),"candidate_sha256":sha(outnpz),"operation_json_path":str(op),"operation_json_sha256":opsha,"status":report["status"],"parents":parents.tolist(),"new_rows":newrows,"newpoint":expected_new_point_local,"D_face_counts":[len(faces),len(newfaces)],"local_surface_edge_bad":len(edge_bad),"pY_boundary_faces":len(vol_split),"synthetic_original_id":synthetic_id,"union_sha256":union_sha},indent=2))
