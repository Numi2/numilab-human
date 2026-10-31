#!/usr/bin/env python3
"""Replay frozen Fat614 comp2/comp6 TetGen cavity topology on the 336 conditioned descendant.

This is a source-only local volume proposal. It binds the rounded D boundary points,
uses frozen .1 tables, preserves the 336 connectivity increment, and does not claim
complete physical admission, mass allocation, or native execution.
"""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"fat614-yy-volume-replay-002"
OUT.mkdir(parents=True, exist_ok=True)
BASE=H/"fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz"
PART=H/"fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz"
PREP=H/"fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz"
REMR=H/"fat614-compact-cavities-004-remesh-002/report.json"
DESC=H/"fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz"
DESC_REPORT=H/"fat-inferred-identity-mask-336-root/fat-run001/report.json"
DESC_INDEPENDENT=H/"fat-inferred-identity-mask-336-root/independent326001/receipt.json"
D_CAND=H/"fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz"
D_AUDIT=H/"fat614-minimal-f32-target-split-001/exact-audit-001/result-001/report.json"
STRUCTURAL=H/"fat614-candidate-structural-audit-001/receipt.json"
CASE_ROOT=H/"fat614-compact-cavities-004-remesh-002"
EXPECTED={
 "base":("c21b708fa591476b1c5ab5f33329362ac8086c6429c73a117e63100555984e8b",BASE),
 "partial_4602":("4602edc3b1db3617ab610c21f91309db10892c0e28ff7d6ba5c381491d13df2c",PART),
 "prepared_shell":("5810a3139b619630b8658c407a3b22cde1939ba9c917d273852622743ce5fe44",PREP),
 "remesh_report":("f2d82ac0d677e63755c43c76ffaf9ed65e63ca73b3f3b49b9da7d5e8758934e0",REMR),
 "descendant_336":("2fc5aa8930fe50f940b071a954c0393b8e41afee514d64ace81e1c419e2b1905",DESC),
 "descendant_report":("2c2861083dc2e38ab4d9c3d9c62f05504c224c3159a3e89e9b6cac02412637a0",DESC_REPORT),
 "D_split_candidate":("602036c9b164bd1eec83b344675b8cabf453665920e45b558b50fe8115c67493",D_CAND),
 "D_full_exact_audit":("79533b3f38c4a202463c90c0b426a561641500c1465c6558406cde6077971bb2",D_AUDIT),
 "structural_receipt":("540b97c344903fb0d3e4ecdbbaad20f64d4667b1ac59d5f4f52e23373b15f4c3",STRUCTURAL),
}
for name,(want,path) in EXPECTED.items():
 got=hashlib.sha256(path.read_bytes()).hexdigest()
 if got!=want: raise ValueError(f"{name} SHA mismatch: {got}")

def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return mod
r=loadmod("fat614_remesh296_007",H/"remesh_fat614_compact_cavities_296_007.py")
m=r.m
caps=loadmod("fat614_caps288",H/"direct_connected_skin_caps_288_root.py")
owner=H.parent/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(owner.parent.parent))
exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])

def read_json(p): return json.loads(Path(p).read_text())
def table(path):
 lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 count=int(lines[0].split()[0])
 rows=np.asarray([[float(v) for v in line.split()] for line in lines[1:]],np.float64)
 if len(rows)!=count: raise ValueError(f"{path}: declared {count} records but read {len(rows)}")
 return rows
def f32key(x): return m.key(np.asarray(x,np.float32))
def canonical_face(f): return tuple(sorted(map(int,f)))
def face_counter(faces):
 ident=m.boundary_identity(np.asarray(faces,np.int64).reshape(-1,3))
 return Counter(tuple(map(int,row)) for row in ident)
def counter_delta(actual,wanted):
 got=face_counter(actual);want=face_counter(wanted)
 missing=want-got;extra=got-want
 return {
  "missing_oriented_records":int(sum(missing.values())),
  "extra_oriented_records":int(sum(extra.values())),
  "missing_samples":[list(k) for k in sorted(missing)[:12]],
  "extra_samples":[list(k) for k in sorted(extra)[:12]],
 }
def exact_flux(points,faces): return int(r.exact_volume(np.asarray(points,np.float32),np.asarray(faces,np.int64)))
def orient_split(face,u,v,newrow):
 f=list(map(int,face));i=f.index(int(u));j=f.index(int(v))
 w=next(x for x in f if x not in (int(u),int(v)))
 if (j-i)%3==1:return [[int(u),int(newrow),w],[int(newrow),int(v),w]]
 if (i-j)%3==1:return [[int(v),int(newrow),w],[int(newrow),int(u),w]]
 raise ValueError("edge endpoints are not adjacent in target face")
def split_global_faces(faces,u,v,newrow,old_faces,label):
 oldkeys={canonical_face(x) for x in old_faces}
 got=[x for x in np.asarray(faces,np.int64) if canonical_face(x) in oldkeys]
 if len(got)!=len(old_faces) or len({canonical_face(x) for x in got})!=len(old_faces):
  raise ValueError(f"{label}: expected each of {len(old_faces)} target parent faces exactly once, found {len(got)}")
 out=[]
 for f in np.asarray(faces,np.int64):
  if canonical_face(f) in oldkeys: out.extend(orient_split(f,u,v,newrow))
  else: out.append(list(map(int,f)))
 return np.asarray(out,np.int64),[list(map(int,x)) for x in got]
def load_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def exact_tet_volume(points,tets):
 ip=m.m.integer_points(np.asarray(points,np.float32))
 vals=[int(m.m.exact_det(ip,t)) for t in np.asarray(tets,np.int64)]
 return vals

base=load_npz(BASE);part=load_npz(PART);prep=load_npz(PREP);desc=load_npz(DESC);drole=load_npz(D_CAND)
for receipt_path,want_status in [(DESC_INDEPENDENT,"PASS_INDEPENDENT_INFERRED_INTERIOR_RETIREMENT_REPLAY")]:
 receipt=read_json(receipt_path)
 if receipt.get("status")!=want_status or receipt.get("physical_admission") is not False or receipt.get("native_steps")!=0:
  raise ValueError(f"descendant independent receipt is not the expected source-only PASS: {receipt.get('status')}")
 if receipt.get("proposal_sha256")!=EXPECTED["descendant_336"][0] or receipt.get("report_sha256")!=EXPECTED["descendant_report"][0]:
  raise ValueError("descendant replay receipt does not bind exact candidate/report")
d_audit=read_json(D_AUDIT)
if d_audit.get("physical_admission") is not False or d_audit.get("native_steps")!=0:
 raise ValueError("D exact surface audit must remain source-only")
if d_audit.get("status") not in ("PASS_EXACT_CURRENT_SURFACE_CONTEXT","PASS_EXACT_CHANGED_SURFACE_CONTEXT","PASS_EXACT_SURFACE_CONTEXT"):
 # The report schema is owner-authenticated by the pinned banks receipt; retain status verbatim if its spelling differs.
 if not str(d_audit.get("status","")).startswith("PASS"):
  raise ValueError(f"D surface audit not PASS: {d_audit.get('status')}")

p0=np.asarray(base["points_world_m"],np.float32)
t0=np.asarray(base["tetrahedra"],np.int64)
mat0=np.asarray(base["material_indices"],np.int32)
ppart=np.asarray(part["points_world_m"],np.float32)
tpart=np.asarray(part["tetrahedra"],np.int64)
mpart=np.asarray(part["material_indices"],np.int32)
pdesc=np.asarray(desc["points_world_m"],np.float32)
tdesc=np.asarray(desc["tetrahedra"],np.int64)
mdesc=np.asarray(desc["material_indices"],np.int32)
if not np.array_equal(p0,ppart) or not np.array_equal(ppart,pdesc):
 raise ValueError("base, 4602, and 336 point arrays must be bitwise identical")
if not np.array_equal(part["desired_boundary_faces"],desc["desired_boundary_faces"]):
 raise ValueError("336 changed desired target boundary before D split")
desc_boundary=m.boundary(tdesc)
if face_counter(desc_boundary)!=face_counter(desc["boundary_faces"]):
 raise ValueError("336 boundary_faces array does not replay from its exact tetrahedra")
if len(desc["replacement_tetrahedra"])!=len(tdesc)-len(desc["retained_input_cell_rows"]):
 raise ValueError("336 replacement tail length mismatch")
if not np.array_equal(desc["replacement_tetrahedra"],tdesc[len(desc["retained_input_cell_rows"]):]):
 raise ValueError("336 replacement tail does not match candidate tetrahedron ordering")
desc_receipt=read_json(DESC_INDEPENDENT)

# The 336 candidate's direct retained-cell map names 4602 input cell rows.
part_source_to_row={int(src):i for i,src in enumerate(np.asarray(part["retained_input_cell_rows"],np.int64))}
if len(part_source_to_row)!=len(part["retained_input_cell_rows"]):
 raise ValueError("4602 retained source row map is not unique")
desc_input_to_output={int(src):i for i,src in enumerate(np.asarray(desc["retained_input_cell_rows"],np.int64))}
if len(desc_input_to_output)!=len(desc["retained_input_cell_rows"]):
 raise ValueError("336 direct retained input row map is not unique")
base_parent_rows=np.asarray(prep["input_cell_rows"],np.int64)
cell_offsets=np.asarray(prep["cell_offsets"],np.int64)
point_offsets=np.asarray(prep["point_offsets"],np.int64)
face_offsets=np.asarray(prep["face_offsets"],np.int64)
component_specs={
 2:{"edge":(217282,221017),"d_point_row":120634,"target_faces":[[217282,221017,220607],[217282,220616,221017]],"expected_boundary_node":1393},
 6:{"edge":(188532,198523),"d_point_row":120633,"target_faces":[[188532,195232,198523],[188532,198523,195234]],"expected_boundary_node":21},
}
with np.load(D_CAND,allow_pickle=False) as z:
 d_point_rows=np.asarray(z["derived_boundary_split_point_rows"],np.int64)
 d_edges=np.asarray(z["derived_boundary_split_point_source_global_volume_edge_rows"],np.int64)
 d_f32=np.asarray(z["points_world_m_f32"],np.float32)
if set(map(int,d_point_rows))!={120633,120634}: raise ValueError("D point rows differ from authenticated two-point split")
d_edge_by_row={int(row):tuple(map(int,edge)) for row,edge in zip(d_point_rows,d_edges)}
for ci,spec in component_specs.items():
 if set(d_edge_by_row[spec["d_point_row"]])!=set(spec["edge"]):
  raise ValueError(f"component {ci}: D split edge rows do not match frozen witness")
 if not np.array_equal(drole["points_world_m_f32"][spec["d_point_row"]],d_f32[spec["d_point_row"]]):
  raise ValueError(f"component {ci}: inconsistent D F32 point view")

# Rebase the frozen source cavity rows from c21b -> 4602 -> 336 by exact retained identity.
comp={};deleted_desc=[];added_global_tets=[];all_point_map={}
for ci,spec in component_specs.items():
 lo,hi=int(cell_offsets[ci]),int(cell_offsets[ci+1])
 source_rows=np.asarray(prep["input_cell_rows"][lo:hi],np.int64)
 if len(np.unique(source_rows))!=len(source_rows):raise ValueError(f"component {ci}: duplicate source cell rows")
 try:part_rows=np.asarray([part_source_to_row[int(x)] for x in source_rows],np.int64)
 except KeyError as e:raise ValueError(f"component {ci}: source row absent from 4602 retained map: {e}")
 try:desc_rows=np.asarray([desc_input_to_output[int(x)] for x in part_rows],np.int64)
 except KeyError as e:raise ValueError(f"component {ci}: 4602 input cell not retained by 336: {e}")
 if np.any(desc_rows>=len(desc["retained_input_cell_rows"])):
  raise ValueError(f"component {ci}: source cell maps outside 336 retained block")
 if not np.array_equal(t0[source_rows],tpart[part_rows]):
  raise ValueError(f"component {ci}: 4602 source-cell tuple differs from c21b exact ordered tuple")
 if not np.array_equal(tpart[part_rows],tdesc[desc_rows]):
  raise ValueError(f"component {ci}: 336 descendant did not preserve the exact ordered source-cell tuple")
 if np.any(mat0[source_rows]!=16) or np.any(mpart[part_rows]!=16) or np.any(mdesc[desc_rows]!=16):
  raise ValueError(f"component {ci}: selected old cells are not uniformly material 16")
 desc_parent_domain=set(map(int,np.asarray(desc["replacement_parent_cell_rows"],np.int64)))
 if set(map(int,part_rows)) & desc_parent_domain:
  raise ValueError(f"component {ci}: source cavity overlaps a 336 replacement parent operation")
 deleted_desc.extend(map(int,desc_rows))
 growlo,growhi=int(point_offsets[ci]),int(point_offsets[ci+1])
 grows=np.asarray(prep["global_point_rows"][growlo:growhi],np.int64)
 flo,fhi=int(face_offsets[ci]),int(face_offsets[ci+1])
 local_faces=np.asarray(prep["local_faces"][flo:fhi],np.int64)
 shell_global=grows[local_faces]
 if len(np.unique(grows))!=len(grows):raise ValueError(f"component {ci}: duplicate shell source point")
 if not np.all(np.isin(shell_global,grows)):raise ValueError(f"component {ci}: bad shell point map")
 old_component_boundary=m.boundary(tdesc[desc_rows])
 old_shell_delta=counter_delta(old_component_boundary,shell_global)
 case=CASE_ROOT/f"component-{ci:04d}"
 for filename in ("cavity.poly","cavity.1.node","cavity.1.ele","cavity.1.face"):
  if not (case/filename).is_file():raise ValueError(f"component {ci} missing frozen TetGen {filename}")
 spec["source_rows"]=source_rows
 spec["partial_input_rows"]=part_rows
 spec["descendant_output_rows"]=desc_rows
 spec["component_global_point_rows"]=grows
 spec["input_shell_faces_global"]=shell_global
 spec["local_input_faces"]=local_faces
 spec["case"]=case
 comp[ci]=spec
if len(set(deleted_desc))!=len(deleted_desc):raise ValueError("component 2/6 source-cell domains overlap")

# Append only the two authenticated boundary points first; all other TetGen nodes are exact F32 interior points.
point_list=[x.copy() for x in pdesc]
point_index={f32key(x):i for i,x in enumerate(pdesc)}
if len(point_index)!=len(pdesc):raise ValueError("336 points contain duplicate F32 coordinate keys")
for ci in (6,2):
 row=component_specs[ci]["d_point_row"];x=d_f32[row]
 k=f32key(x)
 if k in point_index:raise ValueError(f"D target point row {row} already exists in volume point domain")
 point_index[k]=len(point_list);component_specs[ci]["global_boundary_point_row"]=len(point_list);point_list.append(x.copy())

def parse_poly_nodes(poly_path):
 lines=[x.strip() for x in Path(poly_path).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 head=lines[0].split();n=int(head[0])
 if len(head)<4 or int(head[1])!=3:raise ValueError("unexpected TetGen .poly point header")
 rec=[]
 for line in lines[1:1+n]:
  q=line.split()
  if len(q)<4:raise ValueError("malformed .poly node line")
  rec.append((int(q[0]),np.asarray([float(q[1]),float(q[2]),float(q[3])],np.float32)))
 return rec
def f32table(path):
 a=table(path)
 return a

for ci in (2,6):
 spec=component_specs[ci];case=spec["case"]
 poly_nodes=parse_poly_nodes(case/"cavity.poly")
 node=f32table(case/"cavity.1.node")
 ele=f32table(case/"cavity.1.ele")
 faces=f32table(case/"cavity.1.face")
 node_ids=node[:,0].astype(np.int64);xyz=np.asarray(node[:,1:4],np.float32)
 ele_ids=ele[:,0].astype(np.int64);elem_node_ids=ele[:,1:5].astype(np.int64)
 face_ids=faces[:,0].astype(np.int64);face_node_ids=faces[:,1:4].astype(np.int64)
 if len(np.unique(node_ids))!=len(node_ids) or len(np.unique(ele_ids))!=len(ele_ids) or len(np.unique(face_ids))!=len(face_ids):
  raise ValueError(f"component {ci}: duplicate TetGen node/element/face IDs")
 id_to_local={int(q):i for i,q in enumerate(node_ids)}
 if len(ele)>0 and (ele.shape[1]<6 or not np.all(ele[:,5].astype(np.int64)==16)):
  raise ValueError(f"component {ci}: TetGen element material is not 16")
 if len(set(map(int,elem_node_ids.ravel()))-set(id_to_local)):
  raise ValueError(f"component {ci}: element references missing node ID")
 if len(set(map(int,face_node_ids.ravel()))-set(id_to_local)):
  raise ValueError(f"component {ci}: face references missing node ID")
 input_ids={int(q) for q,_ in poly_nodes}
 if not input_ids.issubset(set(map(int,node_ids))):
  raise ValueError(f"component {ci}: TetGen output lost an input PLC node")
 output_pos={int(node_ids[i]):xyz[i] for i in range(len(node_ids))}
 for nid,x in poly_nodes:
  if nid not in output_pos or not np.array_equal(output_pos[nid].view(np.uint32),x.view(np.uint32)):
   raise ValueError(f"component {ci}: TetGen moved an input PLC point")
 # All original .poly points must map exactly to this component's source point set.
 local_source_keys={f32key(pdesc[int(row)]):int(row) for row in spec["component_global_point_rows"]}
 if len(local_source_keys)!=len(spec["component_global_point_rows"]):
  raise ValueError(f"component {ci}: source component point keys not unique")
 mapped_poly_rows=[]
 for nid,x in poly_nodes:
  k=f32key(x)
  if k not in point_index:raise ValueError(f"component {ci}: input PLC node has no exact candidate point")
  global_row=int(point_index[k]);mapped_poly_rows.append(global_row)
  if global_row not in set(map(int,spec["component_global_point_rows"])):
   raise ValueError(f"component {ci}: .poly source node maps outside its authenticated cavity point set")
 if set(mapped_poly_rows)!=set(map(int,spec["component_global_point_rows"])):
  raise ValueError(f"component {ci}: .poly source point set differs from prepared cavity point set")
 # Only the one authenticated target-edge D point may be a new boundary node.
 output_face_node_ids=set(map(int,face_node_ids.ravel()))
 new_ids=set(map(int,node_ids))-input_ids
 new_boundary_ids=new_ids&output_face_node_ids
 if new_boundary_ids!={int(spec["expected_boundary_node"])}:
  raise ValueError(f"component {ci}: new boundary node IDs {sorted(new_boundary_ids)} differ from authenticated split witness")
 if int(spec["expected_boundary_node"]) not in id_to_local:
  raise ValueError(f"component {ci}: expected target boundary node absent")
 # Every emitted node must participate in a tetrahedron or a boundary face.
 used_ids=set(map(int,elem_node_ids.ravel()))|output_face_node_ids
 if not used_ids.issubset(set(map(int,node_ids))):
  raise ValueError(f"component {ci}: TetGen elements/faces reference an absent node ID")
 unused_output_node_ids=sorted(set(map(int,node_ids))-used_ids)
 # The frozen return-code-200 run can leave unreferenced trial nodes in .1.node.
 # Keep those bytes pinned as input evidence, but do not add their unused coordinates.
 node_global=np.full(len(node_ids),-1,np.int64)
 for i,x in enumerate(xyz):
  if int(node_ids[i]) not in used_ids:continue
  k=f32key(x)
  if k in point_index:
   node_global[i]=int(point_index[k])
  else:
   row=len(point_list);point_index[k]=row;point_list.append(x.copy());node_global[i]=row
 if int(node_global[id_to_local[int(spec["expected_boundary_node"])]] )!=int(spec["global_boundary_point_row"]):
  raise ValueError(f"component {ci}: TetGen target boundary node does not map to its D candidate row")
 local_to_global={int(nid):int(node_global[i]) for i,nid in enumerate(node_ids)}
 out_tets_local=np.asarray([[id_to_local[int(n)] for n in row] for row in elem_node_ids],np.int64)
 out_tets_global=node_global[out_tets_local]
 out_faces_global=np.asarray([[local_to_global[int(n)] for n in row] for row in face_node_ids],np.int64)
 # Construct the exact current target/cut shell by splitting only the two authorized D parent triangles.
 u,v=spec["edge"];drow=int(spec["global_boundary_point_row"])
 expected_local,seen_old=split_global_faces(spec["input_shell_faces_global"],u,v,drow,spec["target_faces"],f"component-{ci}")
 expected_global_desired,seen_target=split_global_faces(np.asarray(desc["desired_boundary_faces"],np.int64),u,v,drow,spec["target_faces"],f"whole-target-{ci}")
 if len(expected_local)!=len(spec["input_shell_faces_global"])+2:
  raise ValueError(f"component {ci}: local shell split did not add two faces")
 if len(seen_target)!=2:raise ValueError(f"component {ci}: full target split did not cover both parent faces")
 local_boundary=m.boundary(out_tets_global)
 if face_counter(local_boundary)!=face_counter(expected_local):
  raise ValueError(f"component {ci}: TetGen cell boundary does not match exact target/cut shell")
 face_file_counts=Counter(canonical_face(f) for f in out_faces_global)
 expected_face_counts=Counter(canonical_face(f) for f in expected_local)
 if face_file_counts!=expected_face_counts:
  raise ValueError(f"component {ci}: TetGen .1.face triangles do not match exact target/cut shell")
 if face_file_counts!=Counter(canonical_face(f) for f in local_boundary):
  raise ValueError(f"component {ci}: TetGen .1.face triangles differ from tetrahedron boundary")
 # Local exact shell self-intersection and exact source/new volume-flux replay.
 local_records,_=exact.float32_surface_records(np.asarray(point_list,np.float32),expected_local)
 shell_self=exact._audit_pair(local_records,local_records,same_surface=True)
 if int(shell_self.get("count",0))!=0:
  raise ValueError(f"component {ci}: target/cut shell self-intersects")
 old_tets=tdesc[spec["descendant_output_rows"]]
 old_det=exact_tet_volume(pdesc,old_tets)
 if any(x<=0 for x in old_det):raise ValueError(f"component {ci}: selected source has exact nonpositive tets")
 old_boundary=m.boundary(old_tets)
 old_flux=exact_flux(pdesc,old_boundary)
 old_volume=sum(old_det)
 if old_volume!=old_flux:raise ValueError(f"component {ci}: source tet volume differs from old oriented boundary flux")
 new_det=exact_tet_volume(np.asarray(point_list,np.float32),out_tets_global)
 if any(x<=0 for x in new_det):raise ValueError(f"component {ci}: TetGen output has exact F32 nonpositive tet")
 new_volume=sum(new_det);new_flux=exact_flux(np.asarray(point_list,np.float32),expected_local)
 if new_volume!=new_flux:raise ValueError(f"component {ci}: new TetGen volume differs from revised oriented shell flux")
 if new_volume-old_volume!=new_flux-old_flux:
  raise ValueError(f"component {ci}: exact local volume delta differs from boundary flux delta")
 drowcoord=np.asarray(point_list[int(spec["global_boundary_point_row"])],np.float64)
 A=pdesc[int(u)].astype(np.float64);B=pdesc[int(v)].astype(np.float64);edge=B-A
 segt=float(np.dot(drowcoord-A,edge)/np.dot(edge,edge))
 perp=float(np.linalg.norm(drowcoord-(A+segt*edge)))
 if not 0.0<segt<1.0:raise ValueError(f"component {ci}: D boundary point is outside target edge segment")
 spec.update({
  "tetgen_node_ids":node_ids,
  "tetgen_node_global_rows":node_global,
  "tetgen_output_tetrahedra_global_rows":out_tets_global,
  "tetgen_output_face_global_rows":out_faces_global,
  "local_expected_output_boundary":expected_local,
  "old_local_volume_det6":old_volume,
  "new_local_volume_det6":new_volume,
  "old_local_boundary_flux_det6":old_flux,
  "new_local_boundary_flux_det6":new_flux,
  "local_volume_delta_det6":new_volume-old_volume,
  "new_tets_exact_positive":len(new_det),
  "new_tets_native_failures":int(np.sum(~m.m.passes(*m.m.measurements(np.asarray(point_list,np.float32),out_tets_global)))),
  "new_boundary_node_count":len(new_boundary_ids),
  "new_interior_nodes":len((new_ids&used_ids)-new_boundary_ids),
  "unused_output_node_ids":unused_output_node_ids,
  "d_point_segment_parameter":segt,
  "d_point_support_residual_m":perp,
  "shell_self_intersections":int(shell_self.get("count",0)),
  "tetgen_face_file_unoriented_exact":True,
  "tetgen_face_file_orientation_matches_outward_tet_boundary":face_counter(out_faces_global)==face_counter(local_boundary),
  "new_tets":out_tets_global,
 })
 added_global_tets.append(out_tets_global)

# Build the descendant candidate by removing only the authenticated direct source rows and appending both frozen remesh blocks.
deleted_desc=np.asarray(sorted(set(deleted_desc)),np.int64)
retained=np.asarray(desc["retained_input_cell_rows"],np.int64)
if np.any(deleted_desc>=len(retained)):raise ValueError("component source rows do not map to direct retained 336 cells")
keep=np.ones(len(tdesc),bool);keep[deleted_desc]=False
if not np.all(keep[len(retained):]):raise ValueError("unexpected overlap with existing 336 replacement tail")
# Every local non-target shell face must oppose exactly one untouched descendant tetrahedron.
unselected_boundary=m.boundary(tdesc[keep])
unselected_oriented=face_counter(unselected_boundary)
desired_after6,_=split_global_faces(np.asarray(desc["desired_boundary_faces"],np.int64),*comp[6]["edge"],comp[6]["global_boundary_point_row"],comp[6]["target_faces"],"cut-classification-target-comp6")
desired_afterboth,_=split_global_faces(desired_after6,*comp[2]["edge"],comp[2]["global_boundary_point_row"],comp[2]["target_faces"],"cut-classification-target-comp2")
updated_target_keys={canonical_face(f) for f in desired_afterboth}
original_target_keys={canonical_face(f) for f in np.asarray(desc["desired_boundary_faces"],np.int64)}
whole_old_boundary_keys={canonical_face(f) for f in desc_boundary}
old_selected_union_boundary=m.boundary(tdesc[deleted_desc])
old_internal_cut=np.asarray([f for f in old_selected_union_boundary if canonical_face(f) not in whole_old_boundary_keys and canonical_face(f) not in original_target_keys],np.int64).reshape(-1,3)
new_internal_cut=np.asarray([f for ci in (2,6) for f in comp[ci]["local_expected_output_boundary"] if canonical_face(f) not in updated_target_keys],np.int64).reshape(-1,3)
old_cut_counter=face_counter(old_internal_cut);new_cut_counter=face_counter(new_internal_cut)
cut_missing=new_cut_counter-old_cut_counter;cut_extra=old_cut_counter-new_cut_counter
if cut_missing or cut_extra:
 raise ValueError(f"two-direction oriented internal cut-set mismatch: output_only={sum(cut_missing.values())} source_only={sum(cut_extra.values())} output_samples={list(cut_missing.items())[:8]} source_samples={list(cut_extra.items())[:8]}")
for ci in (2,6):
 cut=np.asarray([f for f in comp[ci]["local_expected_output_boundary"] if canonical_face(f) not in updated_target_keys],np.int64).reshape(-1,3)
 cut_oriented=face_counter(cut)
 absent=[]
 for key,count in cut_oriented.items():
  opposite=key[:3]+(-key[3],)
  if unselected_oriented[opposite]<count:
   absent.append({"face_identity":list(key),"required_opposite_count":int(count),"available_opposite_count":int(unselected_oriented[opposite])})
 if absent: raise ValueError(f"component {ci}: local cut shell does not oppose untouched descendant tets: {absent[:8]}")
 comp[ci]["cut_faces_opposed_by_untouched_tets"]=int(sum(cut_oriented.values()))
 comp[ci]["cut_shell_face_count"]=int(sum(cut_oriented.values()))
newret=retained[keep[:len(retained)]]
if len(newret)!=int(np.sum(keep[:len(retained)])):raise ValueError("new retained row map shape mismatch")
replacement_tail=np.vstack([np.asarray(desc["replacement_tetrahedra"],np.int64),comp[2]["new_tets"],comp[6]["new_tets"]])
newtt=np.vstack([tdesc[keep],comp[2]["new_tets"],comp[6]["new_tets"]])
newmm=np.r_[mdesc[keep],np.full(len(comp[2]["new_tets"]),16,np.int32),np.full(len(comp[6]["new_tets"]),16,np.int32)]
newpp=np.asarray(point_list,np.float32)
if len(replacement_tail)!=len(newtt)-len(newret):raise ValueError("replacement tail is not aligned after comp2/6 integration")
if not np.array_equal(replacement_tail,newtt[len(newret):]):raise ValueError("replacement tetrahedron lineage tail differs from output cell ordering")
# Continue the existing 336 operation map in its 4602 input-cell row domain.
old_parent=np.asarray(desc["replacement_parent_cell_rows"],np.int64)
old_parent_offsets=np.asarray(desc["replacement_parent_offsets"],np.int64)
old_replacement=np.asarray(desc["replacement_tetrahedra"],np.int64)
old_tet_offsets=np.asarray(desc["replacement_tet_offsets"],np.int64)
if old_parent_offsets[0]!=0 or old_parent_offsets[-1]!=len(old_parent) or old_tet_offsets[0]!=0 or old_tet_offsets[-1]!=len(old_replacement):
 raise ValueError("336 operation CSR offsets malformed")
parent_add=np.r_[comp[2]["partial_input_rows"],comp[6]["partial_input_rows"]].astype(np.int64)
parent_offsets=np.r_[old_parent_offsets, len(old_parent)+len(comp[2]["partial_input_rows"]),len(old_parent)+len(parent_add)].astype(np.int64)
tet_offsets=np.r_[old_tet_offsets, len(old_replacement)+len(comp[2]["new_tets"]),len(old_replacement)+len(comp[2]["new_tets"])+len(comp[6]["new_tets"])].astype(np.int64)
parent_flat=np.r_[old_parent,parent_add]
replacement_flat=np.vstack([old_replacement,comp[2]["new_tets"],comp[6]["new_tets"]])
if len(parent_offsets)!=len(tet_offsets):raise ValueError("continued replacement operation offsets differ in count")
if not np.array_equal(newtt[len(newret):],replacement_flat):raise ValueError("candidate replacement operation map does not reproduce tail")

# Full oriented target: preserve 336 desired faces, with exactly these two D-derived edge splits.
desired=np.asarray(desc["desired_boundary_faces"],np.int64)
desired2,split6=split_global_faces(desired,*comp[6]["edge"],comp[6]["global_boundary_point_row"],comp[6]["target_faces"],"whole-target-comp6-final")
desired_updated,split2=split_global_faces(desired2,*comp[2]["edge"],comp[2]["global_boundary_point_row"],comp[2]["target_faces"],"whole-target-comp2-final")
if len(desired_updated)!=len(desired)+4:raise ValueError("two authenticated D edge splits must add exactly four target facets")
whole_boundary=m.boundary(newtt)
if face_counter(whole_boundary)!=face_counter(m.boundary(newtt)):
 raise ValueError("whole candidate boundary orientation is internally inconsistent")
old_whole_diff=counter_delta(m.boundary(tdesc),desired)
new_whole_diff=counter_delta(whole_boundary,desired_updated)
# Exact global boundary-flux delta must equal sum of local replacement deltas.
old_global_flux=exact_flux(pdesc,m.boundary(tdesc))
new_global_flux=exact_flux(newpp,whole_boundary)
local_delta=sum(int(comp[i]["local_volume_delta_det6"]) for i in (2,6))
if new_global_flux-old_global_flux!=local_delta:
 raise ValueError("whole exact oriented boundary flux delta differs from the two local exact cavity volume deltas")
# Exact new-versus-retained and inter-patch SAT; only authenticated replaced source rows are excluded.
tree=caps.TetBVH(pdesc,tdesc)
sat=caps.check_intersections(newpp,tdesc,replacement_flat,deleted_desc,[],tree,max_pairs=10_000_000)
if not sat.get("pass"):raise ValueError("combined exact retained/new Tet SAT failed: "+str(sat))
if not np.array_equal(newpp[:len(pdesc)],pdesc):raise ValueError("descendant points were modified")
if len(np.unique(np.sort(newtt,axis=1),axis=0))!=len(newtt):raise ValueError("duplicate tetrahedra in composed candidate")
if np.any(np.min(newtt,axis=1)==np.max(newtt,axis=1)):raise ValueError("degenerate tetrahedron in composed candidate")
if np.any(newmm!=16):raise ValueError("candidate contains material labels other than 16")
# Source-descendant exact sign comes from the independently replayed 336 base; exact-check the new rows directly above.
all_det,all_q=m.m.measurements(newpp,newtt)
before_det,before_q=m.m.measurements(pdesc,tdesc)
quality_before=int(np.sum(~m.m.passes(before_det,before_q)))
quality_after=int(np.sum(~m.m.passes(all_det,all_q)))
if int(np.sum(all_det<=0))!=0:raise ValueError("candidate F64 metric determinant census has nonpositive row")

outnpz=OUT/"fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz"
payload={k:v for k,v in desc.items()}
payload.update({
 "points_world_m":newpp,
 "tetrahedra":newtt,
 "material_indices":newmm,
 "boundary_faces":whole_boundary,
 "desired_boundary_faces":desired_updated,
 "retained_input_cell_rows":newret,
 "replacement_parent_offsets":parent_offsets,
 "replacement_parent_cell_rows":parent_flat,
 "replacement_tet_offsets":tet_offsets,
 "replacement_tetrahedra":replacement_flat,
 "fat614_boundary_transfer_component_ids":np.asarray([2,6],np.int64),
 "fat614_boundary_transfer_component_source_input_rows_comp2":comp[2]["source_rows"],
 "fat614_boundary_transfer_component_source_input_rows_comp6":comp[6]["source_rows"],
 "fat614_boundary_transfer_component_partial_input_rows_comp2":comp[2]["partial_input_rows"],
 "fat614_boundary_transfer_component_partial_input_rows_comp6":comp[6]["partial_input_rows"],
 "fat614_boundary_transfer_component_descendant_output_rows_comp2":comp[2]["descendant_output_rows"],
 "fat614_boundary_transfer_component_descendant_output_rows_comp6":comp[6]["descendant_output_rows"],
 "fat614_boundary_transfer_component_tetgen_node_global_rows_comp2":comp[2]["tetgen_node_global_rows"],
 "fat614_boundary_transfer_component_tetgen_node_global_rows_comp6":comp[6]["tetgen_node_global_rows"],
 "fat614_boundary_transfer_component_tetrahedra_global_rows_comp2":comp[2]["new_tets"],
 "fat614_boundary_transfer_component_tetrahedra_global_rows_comp6":comp[6]["new_tets"],
 "fat614_boundary_transfer_d_source_point_rows":np.asarray([120634,120633],np.int64),
 "fat614_boundary_transfer_d_global_volume_point_rows":np.asarray([comp[2]["global_boundary_point_row"],comp[6]["global_boundary_point_row"]],np.int64),
 "fat614_boundary_transfer_d_point_world_m_f32":np.asarray([newpp[comp[2]["global_boundary_point_row"]],newpp[comp[6]["global_boundary_point_row"]]],np.float32),
})
np.savez_compressed(outnpz,**payload)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
# Manifest input pins include the exact frozen .1 tables and owner receipts.
inputs={}
for name,(want,path) in EXPECTED.items():
 inputs[name]={"path":str(path),"expected_sha256":want,"actual_sha256":sha(path),"size_bytes":path.stat().st_size}
for ci in (2,6):
 for suffix in ("cavity.poly","cavity.1.node","cavity.1.ele","cavity.1.face","tetgen.stdout.log","tetgen.stderr.log"):
  path=comp[ci]["case"]/suffix
  inputs[f"component{ci}_{suffix}"]={"path":str(path),"sha256":sha(path),"size_bytes":path.stat().st_size}
inputs["descendant_336_independent_receipt"]={"path":str(DESC_INDEPENDENT),"sha256":sha(DESC_INDEPENDENT),"size_bytes":DESC_INDEPENDENT.stat().st_size}
report={
 "schema":"numi.fat614.two-rounded-boundary-points-volume-replay.v2",
 "status":"PASS_LOCAL_COMP2_COMP6_VOLUME_TOPOLOGY_PROPOSAL_PENDING_COMPLETE_BOUNDARY_MASS_AND_NATIVE_GATES",
 "physical_admission":False,"native_steps":0,"mass_mapping_status":"PENDING_SOURCE330_CONSERVATIVE_REBIND",
 "inputs":inputs,
 "descendant_composition":{"parent_candidate_sha256":EXPECTED["descendant_336"][0],"parent_cell_row_domain":"336 output rows map directly to 4602 input cell rows; comp2/comp6 old cells were located by c21b source row -> 4602 retained row -> 336 retained row; exact ordered tetrahedron tuples were checked at every link","inherited_independent_receipt_sha256":sha(DESC_INDEPENDENT),"inherited_complete_quality_failures":int(desc_receipt["final_quality"]["union_failures"])},
 "components":{},
 "whole_candidate":{
  "candidate_path":str(outnpz),"candidate_sha256":sha(outnpz),"candidate_size_bytes":outnpz.stat().st_size,
  "cells":int(len(newtt)),"points":int(len(newpp)),"boundary_faces":int(len(whole_boundary)),"desired_boundary_faces":int(len(desired_updated)),
  "input_oriented_boundary_mismatch":old_whole_diff,"output_oriented_boundary_mismatch":new_whole_diff,
  "complete_boundary_matches_updated_target":new_whole_diff["missing_oriented_records"]==0 and new_whole_diff["extra_oriented_records"]==0,
  "exact_global_boundary_flux_delta_det6":int(new_global_flux-old_global_flux),
  "sum_exact_local_cavity_volume_deltas_det6":int(local_delta),
  "exact_global_flux_delta_matches_local_cavity_delta":True,
  "source_and_output_oriented_internal_cut_sets_identical":True,
  "oriented_internal_cut_face_count":int(sum(old_cut_counter.values())),
  "full_candidate_f64_nonpositive_determinants":int(np.sum(all_det<=0)),
  "native_quality_failures_before":quality_before,"native_quality_failures_after":quality_after,"native_quality_failure_delta":quality_after-quality_before,
  "exact_retained_and_interpatch_sat":sat,
 },
 "gates":{"exact_F32_positive_new_cells":True,"local_oriented_shells_exact":True,"local_exact_volume_and_flux":True,"exact_retained_mesh_SAT":True,"complete_current_surface_D_audit":True,"mass_mapping":"NOT_RUN","native_execution":"NOT_RUN","physical_admission":False},
 "scope":"Versioned source-only volume replay for frozen TetGen comp2/comp6 topology using the exact rounded F32 points already present in authenticated D surface candidate 602036. Exact full D/O/I/walls/all185 surface audit is pinned, but this volume candidate still has unresolved global boundary mismatch and is not mass- or native-admitted."
}
for ci in (2,6):
 spec=comp[ci]
 report["components"][str(ci)]={
  "source_base_input_cell_rows":spec["source_rows"].tolist(),
  "parent_4602_input_cell_rows":spec["partial_input_rows"].tolist(),
  "current_336_output_cell_rows":spec["descendant_output_rows"].tolist(),
  "old_cell_count":int(len(spec["source_rows"])),"new_cell_count":int(len(spec["new_tets"])),
  "material":16,"input_shell_faces":int(len(spec["input_shell_faces_global"])),"updated_shell_faces":int(len(spec["local_expected_output_boundary"])),
  "target_edge_global_point_rows":list(map(int,spec["edge"])),"D_surface_point_row":int(spec["d_point_row"]),
  "D_volume_point_row":int(spec["global_boundary_point_row"]),"D_point_world_m_f32":newpp[spec["global_boundary_point_row"]].tolist(),
  "D_point_perpendicular_support_error_m":float(spec["d_point_support_residual_m"]),"D_point_edge_parameter":float(spec["d_point_segment_parameter"]),
  "tetgen_new_boundary_nodes":int(spec["new_boundary_node_count"]),"tetgen_interior_added_nodes":int(spec["new_interior_nodes"]),
  "new_tets_exact_F32_positive":int(spec["new_tets_exact_positive"]),"new_tets_native_quality_failures":int(spec["new_tets_native_failures"]),
  "old_exact_volume_det6":int(spec["old_local_volume_det6"]),"old_exact_boundary_flux_det6":int(spec["old_local_boundary_flux_det6"]),
  "new_exact_volume_det6":int(spec["new_local_volume_det6"]),"new_exact_boundary_flux_det6":int(spec["new_local_boundary_flux_det6"]),
  "exact_local_volume_delta_det6":int(spec["local_volume_delta_det6"]),"local_shell_self_intersections":int(spec["shell_self_intersections"]),
 }
rp=OUT/"report.json"
rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({
 "status":report["status"],"candidate_path":str(outnpz),"candidate_sha256":sha(outnpz),
 "report_path":str(rp),"report_sha256":sha(rp),"cells":len(newtt),"points":len(newpp),
 "component_old_new_cells":{k:[v["old_cell_count"],v["new_cell_count"]] for k,v in report["components"].items()},
 "oriented_boundary_before_after":[old_whole_diff,new_whole_diff],
 "exact_global_flux_delta_matches_local":True,"quality_failures":[quality_before,quality_after],
 "sat":sat
},indent=2))

