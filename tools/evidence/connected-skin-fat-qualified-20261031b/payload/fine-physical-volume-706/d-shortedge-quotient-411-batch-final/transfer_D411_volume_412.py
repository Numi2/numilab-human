#!/usr/bin/env python3
"""Apply frozen D411 operations to current 406 Fat with indexed379 exact owner; no admission."""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-411-batch-final"
BASE=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
BASE_SHA="5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211"
DSEED=H/"d-shortedge-quotient-409-run001/identity-seed-D.npz"
DSEED_SHA="ba35fe84b620ad235dd9751f3ef23e231c38e994b9e99ec2c27817f20275c729"
D411_TYPED=RUN/"typed-rebind-002/typed-owner/candidate-D-typed-current.npz"
D411_TYPED_SHA="58408e6f77e916517b852b365c6245eac2b206a53048b5c95c4f2c29a97f84b5"
REPORT=RUN/"run001/report.json"
REPORT_SHA="adbdea6519969ee1ee77f24e09448d3a2ef53e0971b4fee1a1eebbc689fa8db4"
RAW=RUN/"run001/candidate-D-short-edge-collapse-raw.npz"
RAW_SHA="fd81696c70df46e090bcb81f27f87b593d34207ab2f8f4a58ad32b191e8faf42"
OWNER_PATH=H/"transfer_surface_contractions_indexed_379_root.py"
OWNER_SHA="ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37"
REG_PATH=H/"interior-quality-conditioning-309-banks/condition_registered_interior_309.py"
PREP=H/"fat614-yy-volume-replay-004/run-037-D407-target-cavity-prep/connected-cavity-shells.npz"
PREP_SHA="d0b813675dbbf5fb7a8ef41b0b961b67f104de769d09afab3ae3e5c7aeb7e089"
OUT=RUN/"volume-transfer-412/run001"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
for p,w in ((BASE,BASE_SHA),(DSEED,DSEED_SHA),(D411_TYPED,D411_TYPED_SHA),(REPORT,REPORT_SHA),(RAW,RAW_SHA),(OWNER_PATH,OWNER_SHA),(PREP,PREP_SHA)):
 if sha(p)!=w:raise ValueError(f"pinned input changed: {p}: {sha(p)}")
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
owner=module("indexed379_D411",OWNER_PATH);m=owner.m;reg=module("exactvolume_delta_412",REG_PATH)
base=np.load(BASE,allow_pickle=False);dseed=np.load(DSEED,allow_pickle=False);raw=np.load(RAW,allow_pickle=False)
rep=json.loads(REPORT.read_text());ops=rep["accepted_operations"]
assert len(ops)==rep["accepted_collapse_count"]==2015 and all(o["role"]=="D" for o in ops)
assert np.array_equal(raw["faces"],np.load(D411_TYPED,allow_pickle=False)["faces"])
assert np.array_equal(raw["points_world_m_f32"],np.load(D411_TYPED,allow_pickle=False)["points_world_m_f32"])
p=np.asarray(base["points_world_m_f32"],np.float32);t=np.asarray(base["tetrahedra"],np.int64).copy();mat=np.asarray(base["material_indices"],np.int32)
assert len(mat)==len(t) and np.all(mat==16) and np.isfinite(p).all()
sp=np.asarray(dseed["points_world_m_f32"],np.float32);sf=np.asarray(dseed["faces"],np.int64)
assert np.array_equal(sp,np.load(D411_TYPED,allow_pickle=False)["points_world_m_f32"])
assert np.array_equal(sf,dseed["faces"])
key_to_vol={owner.key(x):i for i,x in enumerate(p)}
if len(key_to_vol)!=len(p):raise ValueError("ambiguous Float32 volume coordinate keys")
d_to_vol=np.asarray([key_to_vol.get(owner.key(x),-1) for x in sp],np.int64)
if np.any(d_to_vol<0) or len(np.unique(d_to_vol))!=len(d_to_vol):raise ValueError("D source point→406 volume coordinate map is incomplete/ambiguous")
bd=owner.boundary(t);look={owner.key(x):i for i,x in enumerate(p)}
assert len(look)==len(p)
flat=t.ravel();order=np.argsort(flat,kind="stable");offset=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
v={"path":BASE,"p":p,"t":t,"mat":mat,"base_boundary":bd,"boundary_keys":{tuple(sorted(map(int,f))) for f in bd},"look":look,"active":np.ones(len(t),bool),"map":np.arange(len(p),dtype=np.int64),"node_incidence_order":order,"node_incidence_offsets":offset,"node_incidence_added":{}}
base_det,base_q=m.measurements(p,t)
if np.any(base_det<=0):raise ValueError("406 input has nonpositive measured cells")
base_bad=int(np.count_nonzero(~m.passes(base_det,base_q)))
prep=np.load(PREP,allow_pickle=False);protected=np.unique(np.asarray(prep["input_cell_rows"],np.int64))
assert len(protected)==2113 and np.all((protected>=0)&(protected<len(t)))
# e112 comp1 domain is a point-appended derivative with same 406 tetra/material row domain.
compbase=H/"fat614-yy-volume-replay-004/run-035-current-407-target-rebind/rebound/fat-current-typed-target-rebound.npz"
comp=np.load(compbase,allow_pickle=False)
assert np.array_equal(comp["tetrahedra"],base["tetrahedra"]) and np.array_equal(comp["material_indices"],base["material_indices"])
assert np.array_equal(comp["points_world_m_f32"][:len(p)],p)
assert np.array_equal(comp["tetrahedra"][protected],base["tetrahedra"][protected])
assert np.array_equal(comp["material_indices"][protected],base["material_indices"][protected])
protected_keys={ (int(mat[i]),*map(int,np.sort(t[i]))) for i in protected }
sd=np.arange(len(sp),dtype=np.int64)
full_offsets=[0];full_rows=[];changed_offsets=[0];changed_rows=[];removed_offsets=[0];removed_rows=[]
old_changed=[];new_changed=[];deleted_changed=[];records=[];failure=[]
exact_delta=0;conflicts=[]
start=time.monotonic()
for oi,op in enumerate(ops):
 trial,why=owner.try_op(v,sp,sf,op)
 if trial is None:
  failure.append({"operation":oi,"patch_id":op.get("patch_id"),"reason":why})
  continue
 rows=np.asarray(trial["rows"],np.int64);old=np.asarray(trial["old"],np.int64);new=np.asarray(trial["new"],np.int64);deleted=np.asarray(trial["deleted"],bool)
 changed=deleted|np.any(new!=old,axis=1);cr=rows[changed];rr=rows[deleted]
 if np.intersect1d(rows,protected,assume_unique=False).size:
  protected_star=np.intersect1d(rows,protected,assume_unique=False)
 else: protected_star=np.empty(0,np.int64)
 if len(cr):
  for row0,oldt,newt,del0 in zip(cr,old[changed],new[changed],deleted[changed]):
   key0=(int(mat[row0]),*map(int,np.sort(oldt)))
   if key0 in protected_keys: conflicts.append({"operation":oi,"patch_id":op.get("patch_id"),"cell_row":int(row0),"old_tet_key":list(key0[1:]),"deleted":bool(del0)})
 a,b=map(int,(trial["kept_volume_node"],trial["retired_volume_node"]))
 # Exact integer-lattice local volume change, over exactly modified old/new rows.
 delta=int(sum(reg.exact_star_determinants(p,old))-sum(reg.exact_star_determinants(p,new[~deleted])))
 exact_delta+=-delta
 v["t"][rows]=new;v["active"][rows[deleted]]=False;v["map"][b]=a
 v["node_incidence_added"].setdefault(a,set()).update(map(int,rows[~deleted]))
 sd[int(op["collapsed_input_point_row"])]=int(op["retained_input_point_row"])
 full_rows.extend(map(int,rows));full_offsets.append(len(full_rows))
 changed_rows.extend(map(int,cr));changed_offsets.append(len(changed_rows))
 removed_rows.extend(map(int,rr));removed_offsets.append(len(removed_rows))
 old_changed.extend(old[changed].tolist());new_changed.extend(new[changed].tolist());deleted_changed.extend(deleted[changed].tolist())
 records.append({"operation":oi,"patch_id":op["patch_id"],"source_edge_rows":op["edge_rows"],"collapsed_D_point_row":int(op["collapsed_input_point_row"]),"retained_D_point_row":int(op["retained_input_point_row"]),"collapsed_volume_node":b,"retained_volume_node":a,"full_star_cell_rows":rows.tolist(),"changed_cell_rows":cr.tolist(),"removed_cell_rows":rr.tolist(),"comp1_full_star_intersection_count":int(len(protected_star)),"comp1_changed_cell_overlap_count":int(len(np.intersect1d(cr,protected))),"exact_F32_det6_delta":str(-delta),"bad_before":trial["bad_before"],"bad_after":trial["bad_after"]})
 # Apply through exact node/tet state before next operation.
assert len(records)+len(failure)==len(ops)
tt=v["t"][v["active"]];active_rows=np.flatnonzero(v["active"]);newbd=owner.boundary(tt)
expected_oldbd=v["map"][bd];es=np.sort(expected_oldbd,axis=1);expected_oldbd=expected_oldbd[~np.any(es[:,1:]==es[:,:-1],axis=1)]
oldbd_match=np.array_equal(owner.boundary_identity(newbd),owner.boundary_identity(expected_oldbd))
# Map the 406 target chain through the accepted node quotient; this is a target comparison, not a new D rebinding.
target=np.asarray(base["desired_boundary_faces"],np.int64);mapped=v["map"][target];ms=np.sort(mapped,axis=1);target_degenerate=np.any(ms[:,1:]==ms[:,:-1],axis=1);mapped_target=mapped[~target_degenerate]
def counter(f):
 return Counter(tuple(map(int,x)) for x in owner.boundary_identity(np.asarray(f,np.int64)))
actualc=counter(newbd);targetc=counter(mapped_target);missing=sum((targetc-actualc).values());extra=sum((actualc-targetc).values())
det,q=m.measurements(p,tt)
if np.any(det<=0):raise ValueError("output has nonpositive measured cells")
after_bad=int(np.count_nonzero(~m.passes(det,q)))
# Exact all-output delta is exactly the sum of the per-operation local determinant deltas.
unchanged=np.ones(len(t),bool);all_full=np.asarray(full_rows,np.int64);unchanged[np.unique(all_full)]=False
assert np.array_equal(v["t"][unchanged],t[unchanged]) and np.all(v["active"][unchanged])
if len(failure)==0 and not oldbd_match:raise ValueError("full D411 volume quotient boundary differs from mapped 406 actual boundary")
OUT.mkdir(parents=True,exist_ok=False)
side=OUT/"operation-cell-star-map.npz"
np.savez_compressed(side,
 operation_full_star_offsets=np.asarray(full_offsets,np.int64),operation_full_star_cell_rows=np.asarray(full_rows,np.int64),
 operation_changed_cell_offsets=np.asarray(changed_offsets,np.int64),operation_changed_cell_rows=np.asarray(changed_rows,np.int64),
 operation_removed_cell_offsets=np.asarray(removed_offsets,np.int64),operation_removed_cell_rows=np.asarray(removed_rows,np.int64),
 operation_old_changed_tetrahedra=np.asarray(old_changed,np.int64).reshape(-1,4),
 operation_new_changed_tetrahedra=np.asarray(new_changed,np.int64).reshape(-1,4),
 operation_changed_deleted=np.asarray(deleted_changed,bool),comp1_protected_input_cell_rows=protected,
 D_source_point_to_406_volume_node=np.asarray(d_to_vol,np.int64),D411_source_vertex_quotient=sd,
 output_406_input_cell_rows=active_rows,output_406_input_node_quotient=v["map"])
candidate=None
if len(failure)==0:
 candidate=OUT/"fat-D411-379-current406-volume-proposal.npz"
 payload={k:base[k] for k in base.files}
 payload.update({"points_world_m":p,"points_world_m_f32":p,"tetrahedra":tt,"material_indices":mat[v["active"]],"boundary_faces":newbd,"retained_input_cell_rows":active_rows,"input_node_to_output_node":v["map"],"output_node_to_input_node":np.arange(len(p),dtype=np.int64),"source_D_vertex_quotient":sd,"source_D411_operation_indices":np.asarray([x["operation"] for x in records],np.int64),"source_D411_cell_star_map_path":np.asarray([str(side)]),"source_D411_cell_star_map_sha256":np.asarray([sha(side)])})
 # Update current D/I/wall point references through the quotient; geometry and existing desired arrays remain separately pinned in report.
 for k in ("fat_current_target_rebind_D_point_to_volume_rows","fat_current_target_rebind_I_point_to_volume_rows","fat_current_target_rebind_wall_faces_volume_rows"):
  if k in payload:payload[k]=v["map"][np.asarray(payload[k],np.int64)]
 np.savez_compressed(candidate,**payload)
 # Read back core arrays and source-map identity.
 outz=np.load(candidate,allow_pickle=False)
 assert np.array_equal(outz["tetrahedra"],tt) and np.array_equal(outz["points_world_m_f32"],p) and np.array_equal(outz["material_indices"],mat[v["active"]])
assert sum(len(x["changed_cell_rows"]) for x in records)==len(changed_rows)
report={
 "schema":"h706.d-shortedge-quotient-411-indexed379-volume-transfer.v1",
 "status":"PASS_379_VOLUME_QUOTIENT_PROPOSAL" if len(failure)==0 and oldbd_match else "INCOMPLETE_379_VOLUME_TRANSFER_DIAGNOSTIC",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "base_volume_path":str(BASE),"base_volume_sha256":sha(BASE),
 "base_D_typed_source_path":str(DSEED),"base_D_typed_source_sha256":sha(DSEED),
 "surface_D411_typed_path":str(D411_TYPED),"surface_D411_typed_sha256":sha(D411_TYPED),
 "surface_D411_actual_697_receipt_path":str(RUN/"typed-rebind-002/actual-697-D411-load-v2.json"),
 "surface_D411_actual_697_receipt_sha256":sha(RUN/"typed-rebind-002/actual-697-D411-load-v2.json"),
 "surface_D411_raw_report_path":str(REPORT),"surface_D411_raw_report_sha256":sha(REPORT),
 "surface_D411_raw_candidate_path":str(RAW),"surface_D411_raw_candidate_sha256":sha(RAW),
 "indexed379_owner_path":str(OWNER_PATH),"indexed379_owner_sha256":sha(OWNER_PATH),
 "component1_input_path":str(compbase),"component1_input_sha256":sha(compbase),
 "component1_shell_path":str(PREP),"component1_shell_sha256":sha(PREP),
 "component1_cell_key_domain_exactly_matches_base406":True,
 "operation_count_expected":len(ops),"operation_count_transferred":len(records),"operation_failures":failure,
 "input_cells":len(t),"output_cells":len(tt),"input_points":len(p),"output_point_rows_retained":len(p),
 "all_input_point_coordinates_unchanged":True,"all_materials_are_16":bool(np.all(mat==16)),
 "base_bad_union_count":base_bad,"output_bad_union_count":after_bad,"new_nonpositive_cells":0,
 "minimum_output_normalized_det":float(q.min()),"minimum_output_volume_m3":float(det.min()/6),
 "actual_boundary_equals_mapped_406_actual_boundary":bool(oldbd_match),
 "mapped_406_target_boundary_residual_after_quotient":{"missing_oriented_records":int(missing),"extra_oriented_records":int(extra),"degenerate_target_records_removed":int(target_degenerate.sum())},
 "component1_selected_cells":int(len(protected)),"changed_cell_key_overlap_component1":conflicts,
 "component1_full_star_intersection_operation_count":int(sum(x["comp1_full_star_intersection_count"]>0 for x in records)),
 "component1_changed_row_overlap_operation_count":int(sum(x["comp1_changed_cell_overlap_count"]>0 for x in records)),
 "affected_cell_row_count":len(full_rows),"changed_cell_row_event_count":len(changed_rows),"removed_cell_row_event_count":len(removed_rows),
 "exact_F32_det6_volume_delta_sum":str(exact_delta),
 "operation_cell_star_map_path":str(side),"operation_cell_star_map_sha256":sha(side),
 "candidate_path":str(candidate) if candidate else None,"candidate_sha256":sha(candidate) if candidate else None,
 "operations":records,"elapsed_s":time.monotonic()-start,
 "limitations":["This is an indexed379 geometric volume quotient proposal only. Full current O/I/D/wall/all185 context, D407 comp1 and comp0 surface repairs, mass mapping, native execution and physical admission remain separate gates."]
}
rp=OUT/"report.json";rp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("operations","operation_failures","changed_cell_key_overlap_component1")},sort_keys=True))
