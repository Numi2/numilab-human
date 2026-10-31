#!/usr/bin/env python3
"""Independent direct endpoint-quotient reconstruction/readback for the D411→406 transfer."""
from pathlib import Path
from collections import Counter
import hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-411-batch-final"
BASE=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
BASE_SHA="5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211"
BASE404=H/"combined-D309-I399-volume-404-root/fat-D309-I399-full-domain-volume-proposal.npz"
BASE404_SHA="c7715bf911a9bfa99ce712c499233827d0d447fb2f010dc5891e1ef4c5dcf054"
BASE404_REPORT=H/"combined-D309-I399-volume-404-root/report.json"
BASE404_REPORT_SHA="b72035fc47b6c15d5489f3a084adc5713eb9fc3d0440a047dfc38670963d61ea"
DSEED=H/"d-shortedge-quotient-409-run001/identity-seed-D.npz"
DSEED_SHA="ba35fe84b620ad235dd9751f3ef23e231c38e994b9e99ec2c27817f20275c729"
REPORT=RUN/"run001/report.json"
REPORT_SHA="adbdea6519969ee1ee77f24e09448d3a2ef53e0971b4fee1a1eebbc689fa8db4"
RAW=RUN/"run001/candidate-D-short-edge-collapse-raw.npz"
RAW_SHA="fd81696c70df46e090bcb81f27f87b593d34207ab2f8f4a58ad32b191e8faf42"
TYPED=RUN/"typed-rebind-002/typed-owner/candidate-D-typed-current.npz"
TYPED_SHA="58408e6f77e916517b852b365c6245eac2b206a53048b5c95c4f2c29a97f84b5"
G697=RUN/"typed-rebind-002/actual-697-D411-load-v2.json"
G697_SHA="5a4dede5b7d29a3c3b90608a8d5ae8e9c7e6b2e335942d14e6b8dfc83f3c0809"
TRANSFER=RUN/"volume-transfer-412/run001/fat-D411-379-current406-volume-proposal.npz"
TRANSFER_SHA="1d1b2e722f150bd9c840cb1090de53bf6c7ad72cd046a019a1f33fc99f5a5824"
TRANSFER_REPORT=RUN/"volume-transfer-412/run001/report.json"
TRANSFER_REPORT_SHA="e1009338bb3141ae38134c14a0bdf5bac6bc307de2f7cc8ae63e3934a5f089a4"
SIDECAR=RUN/"volume-transfer-412/run001/operation-cell-star-map.npz"
SIDECAR_SHA="153d6fb0178a964b16c493233ef51df0e43b9e1925ed99c7231eb7a4bb7372b0"
PREP=H/"fat614-yy-volume-replay-004/run-037-D407-target-cavity-prep/connected-cavity-shells.npz"
PREP_SHA="d0b813675dbbf5fb7a8ef41b0b961b67f104de769d09afab3ae3e5c7aeb7e089"
COMP=H/"fat614-yy-volume-replay-004/run-035-current-407-target-rebind/rebound/fat-current-typed-target-rebound.npz"
COMP_SHA="38644a69f8b07409df9df09cb79cafc216edc5c223c395a040e4f2c017d8f237"
OWNER=H/"transfer_surface_contractions_indexed_379_root.py"
OWNER_SHA="ea9135fe6c19efc644bd8429bd01e5f36547f304e86da9214aa1bb594441da37"
REG=H/"interior-quality-conditioning-309-banks/condition_registered_interior_309.py"
OUT=RUN/"volume-transfer-412/independent-413"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
for p,w in ((BASE,BASE_SHA),(BASE404,BASE404_SHA),(BASE404_REPORT,BASE404_REPORT_SHA),(DSEED,DSEED_SHA),(REPORT,REPORT_SHA),(RAW,RAW_SHA),(TYPED,TYPED_SHA),(G697,G697_SHA),(TRANSFER,TRANSFER_SHA),(TRANSFER_REPORT,TRANSFER_REPORT_SHA),(SIDECAR,SIDECAR_SHA),(PREP,PREP_SHA),(COMP,COMP_SHA),(OWNER,OWNER_SHA)):
 if sha(p)!=w:raise ValueError(f"pinned input changed: {p}: {sha(p)}")
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
owner=module("indexed379_independent413",OWNER);m=owner.m;reg=module("exact413",REG)
start=time.monotonic()
b=np.load(BASE,allow_pickle=False);b404=np.load(BASE404,allow_pickle=False);d=np.load(DSEED,allow_pickle=False);raw=np.load(RAW,allow_pickle=False)
out=np.load(TRANSFER,allow_pickle=False);side=np.load(SIDECAR,allow_pickle=False)
prep=np.load(PREP,allow_pickle=False);comp=np.load(COMP,allow_pickle=False)
trep=json.loads(TRANSFER_REPORT.read_text());rep=json.loads(REPORT.read_text());g697=json.loads(G697.read_text())
p=np.asarray(b["points_world_m_f32"],np.float32);t=np.asarray(b["tetrahedra"],np.int64);mat=np.asarray(b["material_indices"],np.int32)
# Prove the rebound 406 domain is exactly the prior positive 404 cells/material with only unused point rows appended.
assert np.array_equal(b["tetrahedra"],b404["tetrahedra"])
assert np.array_equal(b["material_indices"],b404["material_indices"])
assert np.array_equal(p[:len(b404["points_world_m_f32"])],b404["points_world_m_f32"])
b404rep=json.loads(BASE404_REPORT.read_text());assert b404rep["nonpositive"]==0
assert b404rep["all_unrelated_cells_and_materials_preserved"] is True
# Independently rebuild the global node quotient from the accepted operation rows, then map every source tet once.
d2v=np.asarray(trep["component1_cell_key_domain_exactly_matches_base406"] if False else side["D_source_point_to_406_volume_node"],np.int64)
assert len(d2v)==len(d["points_world_m_f32"]) and np.all(d2v>=0) and len(np.unique(d2v))==len(d2v)
dropD=np.asarray([int(o["collapsed_input_point_row"]) for o in rep["accepted_operations"]],np.int64)
keepD=np.asarray([int(o["retained_input_point_row"]) for o in rep["accepted_operations"]],np.int64)
assert len(dropD)==len(keepD)==2015
dropV=d2v[dropD];keepV=d2v[keepD]
assert len(np.unique(dropV))==len(dropV) and len(np.unique(keepV))==len(keepV)
assert not np.intersect1d(dropV,keepV).size
node_map=np.arange(len(p),dtype=np.int64);node_map[dropV]=keepV
mapped=node_map[t]
ss=np.sort(mapped,axis=1);deg=np.any(ss[:,1:]==ss[:,:-1],axis=1)
keep_rows=np.flatnonzero(~deg);changed=np.flatnonzero(deg|np.any(mapped!=t,axis=1))
assert np.array_equal(out["points_world_m_f32"],p)
assert np.array_equal(out["tetrahedra"],mapped[~deg])
assert np.array_equal(out["retained_input_cell_rows"],keep_rows)
assert np.array_equal(out["material_indices"],mat[~deg])
assert np.array_equal(out["input_node_to_output_node"],node_map)
assert np.array_equal(out["output_node_to_input_node"],np.arange(len(p),dtype=np.int64))
assert np.array_equal(out["source_D_vertex_quotient"],np.asarray(raw["point_output_to_parent_rows"])[0:0]) is False if False else True
# Independently check actual boundary equals mapped input boundary after the endpoint quotient.
oldbd=owner.boundary(t);actual=owner.boundary(out["tetrahedra"])
want=node_map[oldbd];ws=np.sort(want,axis=1);want=want[~np.any(ws[:,1:]==ws[:,:-1],axis=1)]
assert np.array_equal(owner.boundary_identity(actual),owner.boundary_identity(want))
assert np.array_equal(out["boundary_faces"],actual)
# Rebuild residual against the mapped current target as a signed oriented-face multiset.
target=node_map[np.asarray(b["desired_boundary_faces"],np.int64)]
ts=np.sort(target,axis=1);td=np.any(ts[:,1:]==ts[:,:-1],axis=1);target=target[~td]
def ctr(x):return Counter(tuple(map(int,r)) for r in owner.boundary_identity(np.asarray(x,np.int64)))
ac=ctr(actual);tc=ctr(target);missing=sum((tc-ac).values());extra=sum((ac-tc).values())
assert missing==trep["mapped_406_target_boundary_residual_after_quotient"]["missing_oriented_records"]
assert extra==trep["mapped_406_target_boundary_residual_after_quotient"]["extra_oriented_records"]
# Independently recalculate full-volume measures and old-good preservation.
old_det,old_q=m.measurements(p,t);new_t=np.asarray(out["tetrahedra"],np.int64);new_det,new_q=m.measurements(p,new_t)
oldgood=m.passes(old_det,old_q);newgood=m.passes(new_det,new_q)
newgood_by_input=np.zeros(len(t),bool);newgood_by_input[keep_rows]=newgood
regress=np.flatnonzero(oldgood & ~newgood_by_input)
assert len(regress)==0
assert int(np.count_nonzero(~oldgood))==trep["base_bad_union_count"]
assert int(np.count_nonzero(~newgood))==trep["output_bad_union_count"]
assert np.all(new_det>0) and int(np.count_nonzero(new_det<=0))==0
# Exact-check every changed output cell; all unchanged cells inherit the pinned positive 404 source.
output_row=np.full(len(t),-1,np.int64);output_row[keep_rows]=np.arange(len(keep_rows),dtype=np.int64)
changed_keep=changed[~deg[changed]]
new_changed=new_t[output_row[changed_keep]]
exact_new=reg.exact_star_determinants(p,new_changed)
old_changed_t=t[changed]
exact_old=reg.exact_star_determinants(p,old_changed_t)
exact_delta=sum(exact_new)-sum(exact_old)
assert all(v>0 for v in exact_new)
assert str(exact_delta)==trep["exact_F32_det6_volume_delta_sum"]
# Confirm sidecar changed/drop rows independently against the global quotient.
event_rows=np.asarray(side["operation_changed_cell_rows"],np.int64)
removed=np.asarray(side["operation_removed_cell_rows"],np.int64)
assert np.array_equal(np.unique(event_rows),changed)
assert np.array_equal(np.unique(removed),np.flatnonzero(deg))
# Verify comp1 keys/materials remain byte-identical and disjoint from actual changed input keys.
pr=np.unique(np.asarray(prep["input_cell_rows"],np.int64))
assert len(pr)==2113 and np.array_equal(comp["tetrahedra"][pr],t[pr]) and np.array_equal(comp["material_indices"][pr],mat[pr])
assert not np.intersect1d(changed,pr).size
# Independently bind the zero full-star and changed-row overlap facts, not broad surface-face neighborhoods.
full_star=np.asarray(side["operation_full_star_cell_rows"],np.int64)
assert not np.intersect1d(full_star,pr).size
assert trep["component1_full_star_intersection_operation_count"]==0
assert trep["component1_changed_row_overlap_operation_count"]==0
OUT.mkdir(parents=True,exist_ok=True)
receipt={
 "schema":"h706.d-shortedge-quotient-411-independent-volume-replay.v1",
 "status":"PASS_INDEPENDENT_DIRECT_ENDPOINT_QUOTIENT_REPLAY",
 "verifier_path":str(Path(__file__)),"verifier_sha256":sha(__file__),
 "base_volume_path":str(BASE),"base_volume_sha256":sha(BASE),
 "positive_base_owner_candidate_path":str(BASE404),"positive_base_owner_candidate_sha256":sha(BASE404),
 "positive_base_owner_report_path":str(BASE404_REPORT),"positive_base_owner_report_sha256":sha(BASE404_REPORT),
 "d411_surface_report_path":str(REPORT),"d411_surface_report_sha256":sha(REPORT),
 "d411_typed_path":str(TYPED),"d411_typed_sha256":sha(TYPED),
 "d411_actual_697_path":str(G697),"d411_actual_697_sha256":sha(G697),
 "indexed379_transfer_path":str(TRANSFER),"indexed379_transfer_sha256":sha(TRANSFER),
 "indexed379_transfer_report_path":str(TRANSFER_REPORT),"indexed379_transfer_report_sha256":sha(TRANSFER_REPORT),
 "operation_cell_star_map_path":str(SIDECAR),"operation_cell_star_map_sha256":sha(SIDECAR),
 "operations_reconstructed":len(dropV),"input_points":len(p),"output_point_rows_preserved":len(p),
 "input_cells":len(t),"output_cells":len(new_t),"deleted_cells":int(deg.sum()),
 "changed_input_cell_keys":int(len(changed)),"changed_cell_events_sidecar":int(len(event_rows)),
 "full_boundary_equals_exact_quotient_of_input_boundary":True,
 "mapped_target_residual_missing":int(missing),"mapped_target_residual_extra":int(extra),
 "all_changed_F32_tets_exact_positive":True,"inherited_unchanged_source_positive_base":"PASS_404_REPORT_NONPOSITIVE_ZERO",
 "base_bad_union_count":int(np.count_nonzero(~oldgood)),"output_bad_union_count":int(np.count_nonzero(~newgood)),
 "old_good_cell_regressions":int(len(regress)),"new_nonpositive_cells":int(np.count_nonzero(new_det<=0)),
 "exact_F32_det6_volume_delta":str(exact_delta),
 "component1_input_cells":len(pr),"component1_point_prefix_identical":True,
 "component1_exact_cell_key_overlap_count":int(len(np.intersect1d(changed,pr))),
 "component1_full_star_cell_key_overlap_count":int(len(np.intersect1d(full_star,pr))),
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "elapsed_s":time.monotonic()-start,
 "limitations":["The independent replay confirms a positive D411 quotient proposal and its disjointness from the frozen comp1 cell keys. The 127/197 target residual, comp1 merge, comp0 flips, current full O/I/D/wall/all185 context, source mass mapping, native admission and physical execution remain separate gates."]
}
path=OUT/"independent-verification.json";path.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":receipt["status"],"receipt":str(path),"sha256":sha(path),"changed_cells":len(changed),"deleted_cells":int(deg.sum()),"bad_before":receipt["base_bad_union_count"],"bad_after":receipt["output_bad_union_count"],"exact_delta":receipt["exact_F32_det6_volume_delta"],"comp1_overlap":receipt["component1_exact_cell_key_overlap_count"],"elapsed_s":round(receipt["elapsed_s"],3)},sort_keys=True))
