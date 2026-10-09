#!/usr/bin/env python3
"""Read-only consistency audit for sealed native contoured-bed attempt 1198/002."""
import csv, hashlib, json, math, os, pathlib, struct, sys
from collections import Counter, defaultdict

RUN = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199")
OUT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-contact-trace-audit-1199")
NATIVE = RUN / "native-run"
SUPPORT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/myosim-fullbody-distributed-rigid-digit-support.nhcnt")
DT = 0.002
STEPS = 10000
SEGMENT = 8

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()

def readjson(path):
    with open(path) as f: return json.load(f)

def readcsv(name):
    with open(NATIVE/name,newline="") as f:
        rd=csv.DictReader(f); return rd.fieldnames,list(rd)

def number(x):
    return float(x)

def near(a,b,tol): return abs(a-b) <= tol

errors=[]
execution=readjson(RUN/"execution.json")
decl=readjson(RUN/"run-declaration.json")
meta=readjson(NATIVE/"run-metadata.json")
invocation=readjson(NATIVE/"invocation.json")
if execution.get("returncode") != 0: errors.append("execution returncode not zero")
if execution.get("declaration_sha256") != sha(RUN/"run-declaration.json"): errors.append("execution declaration hash mismatch")
expected_nonruntime_changes={
    "/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/source-region-coverage-1199.json":{
        "before":"42b6ef6d348b68127b44d910eb2028816499dec92e129282a94f0c0277168b07",
        "after":"31f64284bafe6328226eb1aec866912898675e6807e2c4e8275fa182554649c5"},
    "/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/validate_source_regions.py":{
        "before":"198d576c97ad1221c2cd835e2e6ecbb9f388e7c1283524c94502f6c0ebdc8c1c",
        "after":"e07949985b1e9aca6ae20498e6f1ba799b7167d6b09b3a2fb3d457e431bc8aed"}}
if execution.get("changed_inputs") != expected_nonruntime_changes:
    errors.append("execution changed inputs differ from the two explicitly recovered nonruntime audit-only files")
for original, pin in expected_nonruntime_changes.items():
    path=pathlib.Path(original)
    if not path.is_file() or sha(path)!=pin["after"]:
        errors.append("current post-run nonruntime audit file does not match its recorded after hash: "+original)
    suffix="-v1.json" if original.endswith(".json") else "-v1.py"
    recovered=path.with_name(path.stem+suffix)
    if not recovered.is_file() or sha(recovered)!=pin["before"]:
        errors.append("recovered launch-time nonruntime audit file does not match its before hash: "+str(recovered))
if meta.get("exit_code") != 0: errors.append("native run-metadata exit_code not zero")
if meta.get("source_files_changed_during_run") != []: errors.append("runtime source changed during run")
if not meta.get("loaded_metal_runtime",{}).get("verified"): errors.append("loaded Metal runtime not verified")
if meta.get("loaded_metal_runtime",{}).get("expected_sha256") != meta.get("asset_sha256",{}).get(meta.get("loaded_metal_runtime",{}).get("expected_path")):
    errors.append("loaded runtime hash is not the invocation-pinned asset hash")

coupled_h,coupled=readcsv("resting-coupled.csv")
support_h,support=readcsv("resting-com-support-impulses.csv")
com_h,com=readcsv("resting-com-momentum-diagnostic.csv")
surface_h,surface=readcsv("resting-surface-audit.csv")
if len(coupled)!=STEPS//SEGMENT or len(com)!=STEPS//SEGMENT:
    errors.append("coupled or COM row count does not equal accepted 8-step sample count")
if len(com)!=0 and (com[0]["delta_com_valid"]!="0" or any(x["delta_com_valid"]!="1" for x in com[1:])):
    errors.append("COM momentum delta validity does not identify the initial sample followed by 1249 accepted deltas")
if len(support)!=(STEPS//SEGMENT)*32:
    errors.append("per-contact row count is not 32 contacts per accepted segment")
expected_steps=list(range(SEGMENT,STEPS+1,SEGMENT))
steps_c=[int(x["step"]) for x in coupled]
steps_m=[int(x["accepted_step"]) for x in com]
steps_s=sorted({int(x["accepted_step"]) for x in support})
if steps_c!=expected_steps or steps_m!=expected_steps or steps_s!=expected_steps:
    errors.append("accepted trace cadence is not every 8 steps through terminal 10000")
if any(int(x["sample_start_step"])!=int(x["accepted_step"])-SEGMENT or int(x["segment_steps"])!=SEGMENT for x in support):
    errors.append("per-contact impulse segment labeling mismatch")
if any(int(x["sample_start_step"])!=int(x["accepted_step"])-SEGMENT or int(x["segment_steps"])!=SEGMENT for x in com):
    errors.append("COM impulse segment labeling mismatch")

# Verify the exact bound NHCNT1 contact order and friction coefficients used by the runtime.
raw=SUPPORT.read_bytes()
if raw[:8] != b"NHCNT1\0\0": errors.append("support payload is not NHCNT1")
abi,body_count,source_count,reserved=struct.unpack_from("<4I",raw,8)
if abi!=1 or reserved!=0 or len(raw)!=84+48*source_count: errors.append("malformed NHCNT1 extent/ABI")
source_sha=raw[24:56].hex()
friction_by_contact={}
for i in range(source_count):
    off=84+48*i
    body,geom=struct.unpack_from("<II",raw,off)
    mu=struct.unpack_from("<f",raw,off+32)[0]
    friction_by_contact[i]=(body,geom,mu)
if len(friction_by_contact)!=32: errors.append("NHCNT contact count differs from expected 32")

# Exact contact row coverage, world-basis reconstruction, friction cone, and COM sum identity.
rows_by_step=defaultdict(list)
max_unit_err=max_orth_err=max_world_component_err=0.0
max_com_sum_err=max_normal_aggregate_err=max_force_aggregate_err=0.0
max_friction_util=0.0
max_normal_impulse=0.0
min_normal_impulse=float("inf")
negative_normals=0
friction_violations=0
contact_identity_violations=0
basis_vs_global_z_max=0.0
max_contacts_by_step=0
world_sum_by_step=defaultdict(lambda:[0.0,0.0,0.0])
normal_sum_by_step=defaultdict(float)
for row in support:
    step=int(row["accepted_step"]); idx=int(row["contact_index"])
    if idx not in friction_by_contact:
        contact_identity_violations+=1; continue
    body,geom,mu=friction_by_contact[idx]
    if body!=int(row["body_index"]) or geom!=int(row["source_geometry_index"]):
        contact_identity_violations+=1
    rows_by_step[step].append(row)
    n=[number(row["normal_"+a]) for a in "xyz"]
    t0=[number(row["tangent0_"+a]) for a in "xyz"]
    t1=[number(row["tangent1_"+a]) for a in "xyz"]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    norm=lambda a:math.sqrt(dot(a,a))
    max_unit_err=max(max_unit_err,max(abs(norm(v)-1.0) for v in (n,t0,t1)))
    max_orth_err=max(max_orth_err,abs(dot(n,t0)),abs(dot(n,t1)),abs(dot(t0,t1)))
    basis_vs_global_z_max=max(basis_vs_global_z_max,math.sqrt(n[0]*n[0]+n[1]*n[1]))
    ln,l0,l1=(number(row[k]) for k in ("normal_impulse_ns","tangent0_impulse_ns","tangent1_impulse_ns"))
    if not all(math.isfinite(v) for v in (ln,l0,l1,mu,*n,*t0,*t1)):
        if "nonfinite contact value in trace" not in errors: errors.append("nonfinite contact value in trace")
    if mu < 0 and "negative friction coefficient in trace" not in errors: errors.append("negative friction coefficient in trace")
    if ln < -1e-8: negative_normals+=1
    min_normal_impulse=min(min_normal_impulse,ln); max_normal_impulse=max(max_normal_impulse,ln)
    tangent=math.hypot(l0,l1)
    util=tangent/(mu*ln) if mu*ln>1e-14 else (0.0 if tangent<=1e-14 else float("inf"))
    max_friction_util=max(max_friction_util,util)
    if util>1.00001: friction_violations+=1
    reconstructed=[ln*n[j]+l0*t0[j]+l1*t1[j] for j in range(3)]
    recorded=[number(row["impulse_world_"+a+"_ns"]) for a in "xyz"]
    max_world_component_err=max(max_world_component_err,max(abs(x-y) for x,y in zip(reconstructed,recorded)))
    normal_sum_by_step[step]+=ln
    for j,a in enumerate("xyz"): world_sum_by_step[step][j]+=recorded[j]

com_by_step={int(x["accepted_step"]):x for x in com}
coupled_by_step={int(x["step"]):x for x in coupled}
for step in expected_steps:
    if len(rows_by_step.get(step,[]))!=32 or sorted(int(x["contact_index"]) for x in rows_by_step[step])!=list(range(32)):
        errors.append(f"step {step} does not contain one row for each contact 0..31")
    cm=com_by_step.get(step); cp=coupled_by_step.get(step)
    if cm is None or cp is None: continue
    max_contacts_by_step=max(max_contacts_by_step,int(cm["active_contact_count"]))
    vec=[number(cm["world_contact_impulse_"+a+"_ns"]) for a in "xyz"]
    max_com_sum_err=max(max_com_sum_err,max(abs(x-y) for x,y in zip(vec,world_sum_by_step[step])))
    max_normal_aggregate_err=max(max_normal_aggregate_err,abs(normal_sum_by_step[step]-number(cm["normal_impulse_last_physical_step_ns"])),abs(normal_sum_by_step[step]-number(cp["normal_impulse_ns"])))
    max_force_aggregate_err=max(max_force_aggregate_err,abs(number(cm["normal_force_last_physical_step_n"])-normal_sum_by_step[step]/DT))
if contact_identity_violations: errors.append("contact index/body/source-geometry identity differs from pinned NHCNT order")
if negative_normals: errors.append("negative normal contact impulses exceed numerical tolerance")
if friction_violations: errors.append("Coulomb impulse disk exceeded by more than CSV/F32 tolerance")
if max_unit_err>2e-5 or max_orth_err>2e-5: errors.append("contact basis is not unit/orthogonal within 2e-5")
if max_world_component_err>2e-9: errors.append("per-contact component-to-world reconstruction mismatch")
if max_com_sum_err>2e-8: errors.append("COM world contact impulse does not equal sum of per-contact vectors")
if max_normal_aggregate_err>2e-6: errors.append("normal impulse totals differ beyond float output tolerance")
if max_force_aggregate_err>2e-3: errors.append("normal force does not equal last-step impulse/dt within tolerance")

# Accepted body/physiology clock, assistance, geometry flags, and sampled gaps.
time_errors=[]
for row in coupled:
    st=int(row["step"]); t=number(row["time_s"])
    time_errors.append(abs(t-st*DT))
    for key in row:
        try: val=float(row[key])
        except (ValueError,TypeError): continue
        if not math.isfinite(val): errors.append(f"nonfinite value in coupled field {key} at step {st}")
if max(time_errors,default=0)>2e-6: errors.append("coupled physiology clock diverges from accepted body step clock")
root_n=max(abs(number(x["root_assistance_n"])) for x in coupled)
root_nm=max(abs(number(x["root_assistance_nm"])) for x in coupled)
if root_n!=0.0 or root_nm!=0.0: errors.append("root assistance is nonzero")
if int(coupled[-1]["step"])!=STEPS or not near(number(coupled[-1]["time_s"]),20.0,2e-6): errors.append("terminal physical clock is not 10000 steps / 20 seconds")
invalid_surface=0
for row in surface:
    for key in ("vertices_below_1mm","nonfinite_skin_vertices","mesh_zero_area_triangles","mesh_nonfinite_area_triangles"):
        if int(row[key])!=0: invalid_surface+=1
if invalid_surface: errors.append("surface audit reports invalid skin/mesh rows")
geom_status=Counter(x["functional_geometry_status"] for x in surface)
solver_status=Counter(x["common_coordinate_solver_status"] for x in surface)
geometry_modes=Counter(x["geometry_mode"] for x in surface)
common_coordinate_fields=("common_coordinate_RA","common_coordinate_RV","common_coordinate_LA","common_coordinate_LV","common_coordinate_RA_material","common_coordinate_ventricular_material","common_coordinate_LA_material")
common_residuals=[number(x["common_coordinate_normalized_residual"]) for x in surface]
common_iterations=[int(x["common_coordinate_solver_iterations"]) for x in surface]
common_domain_boxes=[int(x["common_coordinate_domain_box"]) for x in surface]
legacy_sentinel_fields=("q_ra","q_rv","q_la","q_lv","ventricular_closure_mm")
legacy_nan_sentinel_ok=all(all(math.isnan(number(x[k])) for k in legacy_sentinel_fields) for x in surface)
common_coordinates_finite=all(all(math.isfinite(number(x[k])) for k in common_coordinate_fields) for x in surface)
common_audit_ok=all(int(x["common_coordinate_solver_status"])==0 and int(x["ventricular_material_status"])==0 and int(x["functional_geometry_status"])==0 and int(x["ventricular_closure_mm_applicable"])==0 and math.isfinite(number(x["common_coordinate_normalized_residual"])) and 0<=number(x["common_coordinate_normalized_residual"])<=2e-5 and number(x["max_functional_volume_relative_error"])<=2e-4 and all(number(x[k])>0 for k in ("ra_target_ml","rv_target_ml","la_target_ml","lv_target_ml","lung_target_ml")) for x in surface)
if not legacy_nan_sentinel_ok: errors.append("legacy cardiac coordinates/closure are not the declared common-geometry NaN sentinel")
if not common_coordinates_finite or not common_audit_ok: errors.append("surface cardiac/common-geometry status or residual is invalid")

# Hash the invocation-declared inputs and retained trace artifacts.
asset_rechecks={}
for path,expected in meta.get("asset_sha256",{}).items():
    try:
        actual=sha(path)
        asset_rechecks[path]={"expected_sha256":expected,"actual_sha256":actual,"matches":actual==expected}
        if actual!=expected: errors.append("asset hash mismatch: "+path)
    except Exception as exc:
        asset_rechecks[path]={"expected_sha256":expected,"error":repr(exc),"matches":False}
        errors.append("cannot rehash asset: "+path)

file_pins={}
for path in [RUN/"execution.json",RUN/"run-declaration.json",RUN/"runtime-source-at-launch.json",RUN/"runtime-source-at-launch.patch",RUN/"run.py",NATIVE/"run-metadata.json",NATIVE/"invocation.json",NATIVE/"native.log",RUN/"owner-stdout.log",NATIVE/"resting-coupled.csv",NATIVE/"resting-com-support-impulses.csv",NATIVE/"resting-com-momentum-diagnostic.csv",NATIVE/"resting-surface-audit.csv",SUPPORT]:
    file_pins[str(path)]={"sha256":sha(path),"size_bytes":path.stat().st_size}
source_files=[RUN/"runtime-source/apps/numilab_human_myosim_visual_probe.mm",RUN/"runtime-source/apps/NumiHumanRestingSupportGeometry.hpp",RUN/"runtime-source/src/metal/NumiHumanStandSolve.metalinc",RUN/"runtime-source/src/metal/NumiHumanStand.metal",RUN/"runtime-source/src/metal/MetalArticulatedOperator.mm",pathlib.Path("/Users/n/numi-human-contoured-bed-source-1196/matter/tools/resting_intervention_study.py")]
for path in source_files:
    role="source validator for common-coordinate NaN sentinels" if path.name=="resting_intervention_study.py" else "contact generation/normal-basis/solver audit source snapshot"
    file_pins[str(path)]={"sha256":sha(path),"size_bytes":path.stat().st_size,"role":role}
source_pin=readjson(RUN/"runtime-source-at-launch.json")
file_pins[str(RUN/"runtime-source-at-launch.json")]={"sha256":sha(RUN/"runtime-source-at-launch.json"),"size_bytes":(RUN/"runtime-source-at-launch.json").stat().st_size,"role":"changed-source list, base revision, and retained patch pin"}
file_pins[str(RUN/"runtime-source-at-launch.patch")]={"sha256":sha(RUN/"runtime-source-at-launch.patch"),"size_bytes":(RUN/"runtime-source-at-launch.patch").stat().st_size,"role":"source patch reconstructed from pinned base"}
base_repo=pathlib.Path("/Users/n/numi-human-contoured-bed-source-1196")
base_header=base_repo/"include/metalrobo/NumiHumanSupport.hpp"
file_pins[str(base_header)]={"sha256":sha(base_header),"git_head":os.popen("cd "+str(base_repo)+" && git rev-parse HEAD").read().strip(),"role":"support record decoder/header at pinned runtime base; no launch delta recorded"}

report={
 "schema":"numi.native-contact-trace-consistency.v1",
 "scope":"Read-only audit of the sealed 20 s native contoured-bed smoke run; accepted segment endpoint contact traces, not per-step impulse closure or long-horizon qualification. Rows every 8 steps expose the returned last-step contact impulse basis/impulse at each accepted endpoint; the seven interior physical steps are not individually recorded in these CSVs.",
 "run_path":str(RUN),"run_returncode":execution.get("returncode"),"run_wall_seconds":execution.get("wall_seconds"),"internal_simulated_seconds":meta.get("wall_seconds"),"runtime_exit_code":meta.get("exit_code"),
 "run_declaration_sha256":sha(RUN/"run-declaration.json"),"expected_declaration_sha256":execution.get("declaration_sha256"),"loaded_runtime":meta.get("loaded_metal_runtime"),"source_files_changed_during_run":meta.get("source_files_changed_during_run"),"changed_inputs":execution.get("changed_inputs"),
 "nonruntime_input_change_resolution":{
   path:{"before_sha256":pin["before"],"after_sha256":pin["after"],
         "reconstructed_launch_copy":str(pathlib.Path(path).with_name(pathlib.Path(path).stem+("-v1.json" if path.endswith(".json") else "-v1.py"))),
         "recovery_status":"exact_before_hash_verified; retained_original_post_run_file_hash_verified"}
   for path,pin in expected_nonruntime_changes.items()},
 "step_clock":{"dt_s":DT,"segment_steps":SEGMENT,"accepted_sample_count":len(coupled),"first_accepted_step":steps_c[0] if steps_c else None,"terminal_step":steps_c[-1] if steps_c else None,"terminal_time_s":number(coupled[-1]["time_s"]) if coupled else None,"max_abs_time_minus_step_dt_s":max(time_errors,default=0.0),"physiology_breath_count_terminal":int(coupled[-1]["breaths"]),"complete_filling_ejection_cycles_terminal":int(coupled[-1]["complete_filling_ejection_cycles"])},
 "contact_rows":{"count":len(support),"contact_count_per_endpoint":32,"unique_identity_count":len({(int(x["body_index"]),int(x["source_geometry_index"])) for x in support}),"all_samples_have_contact_indices_0_through_31":all(len(rows_by_step.get(st,[]))==32 and sorted(int(x["contact_index"]) for x in rows_by_step[st])==list(range(32)) for st in expected_steps),"pinned_payload":{"magic":raw[:8].decode("ascii","replace"),"abi":abi,"body_count":body_count,"source_record_count":source_count,"expanded_contact_count":len(friction_by_contact),"source_rigid_sha256":source_sha,"friction_min":min(x[2] for x in friction_by_contact.values()),"friction_max":max(x[2] for x in friction_by_contact.values())},"basis_max_unit_error":max_unit_err,"basis_max_pairwise_dot_abs":max_orth_err,"max_normal_xy_magnitude_relative_to_static_plus_z":basis_vs_global_z_max,"normal_impulse_min_ns":min_normal_impulse,"normal_impulse_max_ns":max_normal_impulse,"negative_normal_impulse_rows_over_1e-8":negative_normals,"maximum_coulomb_disk_utilization":max_friction_util,"friction_cone_violations_over_1e-5":friction_violations,"max_per_contact_component_to_world_vector_abs_error_ns":max_world_component_err,"max_com_aggregate_world_impulse_vs_per_contact_sum_abs_error_ns":max_com_sum_err,"max_contact_normal_sum_vs_coupled_and_com_normal_impulse_abs_error_ns":max_normal_aggregate_err,"max_com_normal_force_vs_final_segment_impulse_over_dt_abs_error_n":max_force_aggregate_err,"active_contacts_max":max_contacts_by_step},
 "assistance":{"max_abs_root_assistance_n":root_n,"max_abs_root_assistance_nm":root_nm},
 "sampled_support":{"common_coordinate_solver_status_all_zero":all(int(x["common_coordinate_solver_status"])==0 for x in surface),"common_coordinate_normalized_residual_max":max(common_residuals),"common_coordinate_iterations_range":[min(common_iterations),max(common_iterations)],"common_coordinate_domain_box_range":[min(common_domain_boxes),max(common_domain_boxes)],"common_coordinate_state_fields_finite":common_coordinates_finite,"legacy_chamber_q_and_closure_are_common-geometry_NaN_sentinels":legacy_nan_sentinel_ok,"max_functional_volume_relative_error":max(number(x["max_functional_volume_relative_error"]) for x in surface),"min_coupled_contact_gap_m":min(number(x["min_contact_gap_m"]) for x in coupled),"max_coupled_penetration_m":max(number(x["peak_penetration_m"]) for x in coupled),"min_skin_bed_gap_m":min(number(x["min_skin_bed_gap_m"]) for x in surface),"surface_audit_rows":len(surface),"surface_audit_status_counts":dict(geom_status),"common_coordinate_status_counts":dict(solver_status),"geometry_mode_counts":dict(geometry_modes),"max_vertices_below_1mm":max(int(x["vertices_below_1mm"]) for x in surface),"max_nonfinite_skin_vertices":max(int(x["nonfinite_skin_vertices"]) for x in surface),"max_zero_area_triangles":max(int(x["mesh_zero_area_triangles"]) for x in surface),"max_nonfinite_area_triangles":max(int(x["mesh_nonfinite_area_triangles"]) for x in surface)},
 "trace_checks":{"accepted_sampling_semantics":"1250 accepted 8-step endpoint samples; per-contact CSV rows contain one row for each of 32 contacts at each endpoint; no claim is made about summed impulse closure across all eight substeps","coupled_rows":len(coupled),"com_rows":len(com),"surface_audit_rows":len(surface),"all_finite_coupled_fields":not any("nonfinite value in coupled" in e for e in errors),"delta_com_valid_count":sum(x["delta_com_valid"]=="1" for x in com),"first_com_sample_has_no_prior_delta":com[0]["delta_com_valid"]=="0","momentum_residual_status_counts":dict(Counter(x["momentum_residual_status"] for x in com)),"momentum_residual_status_is_not_a_per_step_integrated_closure":all(x["momentum_residual_status"]=="not_per_step_or_initial_sample" for x in com),"contact_impulse_vector_available_all_samples":all(x["contact_impulse_vector_available"]=="1" for x in com)},
 "asset_rechecks":asset_rechecks,"input_and_source_pins":file_pins,"errors":errors,"audit_status":"pass" if not errors else "fail"
}
OUT.mkdir(parents=True,exist_ok=True)
report_path=OUT/"contact-trace-audit.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"audit_status":report["audit_status"],"errors":errors,"report_path":str(report_path),"report_sha256":sha(report_path),"script_sha256":sha(__file__),"counts":{"coupled":len(coupled),"contact":len(support),"com":len(com),"surface":len(surface)},"metrics":report["contact_rows"]},indent=2))
if errors: sys.exit(1)
