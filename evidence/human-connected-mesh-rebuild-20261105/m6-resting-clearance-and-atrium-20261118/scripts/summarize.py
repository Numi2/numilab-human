"""Assemble the matched benchmark and exact-geometry comparisons on the pinned host."""
import csv
import json
from pathlib import Path
import re
from statistics import mean
import sys

def read(path):
    return json.loads(path.read_text())

def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def geometry(audit):
    return {
        "time_s": audit["accepted_time_s"],
        "skin_self_pairs": audit["self_intersections"]["count"],
        "target_pairs": sum(audit["crossing_pairs_by_semantic"].values()),
        "target_pairs_by_semantic": audit["crossing_pairs_by_semantic"],
        "target_banks": len(audit["surfaces"]),
        "pack_sha256": audit["accepted_pack_sha256"],
        "limit": audit["limit"],
    }

def pair_sets(audit):
    return {(r["semantic"], r["stable_id"], r["primitive"]): set(map(tuple,r["triangle_pairs"]))
            for r in audit["surfaces"]}

def motion(run):
    log = (run/"stdout.log").read_text()
    trace = rows(run/"native/resting-coupled.csv")
    execution = read(run/"execution.json")
    end = trace[-1]
    info = {"exit": execution["exit_code"], "pressure_setting": read(
        run/"launch.json")["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"],
        "last_tabulated_state": {key: float(end[key]) for key in
            ("step","time_s","breaths","tidal_ml","PaO2_mmhg","PaCO2_mmhg","last_lv_stroke_ml",
             "complete_filling_ejection_cycles","right_atrium_ml","left_atrium_ml")},
        "wall_seconds_launch_to_exit": execution["seconds_process"],
        "max_abs_sampled_blood_error_ml": max(abs(float(r["blood_error_ml"])) for r in trace),
        "max_abs_root_assistance_n": max(abs(float(r["root_assistance_n"])) for r in trace),
        "max_abs_root_assistance_nm": max(abs(float(r["root_assistance_nm"])) for r in trace),
        "native_renders": log.count("\nresting_render_profile step=")}
    for line in log.splitlines():
        if line.startswith(("resting_integrated_throughput ", "resting_integrated_body=completed ")):
            name = "sustained" if line.startswith("resting_integrated_throughput") else "body_with_terminal"
            info[name+"_timing"] = dict(re.findall(r"\b(\w+)=([^ ]+)", line))
    support = rows(run/"native/resting-com-momentum-diagnostic.csv")
    late = [r for r in support if float(r["time_s"]) > float(end["time_s"])-2]
    info["late_mean_bed_load_n"] = mean(float(r["normal_force_last_physical_step_n"]) for r in late)
    info["late_weight_n"] = float(late[-1]["total_body_mass_kg"])*9.81
    info["surface_online_maxima"] = {k:max(int(r[k]) for r in rows(run/"native/resting-surface-audit.csv"))
        for k in ("mesh_zero_area_triangles","mesh_nonfinite_area_triangles","vertices_below_1mm","nonfinite_skin_vertices")}
    if "body_with_terminal_timing" in info:
        info["launch_to_exit_rtf"] = float(end["time_s"])/execution["seconds_process"]
    return info

def main(q):
    previous = q.parent/"coupled-crossing-repair-001"
    control = q.parent/"coupled-thumb-reference-001"
    base = read(previous/"thumb-exact.json")
    early = read(q/"rebound-four-exact.json")
    late = read(q/"rebound-original-16-exact.json")
    a,b = pair_sets(base),pair_sets(early)
    assert a.keys() == b.keys() and len(a) == 878
    new = [(k,p) for k in b for p in sorted(b[k]-a[k])]
    removed = [(k,p) for k in a for p in sorted(a[k]-b[k])]
    earliest = control/"thumb-4s-capture/native/resting-local-bed-sections.csv"
    same_bed = earliest.read_bytes() == (q/"rebound-enabled-four/native/resting-local-bed-sections.csv").read_bytes()
    assert same_bed
    assert set(map(tuple,early["self_intersections"]["triangle_pairs"])) < set(map(tuple,base["self_intersections"]["triangle_pairs"]))
    run = q/"rebound-pressure-16"
    failure = read(run/"common-failure.json")
    latch = failure["first_failure_latch"]
    assert latch["common_status"][0] == 5
    assert int(latch["brain_accepted_metadata"][1]) == latch["step"]*2000
    failure_log = (run/"stdout.log").read_text()
    t_start, t_end = [float(re.search(
        rf"^human_execution_stage=authoritative_segment_end wall_elapsed_ms=([\d.]+) stage_step={step}$",
        failure_log, re.MULTILINE)[1]) for step in (8,2776)]
    calibration = rows(q/"physiology-dynamic-8000.csv")
    native = rows(run/"native/resting-coupled.csv")
    lookup = {round(float(r["time_s"]),6):r for r in calibration}
    fields = ("pleural_pa","right_atrium_ml","left_atrium_ml","rv_ml","lv_ml")
    matched = [(r,lookup[round(float(r["time_s"]),6)]) for r in native
               if round(float(r["time_s"]),6) in lookup]
    assert len(matched)==347 and all(a[k]==b[k] for a,b in matched for k in fields)
    result = {"motion": {name:motion(q/name) for name in
        ("rebound-enabled-four","rebound-pressure-16","rebound-original-16")},
        "geometry":{"prior_4s":geometry(base),"rebind_4s":geometry(early),"rebind_16s":geometry(late),
            "new_target_witnesses_at_4s": new, "removed_target_witnesses_at_4s": removed,
            "bed_section_measurements_byte_identical_at_4s":same_bed,
            "largest_late_fold_clusters": [r["witness_pairs"] for r in read(q/"skin-late-witness-regions.json")["regions"][:2]]},
        "pressure_failure":{
            "rejected_step_zero_based":latch["step"],"last_accepted_seconds":latch["step"]*.002,
            "last_accepted_brain_time_us":latch["brain_accepted_metadata"][1],
            "relative_volume_error":latch["common_diagnostics"][0],
            "relative_volume_error_limit":.00002,
            "candidate_right_atrium_ml":latch["chamber_targets_m3"][0]*1e6,
            "candidate_right_atrium_q":latch["common_first"][0],
            "atlas_box_96_right_atrium_upper":failure["matched_box_upper"][0],
            "native_prefix_timing": {"start_step":8,"end_step":2776,
                "wall_seconds":(t_end-t_start)/1000,"rtf":(2776-8)*.002*1000/(t_end-t_start)},
            "matching_calibration_samples":len(matched),"matching_calibration_fields":fields,
            "calibration_samples":len(calibration),"calibration_last_seconds":float(calibration[-1]["time_s"])},
        "scope":{"geometry_admission_change":False,"default_scene_change":False,
                 "zero_crossings_verified":False,"joint_acceptance":False}}
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
