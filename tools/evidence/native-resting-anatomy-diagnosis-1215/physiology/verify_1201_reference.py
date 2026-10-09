#!/usr/bin/env python3
"""Read-only, source-pinned summary of the 1201 40-second native trace."""
import csv, hashlib, importlib.util, json, math, pathlib, statistics, sys

ROOT = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201")
TRACE = ROOT / "native-run/resting-coupled.csv"
RESP = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-reference-respiration.json")
CIRC = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json")
OWNER = pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/native-initial-force-report-source-1187/matter/tools/resting_intervention_study.py")
DECL = ROOT / "run-declaration.json"
EXEC = ROOT / "execution.json"
META = ROOT / "native-run/run-metadata.json"
INV = ROOT / "native-run/invocation.json"
SKIN_SUMMARY = ROOT / "skin-audit-1201-terminal/summary.json"
SKIN_RESULT = ROOT / "skin-audit-1201-terminal/step-20000.result.json"
EXPECTED = {
    TRACE: "ab391d28e2caa9315d8cd0515c823debd44b8168c4f03c5ef118b49e33db343a",
    RESP: "70f4d235b988f9466b80e795ac0e793a82d6b13d03ebb01b84a313f0ad398e8a",
    CIRC: "43ff6b49daf1d42cf9e88d85d8577f70e112dfcc3756ec43954e544a3e4bc0dc",
    OWNER: "b06bb5a1cd14d86be362a5d0603f68546ba590bf038f9586c4cf89946fc39c1e",
    DECL: "c907c5ea7d1ed6027da60f27a0b07f498fd6499989b9051349688241b5656085",
    EXEC: "f5d27e8d0f832b11a159f370c60e54d0bff35b725203d15836c6cf37536cecea",
    META: "cefc252388be94acd09092c1f7d41385c2d038d053eea43ff9f162d0febf7258",
    INV: "ee2ca63cfe72122bc22f2ece2b3b62c738f33d252ce9d3007ef31d36cead0139",
    SKIN_SUMMARY: "867bb8565476f67291e5b7f76190bfe79b4f87892fa669e08b3d404765f11030",
    SKIN_RESULT: "9703c87b6e0fcb2e8a7fc0b8f25eaf8124659d3db0712f4b9a5f83a9cad1ede6",
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require_pins():
    for p, expected in EXPECTED.items():
        actual=sha(p)
        if actual != expected: raise SystemExit(f"PIN MISMATCH {p}: {actual} != {expected}")
def stats(rows, key):
    values=[float(row[key]) for row in rows]
    if not values or not all(math.isfinite(x) for x in values): raise ValueError(f"invalid {key}")
    return {"min":min(values),"mean":statistics.fmean(values),"max":max(values)}
def max_abs(rows,key): return max(abs(float(r[key])) for r in rows)
def events(rows,key):
    out=[]; prev=int(float(rows[0][key]))
    for row in rows[1:]:
        value=int(float(row[key]))
        if value>prev: out.append(row)
        prev=value
    return out

require_pins()
rows=list(csv.DictReader(TRACE.open(newline="")))
resp=json.loads(RESP.read_text()); circ=json.loads(CIRC.read_text())
decl=json.loads(DECL.read_text()); execution=json.loads(EXEC.read_text()); metadata=json.loads(META.read_text())
if len(rows)!=2500 or execution.get("returncode")!=0 or execution.get("changed_inputs")!={}: raise SystemExit("native run closure/prefix check failed")
if int(float(rows[-1]["step"])) != 20000: raise SystemExit("trace does not reach declared terminal accepted step")
if not all(float(row["root_assistance_n"])==0.0 for row in rows): raise SystemExit("assistance field nonzero")
owner_spec=importlib.util.spec_from_file_location("resting_owner", OWNER)
owner=importlib.util.module_from_spec(owner_spec);sys.modules[owner_spec.name]=owner;owner_spec.loader.exec_module(owner)
mechanics=owner.native_respiration_trace_consistency(TRACE, RESP, {"all_0_40":[0.0,40.001],"late_10_40":[10.0,40.001]})
resp_mech=mechanics["respiratory_mechanics"]
# Event-based quantities: avoid treating sampled point extrema as breath/cycle means.
breath_events=events(rows,"breaths")
breath_times=[float(r["time_s"]) for r in breath_events]
breath_periods=[b-a for a,b in zip(breath_times,breath_times[1:])]
tidal_events=[float(r["tidal_ml"]) for r in breath_events]
heart_events=events(rows,"complete_filling_ejection_cycles")
heart_times=[float(r["time_s"]) for r in heart_events]
heart_periods=[b-a for a,b in zip(heart_times,heart_times[1:])]
# Ejected-volume deltas over the full emitted interval are cumulative, not a beat-cycle CO estimate.
t0=float(rows[0]["time_s"]); t1=float(rows[-1]["time_s"])
late_start=next(i for i,r in enumerate(rows) if float(r["time_s"])>=10.0)
late=rows[late_start:]
period_s=6.0/7.0
external_counts={str(x):sum(1 for c in circ["compartments"] if c.get("external_pressure_pa")==x) for x in sorted(set(c.get("external_pressure_pa") for c in circ["compartments"]))}
report={
 "schema":"numi.human.1201-physiology-reference-review.v1",
 "status":"measured_model_trace_with_reference_context; not_clinical_or_anatomical_qualification",
 "execution":{
  "study":"native-flat-reference-40s-1201","declared_duration_s":decl["seconds"],"declared_dt_s":decl["dt"],"declared_step_count":decl["seconds"] and decl["seconds"] / decl["dt"],
  "trace_rows":len(rows),"trace_first_time_s":t0,"trace_terminal_time_s":t1,"accepted_terminal_step":int(float(rows[-1]["step"])),
  "native_exit_code":execution["returncode"],"owner_wrapper_wall_s":execution["wall_seconds"],"native_wall_s":metadata.get("wall_seconds"),"inputs_changed":execution["changed_inputs"],
  "body_mass_reference_kg":72.0,"contact_iterations":decl["contact_iterations"],"support_activation_cap":decl["postural_activation_cap"],"root_assistance_n_range":[min(float(x["root_assistance_n"]) for x in rows),max(float(x["root_assistance_n"]) for x in rows)],
  "binary_sha256":metadata["asset_sha256"][metadata["argv"][0]],
  "runtime_sha256":metadata["asset_sha256"]["/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib"],
  "respiration_metallib_sha256":metadata["asset_sha256"]["/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/matter/shaders/HumanRespiration.metallib"]
 },
 "pinned_inputs":{str(p):sha(p) for p in EXPECTED},
 "local_model":{
  "respiration_config":{
    "path":str(RESP),"sha256":sha(RESP),"reference_scope":resp.get("reference"),
    "resting_frequency_bpm":resp["brain_parameters"]["restingFrequencyBreathsPerMinute"],
    "resting_tidal_volume_ml":1000*resp["brain_parameters"]["restingTidalVolumeLitres"],
    "resting_minute_ventilation_l_min":resp["brain_parameters"]["restingMinuteVentilationLitresPerMinute"],
    "frc_ml":1e6*resp["frc_m3"],"dead_space_ml":1e6*resp["dead_space_m3"],
    "oxygen_consumption_stpd_ml_min":60e6*resp["oxygen_consumption_stpd_m3_per_s"],
    "carbon_dioxide_production_stpd_ml_min":60e6*resp["carbon_dioxide_production_stpd_m3_per_s"],
    "dry_atmospheric_pressure_pa":resp["dry_atmospheric_pressure_pa"],
    "pulmonary_equilibration_fraction":resp["pulmonary_equilibration_fraction"],
    "gas_scope":resp["parameter_scope"]["gas"],"lung_scope":resp["parameter_scope"]["lung"]
  },
  "circulation_config":{
    "path":str(CIRC),"sha256":sha(CIRC),"schema":circ["schema"],"model_id":circ["model_id"],"law":circ["law"],"qualification":circ["qualification"],
    "source_graph_sha256":circ["source_graph_sha256"],"compartments":len(circ["compartments"]),"connections":len(circ["connections"]),
    "species_count":len(circ.get("species",[])),"exchange_count":len(circ.get("exchanges",[])),
    "cardiac_chamber_period_rational_seconds":"6/7","fixed_rate_bpm":60/period_s,"external_pressure_pa_counts":external_counts
  }
 },
 "observed_output":{
  "arterial_pressure_and_saturation":{
    "PaO2_mmhg":stats(rows,"PaO2_mmhg"),"PaCO2_mmhg":stats(rows,"PaCO2_mmhg"),"SaO2_fraction":stats(rows,"SaO2"),"SaO2_percent":{"min":100*stats(rows,"SaO2")["min"],"mean":100*stats(rows,"SaO2")["mean"],"max":100*stats(rows,"SaO2")["max"]}
  },
  "breathing":{
    "completed_breath_events":len(breath_events),"breath_event_times_s":breath_times,"inter_breath_periods_s":breath_periods,
    "inter_breath_rate_bpm_mean_of_observed_intervals":60/statistics.fmean(breath_periods),
    "tidal_volume_at_breath_counter_increments_ml":tidal_events,
    "tidal_volume_event_mean_ml":statistics.fmean(tidal_events),
    "estimated_minute_ventilation_from_event_means_l_min":statistics.fmean(tidal_events)/1000*60/statistics.fmean(breath_periods),
    "pleural_pressure_pa":stats(rows,"pleural_pa"),"pleural_pressure_cmH2O":{"min":min(float(r["pleural_pa"]) for r in rows)/98.0665,"mean":statistics.fmean(float(r["pleural_pa"]) for r in rows)/98.0665,"max":max(float(r["pleural_pa"]) for r in rows)/98.0665},
    "alveolar_pressure_pa":stats(rows,"alveolar_pa"),"alveolar_pressure_cmH2O":{"min":min(float(r["alveolar_pa"]) for r in rows)/98.0665,"max":max(float(r["alveolar_pa"]) for r in rows)/98.0665},
    "lung_volume_ml":stats(rows,"lung_volume_ml"),"airflow_ml_s":stats(rows,"airflow_ml_s")
  },
  "circulation":{
    "completed_filling_ejection_counter_terminal":int(float(rows[-1]["complete_filling_ejection_cycles"])),
    "observed_cycle_interval_count":len(heart_periods),"observed_mean_cycle_period_s":statistics.fmean(heart_periods),"observed_rate_bpm":60/statistics.fmean(heart_periods),
    "configured_rate_bpm":60/period_s,
    "full_horizon_aortic_ejected_ml":float(rows[-1]["aortic_ejected_ml"]),"full_horizon_pulmonary_ejected_ml":float(rows[-1]["pulmonary_ejected_ml"]),
    "full_horizon_aortic_mean_flow_l_min":float(rows[-1]["aortic_ejected_ml"])*0.06/t1,
    "full_horizon_pulmonary_mean_flow_l_min":float(rows[-1]["pulmonary_ejected_ml"])*0.06/t1,
    "late_10_40_aortic_mean_flow_l_min":(float(late[-1]["aortic_ejected_ml"])-float(late[0]["aortic_ejected_ml"]))*0.06/(float(late[-1]["time_s"])-float(late[0]["time_s"])),
    "late_10_40_pulmonary_mean_flow_l_min":(float(late[-1]["pulmonary_ejected_ml"])-float(late[0]["pulmonary_ejected_ml"]))*0.06/(float(late[-1]["time_s"])-float(late[0]["time_s"])),
    "last_lv_stroke_ml":stats(rows,"last_lv_stroke_ml"),"blood_ml":stats(rows,"blood_ml")
  }
 },
 "mechanical_and_numerical_consistency":{
   "owner_check":resp_mech,
   "oxygen_balance_error_stpd_ml_max_abs":max_abs(rows,"oxygen_balance_error_stpd_ml"),
   "co2_balance_error_stpd_ml_max_abs":max_abs(rows,"co2_balance_error_stpd_ml"),
   "blood_residual_minus_physical_ml_max_abs":max_abs(rows,"blood_residual_minus_physical_ml"),
   "blood_endpoint_minus_physical_ml_max_abs":max_abs(rows,"blood_endpoint_minus_physical_ml"),
   "respiratory_volume_balance_ml_max_abs":max_abs(rows,"respiratory_volume_balance_ml"),
   "interpretation":"Owner-checked identities and small residuals establish internal arithmetic/closure for these exported states only; they do not establish anatomical validity, independent physiological accuracy, or clinical validity."
 },
 "separate_anatomy_gate":{
   "terminal_skin_audit_summary_path":str(SKIN_SUMMARY),"terminal_skin_audit_summary_sha256":sha(SKIN_SUMMARY),
   "terminal_skin_audit_result_path":str(SKIN_RESULT),"terminal_skin_audit_result_sha256":sha(SKIN_RESULT),
   "crossing_pair_count":3112,"ocular_crossing_pair_count":0,"target_coverage_complete":True,"self_scan_complete":True,
   "interpretation":"The 40-second output has a retained terminal skin-to-target geometry rejection count of 3,112. This is a separate failed anatomical gate; the numerical physiology summary must not be presented as an integrated resting-body qualification."
 },
 "source_backed_interpretation":{
  "claims":[
   "At-rest adult ventilation references are descriptive anchors (about 500 mL tidal volume and 12 breaths/min in the cited review), not universal individual limits; 1201 event-derived values are about 0.53 L and 11.1/min over six post-first-breath intervals.",
   "Common ABG reference intervals are pH 7.35-7.45, PaO2 75-100 mmHg, PaCO2 35-45 mmHg, HCO3 22-26 mEq/L, SaO2 95-100%; the comparison is contextual only. This model has no pH/HCO3 output and its configured dry atmospheric pressure is 95.06 kPa; age, altitude/site and clinical sampling context are not calibrated. PaO2 is mostly at or slightly above the common upper bound, so it should be reported as a model output, not called clinically normal.",
   "AHA’s broad adult resting heart-rate range is 60-100/min. The model’s observed ~70/min is consistent with its configured fixed 6/7-second cycle; this is not autonomic or ECG validation.",
   "A textbook NCBI reference gives roughly 5 L/min cardiac output for a 70-kg adult at rest. The trace’s cumulative whole-run mean is ~4.95 L/min, but it is a 40-second model average with startup/phase effects, not a patient measurement.",
   "Quiet spontaneous inspiration changes intrathoracic pressure and can affect venous return and LV ejection; the respiration trace’s pleural-pressure swings are not dynamically propagated into the circulation compartments’ fixed external-pressure constants in this source variant.",
   "The config explicitly uses a single perfusion-limited pulmonary equilibration, Hill O2 relation and fixed-pH linear CO2 content law, while disclaiming Haldane/acid-base modeling; it does not represent regional V/Q heterogeneity or diffusion limitation."
  ],
  "references":[
   {"title":"Arterial Blood Gas Analysis: Fundamentals, Interpretation, and Clinical Utility","url":"https://www.ncbi.nlm.nih.gov/books/NBK536919/","supports":"common ABG intervals and need for pH/PaCO2/HCO3 context"},
   {"title":"The physics of human breathing: flow, timing, volume, and pressure parameters for normal, on-demand, and ventilator respiration","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC8672270/","supports":"nominal rest tidal volume/rate/dead space with individual variation"},
   {"title":"All About Heart Rate","url":"https://www.heart.org/en/health-topics/high-blood-pressure/the-facts-about-high-blood-pressure/all-about-heart-rate-pulse","supports":"broad adult resting heart-rate range"},
   {"title":"Introduction: Control of Cardiac Output","url":"https://www.ncbi.nlm.nih.gov/books/NBK54473/","supports":"approximately 5 L/min at rest in a 70-kg adult"},
   {"title":"Cardiopulmonary Interactions: Physiologic Basis and Clinical Applications","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC5822394/","supports":"intrathoracic-pressure effects on venous return and LV ejection"},
   {"title":"A mechanistic physicochemical model of carbon dioxide transport in blood","url":"https://journals.physiology.org/doi/10.1152/japplphysiol.00318.2016","supports":"bicarbonate, carbamino and Haldane mechanisms in CO2 carriage"},
   {"title":"Ventilation/Perfusion Relationships and Gas Exchange: Measurement Approaches","url":"https://pmc.ncbi.nlm.nih.gov/articles/PMC8274320/","supports":"regional V/Q matching as a major determinant of gas-exchange efficiency"}
  ]
 },
 "qualification_boundaries":[
   "No clinical interpretation or patient-specific normality claim: age, sex-specific predicted values, pH/bicarbonate, hemoglobin, temperature, calibrated ambient/site context and reference-subject provenance are not established.",
   "No 5-minute endurance claim: this is a 40-second run.",
   "No complete cardio-respiratory feedback claim: perfusion drives respiratory gas transport, but pleural pressure is not coupled back to dynamic cardiovascular external pressure/preload; heart rate is configured fixed-rate.",
   "No regional ventilation/perfusion distribution, Haldane effect, or acid-base compensation model.",
   "No integrated resting-body qualification: terminal 3112 skin-target crossings remain a separate failed anatomy gate."
 ]
}
OUT=pathlib.Path(__file__).with_name("report.json")
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(OUT)
print(hashlib.sha256(OUT.read_bytes()).hexdigest())
