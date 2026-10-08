#!/usr/bin/env python3
"""Offline audit of native 909 accepted q/v, source limits, COM and contacts."""
import csv, hashlib, json, math, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

E=Path("/Users/n/numi-human-resting-evidence-20261005")
RUN=E/"native-composed-resting-cycle-909"
OUT=E/"native-composed-resting-cycle-mechanics-review-910"
MANIFEST=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json")
EXPECTED_MANIFEST_SHA="844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7"
N=10000
DT_REQUESTED=.002
G=9.81

def req(x,m):
    if not x: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def rows(p):
    with open(p,"r",encoding="utf-8",newline="") as f:
        yield from csv.DictReader(f)
def vec(s):
    return [] if not s else [float(x) for x in s.split(";")]
def n3(x): return math.sqrt(math.fsum(v*v for v in x))
def stat(xs):
    xs=sorted(float(x) for x in xs)
    if not xs: return None
    def pct(p):
        i=(len(xs)-1)*p/100; a=int(i); b=math.ceil(i)
        return xs[a] if a==b else xs[a]*(b-i)+xs[b]*(i-a)
    return {"count":len(xs),"mean":statistics.fmean(xs),"min":xs[0],"p50":pct(50),
            "p95":pct(95),"p99":pct(99),"max":xs[-1]}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report_path=OUT/"report.json"
    req(not report_path.exists(),"report already exists")
    meta_p=RUN/"run-metadata.json"; inv_p=RUN/"invocation.json"
    meta=json.loads(meta_p.read_text()); inv=json.loads(inv_p.read_text())
    req(meta["exit_code"]==0,"run did not exit 0")
    req(meta["source_files_changed_during_run"]==[],"source changed during run")
    req(meta["loaded_metal_runtime"]["verified"] is True,"loaded runtime image not verified")
    argv=meta["argv"]; req(inv["argv"]==argv,"invocation differs from completion receipt")
    def arg(flag): return argv[argv.index(flag)+1]
    req(arg("--muscle-step-count")=="10000","native step count mismatch")
    req(abs(float(arg("--muscle-step-seconds"))-DT_REQUESTED)<1e-12,"requested dt mismatch")
    req(argv[4]==str(RUN),"native output path mismatch")
    asset_hashes={}
    for p,h in meta["asset_sha256"].items():
        req(Path(p).is_file() and sha(p)==h,"runtime asset hash mismatch: "+p)
        asset_hashes[p]=h
    map_sha=sha(MANIFEST); req(map_sha==EXPECTED_MANIFEST_SHA,"source joint map manifest pin mismatch")
    mdoc=json.loads(MANIFEST.read_text()); core=mdoc["core_tree"]; jm=core["source_joint_map"]
    req(core["nq"]==129 and core["nv"]==128 and len(jm)==122,"source map q/v dimensions mismatch")
    req(sorted(x["core_q_index"] for x in jm)==list(range(7,129)),"source q indices not bijective")
    req(sorted(x["core_v_index"] for x in jm)==list(range(6,128)),"source v indices not bijective")
    for x in jm:
        lo,hi=x["source_range"]
        req(x["source_limited"] is True and math.isfinite(lo) and math.isfinite(hi) and lo<=hi,
            "invalid source joint limit record")

    js={}
    for x in jm:
        qi=x["core_q_index"]
        js[qi]={"source_joint_id":x["source_joint_id"],"source_name":x["source_name"],
          "core_joint_index":x["core_joint_index"],"core_q_index":qi,"core_v_index":x["core_v_index"],
          "source_type":x["source_type"],"source_limited":x["source_limited"],
          "source_range":x["source_range"],"core_limit_status":x["core_limit_status"],
          "q_min":math.inf,"q_max":-math.inf,"abs_v_max":0.0,"abs_v_max_step":None,
          "below_range_samples":0,"above_range_samples":0,"largest_excursion":0.0,
          "largest_excursion_step":None,"final_q":None}
    qfile=RUN/"resting-com-q-integration.csv"
    qcount=0; next_step=1; prevq=prevv=None; qmatches=vmatches=0; qfails=[]; vfails=[]
    qerr=(0.0,None); rootv=(0.0,None); rootw=(0.0,None); dts=[]; tfirst=tlast=None
    for r in rows(qfile):
        qcount+=1; s=int(r["accepted_step"]); req(s==next_step,"accepted-step gap at "+str(next_step)); next_step+=1
        t=float(r["time_s"])
        if tfirst is None: tfirst=t
        tlast=t; dts.append(float(r["dt_s"]))
        req(int(r["configuration_count"])==129 and int(r["velocity_count"])==128,"observer q/v dimension mismatch")
        parsed={}
        for k in ("q_before_f32_semicolon","v_before_f32_semicolon","q_preprojection_f32_semicolon",
                  "v_preprojection_f32_semicolon","q_accepted_f32_semicolon","v_accepted_f32_semicolon"):
            a=vec(r[k]); want=129 if k.startswith("q_") else 128
            req(len(a)==want and all(math.isfinite(y) for y in a),"nonfinite/wrong-length "+k+" step "+str(s))
            parsed[k]=a
        qa=parsed["q_accepted_f32_semicolon"]; va=parsed["v_accepted_f32_semicolon"]
        e=abs(n3(qa[3:7])-1.0)
        if e>qerr[0]: qerr=(e,s)
        if n3(va[:3])>rootv[0]: rootv=(n3(va[:3]),s)
        if n3(va[3:6])>rootw[0]: rootw=(n3(va[3:6]),s)
        for x in jm:
            qidx=x["core_q_index"]; v=x["core_v_index"]; st=js[qidx]; value=qa[qidx]
            st["q_min"]=min(st["q_min"],value); st["q_max"]=max(st["q_max"],value)
            if abs(va[v])>st["abs_v_max"]:
                st["abs_v_max"]=abs(va[v]); st["abs_v_max_step"]=s
            st["final_q"]=value
            lo,hi=x["source_range"]
            exc=lo-value if value<lo else value-hi if value>hi else 0.0
            if value<lo: st["below_range_samples"]+=1
            if value>hi: st["above_range_samples"]+=1
            if exc>st["largest_excursion"]:
                st["largest_excursion"]=exc; st["largest_excursion_step"]=s
        if prevq is not None:
            if r["q_before_fingerprint_fnv64"]==prevq: qmatches+=1
            else: qfails.append([s-1,s,prevq,r["q_before_fingerprint_fnv64"]])
            if r["v_before_fingerprint_fnv64"]==prevv: vmatches+=1
            else: vfails.append([s-1,s,prevv,r["v_before_fingerprint_fnv64"]])
        prevq=r["q_accepted_fingerprint_fnv64"]; prevv=r["v_accepted_fingerprint_fnv64"]
    req(qcount==N and not qfails and not vfails and qmatches==N-1 and vmatches==N-1,"q/v continuity/count failed")
    req(abs((tlast-tfirst)-(N-1)*statistics.fmean(dts))<1e-5,"trace time does not match dt/count")
    violations=[]
    for st in js.values():
        st["samples"]=N
        if st["below_range_samples"] or st["above_range_samples"]:
            violations.append({k:st[k] for k in ("source_joint_id","source_name","core_joint_index","core_q_index",
              "core_v_index","source_type","source_limited","source_range","core_limit_status",
              "below_range_samples","above_range_samples","largest_excursion","largest_excursion_step",
              "q_min","q_max","abs_v_max","abs_v_max_step","final_q")})
    statuses=Counter(x["core_limit_status"] for x in jm); types=Counter(str(x["source_type"]) for x in jm)
    violations_by_status=Counter(x["core_limit_status"] for x in violations)
    violations_by_type=Counter(str(x["source_type"]) for x in violations)
    worst_range=sorted(violations,key=lambda x:x["largest_excursion"],reverse=True)[:8]
    fastest_joints=sorted(js.values(),key=lambda x:x["abs_v_max"],reverse=True)[:8]

    mom=list(rows(RUN/"resting-com-momentum-diagnostic.csv"))
    coup=list(rows(RUN/"resting-coupled.csv"))
    req(len(mom)==N and len(coup)==N,"COM/coupled trace row count mismatch")
    masses=[float(x["total_body_mass_kg"]) for x in mom]
    mstats=stat(masses)
    residual=[]; residual_last10=[]; residual_witnesses=[]
    for x in mom:
        a=[x["unaccounted_delta_p_"+k+"_ns"] for k in "xyz"]
        if all(v!="" for v in a):
            vals=[float(v) for v in a]; residual += [abs(v) for v in vals]
            if int(x["accepted_step"])>5000: residual_last10 += [abs(v) for v in vals]
            residual_witnesses.append({"accepted_step":int(x["accepted_step"]),"residual_xyz_ns":vals,
              "maximum_abs_component_ns":max(abs(v) for v in vals),
              "delta_com_p_xyz_kg_m_s":[float(x["delta_com_p"+k+"_kg_m_s"]) for k in "xyz"],
              "gravity_impulse_xyz_ns":[float(x["gravity_impulse_"+k+"_ns"]) for k in "xyz"],
              "world_contact_impulse_xyz_ns":[float(x["world_contact_impulse_"+k+"_ns"]) for k in "xyz"],
              "status":x["momentum_residual_status"]})
    residual_witnesses=sorted(residual_witnesses,key=lambda x:x["maximum_abs_component_ns"],reverse=True)[:8]
    momstatus=Counter(x["momentum_residual_status"] for x in mom)
    postmom=[x for x in mom if int(x["accepted_step"])>5000]
    req(len(postmom)==5000,"last-10s momentum window mismatch")
    comwindow=[x for x in mom if int(x["accepted_step"])>=5000]
    req(len(comwindow)==5001,"COM endpoint sample coverage mismatch")
    c0=[float(comwindow[0]["com_"+a+"_m"]) for a in "xyz"]
    c1=[float(comwindow[-1]["com_"+a+"_m"]) for a in "xyz"]
    delta=[c1[i]-c0[i] for i in range(3)]
    times=[float(x["time_s"]) for x in comwindow]; mt=statistics.fmean(times)
    denom=math.fsum((t-mt)**2 for t in times); slopes=[]
    for a in "xyz":
        vals=[float(x["com_"+a+"_m"]) for x in comwindow]; mv=statistics.fmean(vals)
        slopes.append(math.fsum((t-mt)*(v-mv) for t,v in zip(times,vals))/denom)
    cv=[[float(x["com_v"+a+"_m_s"]) for a in "xyz"] for x in postmom]
    com_speeds=[(n3([float(x["com_v"+a+"_m_s"]) for a in "xyz"]),int(x["accepted_step"]),[float(x["com_v"+a+"_m_s"]) for a in "xyz"]) for x in mom]
    com_speed_peak=max(com_speeds,key=lambda x:x[0])
    support_force=[float(x["normal_force_last_physical_step_n"]) for x in postmom]
    normals=[[float(x["support_normal_"+a]) for a in "xyz"] for x in postmom]
    dtmean=statistics.fmean(dts)
    grav=[math.fsum(float(x["gravity_impulse_"+a+"_ns"]) for x in postmom)/(len(postmom)*dtmean) for a in "xyz"]

    contact=defaultdict(lambda:{"normal":0.0,"world":[0.0,0.0,0.0],"rows":0,"nonzero_normal_rows":0,"tangent_sum":0.0,"slip_max":0.0})
    for x in rows(RUN/"resting-com-support-impulses.csv"):
        s=int(x["accepted_step"]); z=contact[s]; normal=float(x["normal_impulse_ns"]); z["normal"]+=normal
        if normal != 0.0: z["nonzero_normal_rows"]+=1
        for i,a in enumerate("xyz"): z["world"][i]+=float(x["impulse_world_"+a+"_ns"])
        z["tangent_sum"]+=math.hypot(float(x["tangent0_impulse_ns"]),float(x["tangent1_impulse_ns"]))
        z["slip_max"]=max(z["slip_max"],float(x["point_slip_pre_step_contact_J_v_accepted_speed_m_s"]))
        z["rows"]+=1
    req(set(contact)==set(range(1,N+1)),"support impulse steps incomplete")
    nbal=wbal=0.0; countdiff=[]
    for i,x in enumerate(coup):
        s=int(x["step"]); z=contact[s]
        nbal=max(nbal,abs(z["normal"]-float(x["normal_impulse_ns"])))
        countdiff.append(abs(z["rows"]-int(mom[i]["active_contact_count"])))
        for j,a in enumerate("xyz"): wbal=max(wbal,abs(z["world"][j]-float(mom[i]["world_contact_impulse_"+a+"_ns"])))
    post_contact=[contact[s] for s in range(5001,N+1)]
    duration=len(post_contact)*dtmean
    normal_total=math.fsum(x["normal"] for x in post_contact)
    avg_normal=normal_total/duration
    tangent_total=math.fsum(x["tangent_sum"] for x in post_contact)
    active_counts=[int(x["active_contact_count"]) for x in mom]
    candidate_slots=[contact[s]["rows"] for s in range(1,N+1)]
    nonzero_normal_slots=[contact[s]["nonzero_normal_rows"] for s in range(1,N+1)]
    progress=[]
    preg=re.compile(r"human_standing_progress=accepted\s+(.*)")
    for line in (RUN/"native.log").open("r",encoding="utf-8",errors="replace"):
        mtch=preg.search(line)
        if not mtch: continue
        f=dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)",mtch.group(1)))
        if "step" not in f: continue
        progress.append({"step":int(f["step"]),
          "friction_witness":f.get("friction_witness"),
          "util":float(f["maximum_friction_utilization"]),
          "sat":int(f["saturated_friction_contacts"]),
          "assist_f":float(f["root_assistance_force_n"]),
          "assist_t":float(f["root_assistance_torque_nm"]),
          "tangent_force":float(f["maximum_tangent_force_n"])})
    req(len(progress)==N and [x["step"] for x in progress]==list(range(1,N+1)),"progress/friction log sequence mismatch")
    util=[x["util"] for x in progress]; sat=[x["sat"] for x in progress]
    pen=[(float(x["peak_penetration_m"]),int(x["step"])) for x in coup]
    peak=max(pen); gaps=[float(x["min_contact_gap_m"]) for x in coup]
    last10_pen=[v for v,s in pen if s>5000]
    tangent_peak=max(progress,key=lambda x:x["tangent_force"])
    af=[float(x["root_assistance_n"]) for x in coup]; at=[float(x["root_assistance_nm"]) for x in coup]
    for x in progress: af.append(x["assist_f"]); at.append(x["assist_t"])
    inputs={str(p):sha(p) for p in [meta_p,inv_p,MANIFEST,RUN/"resting-com-q-integration.csv",
       RUN/"resting-com-momentum-diagnostic.csv",RUN/"resting-com-support-impulses.csv",
       RUN/"resting-coupled.csv",RUN/"native.log"]}
    report={
      "schema":"numi.human.native-mechanical-correctness-review.v1","complete":True,
      "scope":"Offline accepted-state and contact audit of completed native 909; no simulation or source/assets modified.",
      "run":{"path":str(RUN),"exit_code":meta["exit_code"],"native_wall_seconds":meta["wall_seconds"],
        "accepted_steps":N,"requested_dt_s":DT_REQUESTED,"trace_dt_s":stat(dts),
        "first_sample_time_s":tfirst,"last_sample_time_s":tlast,
        "source_files_changed_during_run":meta["source_files_changed_during_run"],
        "loaded_metal_runtime":meta["loaded_metal_runtime"],
        "native_executable_sha256":meta["asset_sha256"].get(argv[0])},
      "source_joint_map":{"manifest_path":str(MANIFEST),"manifest_sha256":map_sha,
        "core_nq":core["nq"],"core_nv":core["nv"],"source_joint_count":len(jm),
        "q_state_count_per_sample":129,"v_state_count_per_sample":128,
        "source_type_counts":dict(types),"core_limit_status_counts":dict(statuses),
        "all_map_entries_source_limited_with_finite_ranges":True,
        "q_index_coverage":[7,128],"v_index_coverage":[6,127],
        "range_comparison":"Direct against source_range using the 9-significant-digit serialized accepted q text, without added tolerance.",
        "source_joints":list(js.values()),"range_excursion_joint_count":len(violations),
        "range_excursion_joints":violations,
        "source_range_excursion_samples_total":sum(x["below_range_samples"]+x["above_range_samples"] for x in js.values()),
        "range_excursion_joint_counts_by_core_limit_status":dict(violations_by_status),
        "range_excursion_joint_counts_by_source_type":dict(violations_by_type),
        "largest_source_range_excursions":worst_range,
        "highest_observed_source_joint_speeds":fastest_joints},
      "accepted_qv_state":{"rows":qcount,"q_length":129,"v_length":128,
        "all_before_preprojection_accepted_vectors_finite":True,
        "adjacent_accepted_to_next_before_fingerprint_matches":{"q":qmatches,"v":vmatches},
        "fingerprint_discontinuities":{"q":qfails,"v":vfails},
        "terminal_accepted_fingerprints_fnv64":{"q":prevq,"v":prevv},
        "max_root_quaternion_unit_norm_error":{"absolute_error":qerr[0],"step":qerr[1]},
        "max_root_linear_speed_m_s":{"speed":rootv[0],"step":rootv[1]},
        "max_root_angular_speed_rad_s":{"speed":rootw[0],"step":rootw[1]}},
      "support_and_momentum":{"window":"Force/impulse steps 5001..10000 ([10,20] s); COM endpoint states step 5000 and 10000.",
        "represented_body_mass_kg":stat(masses),"reference_weight_at_9.81_m_s2_n":mstats["mean"]*G,
        "gravity_force_vector_from_observer_impulse_n_xyz":grav,
        "gravity_force_magnitude_n":n3(grav),
        "mean_last10s_support_normal_force_n":stat(support_force),
        "last10s_normal_impulse_over_duration_n":avg_normal,
        "last10s_support_to_72kg_reference_weight_ratio":avg_normal/(72.0*G),
        "mean_support_normal_direction_xyz":[statistics.fmean(x[i] for x in normals) for i in range(3)],
        "com_last10s":{"endpoint_delta_mm_xyz":[1000*x for x in delta],"endpoint_displacement_mm":1000*n3(delta),
          "linear_trend_mm_per_min_xyz":[60000*x for x in slopes],
          "mean_velocity_m_s_xyz":[statistics.fmean(x[i] for x in cv) for i in range(3)],
          "max_speed_m_s":max(n3(x) for x in cv),"terminal_velocity_m_s_xyz":cv[-1],
          "maximum_com_speed_full_horizon":{"speed_m_s":com_speed_peak[0],"accepted_step":com_speed_peak[1],"velocity_xyz_m_s":com_speed_peak[2]}},
        "support_normal_impulse_csv_vs_coupled_max_abs_error_ns":nbal,
        "world_contact_impulse_csv_vs_com_diagnostic_max_abs_error_ns":wbal,
        "candidate_contact_slots_per_step":stat(candidate_slots),
        "nonzero_normal_impulse_slots_per_step":stat(nonzero_normal_slots),
        "support_slot_counts_are_not_active_contact_counts":True,
        "last10s_sum_contact_tangent_impulse_magnitude_ns":tangent_total,
        "last10s_sum_contact_normal_impulse_ns":normal_total,
        "last10s_tangent_to_normal_impulse_magnitude_ratio_diagnostic_only":tangent_total/normal_total if normal_total else None,
        "momentum_residual_status_counts":dict(Counter(x["momentum_residual_status"] for x in mom)),
        "absolute_unaccounted_delta_p_component_ns":stat(residual),
        "absolute_unaccounted_delta_p_component_last10s_ns":stat(residual_last10),
        "largest_unaccounted_delta_p_steps":residual_witnesses,
        "observer_scope":"The native label is per_step_delta_p_minus_gravity_and_contact_only; this is not a complete all-force closure test."},
      "contacts":{"peak_penetration_m":{"maximum":peak[0],"step":peak[1],"distribution":stat([x[0] for x in pen])},
        "minimum_contact_gap_m":min(gaps),"active_contact_count_distribution":stat(active_counts),
        "peak_penetration_last10s_m":stat(last10_pen),
        "friction":{"source":"Native per-accepted-step maximum_friction_utilization with friction_witness=measured.",
          "witness_counts":dict(Counter(x["friction_witness"] for x in progress)),
          "maximum_utilization":max(util),"utilization_distribution":stat(util),
          "steps_reported_over_one":sum(x>1.0 for x in util),
          "saturated_contacts_per_step":stat(sat),"max_tangent_force_n":max(x["tangent_force"] for x in progress),
          "maximum_tangent_force_witness":{"accepted_step":tangent_peak["step"],"force_n":tangent_peak["tangent_force"]},
          "interpretation":"Native measured Coulomb utilization/saturation observer; not independently reimplemented."}},
      "root_assistance":{"nonzero_force_steps":sum(x!=0.0 for x in af),
        "nonzero_torque_steps":sum(x!=0.0 for x in at),
        "max_abs_force_n":max(abs(x) for x in af),"max_abs_torque_nm":max(abs(x) for x in at)},
      "interpretation":{"numerical":"Fingerprint continuity, source-range checks, dimensions, and momentum/contact impulse reconciliation are numeric/runtime evidence.",
        "plausibility":"Support force, drift, friction, and penetration are observed mechanics; they alone do not establish human biomechanical validity or clinical realism."},
      "input_sha256":inputs,"runtime_asset_sha256":meta["asset_sha256"],
      "reproducer_script_path":str(Path(__file__).resolve()),"reproducer_script_sha256":sha(__file__)}
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"report_path":str(report_path),"report_sha256":sha(report_path),
      "script_path":str(Path(__file__).resolve()),"script_sha256":sha(__file__),
      "source_range_excursion_joint_count":len(violations),
      "largest_source_range_excursions":worst_range,"qv_boundary_matches":[qmatches,vmatches],
      "quaternion_norm_error":qerr,"body_mass_kg":mstats,
      "last10s_support_normal_force_n":stat(support_force),"weight_n":mstats["mean"]*G,
      "last10s_com_delta_mm":[1000*x for x in delta],"peak_penetration":peak,
      "max_friction_utilization":max(util),"momentum_residual_abs_component_ns":stat(residual),
      "momentum_residual_last10s_abs_component_ns":stat(residual_last10),
      "nonzero_assistance":[sum(x!=0.0 for x in af),sum(x!=0.0 for x in at)]},indent=2,sort_keys=True))
if __name__=="__main__": main()
