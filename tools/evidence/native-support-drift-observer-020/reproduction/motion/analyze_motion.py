#!/usr/bin/env python3
"""Read-only body/support attribution for closed opt-in native observer runs."""
from pathlib import Path
import argparse, csv, hashlib, json, math

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def rows(path):
    with Path(path).open(newline="") as f:yield from csv.DictReader(f)

def f(row,key):
    v=float(row[key])
    if not math.isfinite(v):raise ValueError("nonfinite "+key)
    return v

def analyze(scene,manifest_path,first,last,out):
    scene=Path(scene).resolve();out=Path(out).resolve()
    if out.exists() or scene==out or scene in out.parents:raise ValueError("require fresh output outside run")
    if first<0 or last<=first:raise ValueError("invalid accepted-step interval")
    names=["resting-body-motion-partition.csv","resting-support-selected-skin-witness.csv",
           "resting-support-point-motion.csv","resting-com-support-impulses.csv",
           "resting-com-momentum-diagnostic.csv","invocation.json"]
    paths={n:scene/n for n in names}
    for p in paths.values():
        if p.is_symlink() or not p.is_file():raise ValueError("missing or indirect input "+str(p))
    inputs={n:{"path":str(p),"sha256":sha(p)} for n,p in paths.items()}
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text());body_names=manifest["core_tree"]["body_order"]
    rigid=manifest_path.parent/manifest["payloads"]["rigid"]["file"]
    rigid_sha=sha(rigid);inv=json.loads(paths["invocation.json"].read_text())
    if not (rigid_sha==manifest["payloads"]["rigid"]["sha256"]==inv["asset_sha256"].get(str(rigid))):
        raise ValueError("source body names are not bound to native rigid input")
    selected=lambda r:first<=int(r["accepted_step"])<=last
    bodies={};steps=set()
    for r in rows(paths["resting-body-motion-partition.csv"]):
        if not selected(r):continue
        i=int(r["body_index"]);step=int(r["accepted_step"]);steps.add(step)
        b=bodies.setdefault(i,{"first":r,"last":r,"count":0,"velocity_sum":[0.,0.,0.],"angular_sum":[0.,0.,0.]})
        if b["count"] and step<=int(b["last"]["accepted_step"]):raise ValueError("nonmonotonic body samples")
        b["last"]=r;b["count"]+=1
        for k,axis in enumerate("xyz"):
            b["velocity_sum"][k]+=f(r,"linear_velocity_"+axis+"_m_s")
            b["angular_sum"][k]+=f(r,"angular_velocity_"+axis+"_rad_s")
    if not steps or min(steps)!=first or max(steps)!=last:raise ValueError("requested endpoints absent")
    if set(bodies)!=set(range(len(body_names))) or any(b["count"]!=len(steps) for b in bodies.values()):
        raise ValueError("incomplete body inventory/cadence")
    mass=sum(f(b["last"],"mass_kg") for b in bodies.values())
    root0,root1=bodies[0]["first"],bodies[0]["last"]
    root_delta=[f(root1,"root_"+a+"_m")-f(root0,"root_"+a+"_m") for a in "xyz"]
    perbody=[]
    for i,b in bodies.items():
        a,z=b["first"],b["last"];m=f(z,"mass_kg")
        if m!=f(a,"mass_kg"):raise ValueError("body mass changed")
        delta=[(f(z,"mass_weighted_relative_com_"+k+"_kg_m")-f(a,"mass_weighted_relative_com_"+k+"_kg_m"))/mass for k in "xyz"]
        qa=[f(a,"orientation_"+k) for k in "xyzw"];qz=[f(z,"orientation_"+k) for k in "xyzw"]
        dot=abs(sum(x*y for x,y in zip(qa,qz)))/(math.sqrt(sum(x*x for x in qa))*math.sqrt(sum(x*x for x in qz)))
        perbody.append({"body_index":i,"source_name":body_names[i],"mass_kg":m,
           "relative_com_contribution_delta_m":delta,
           "mean_world_linear_velocity_m_s":[x/b["count"] for x in b["velocity_sum"]],
           "mean_world_angular_velocity_rad_s":[x/b["count"] for x in b["angular_sum"]],
           "world_orientation_endpoint_angle_rad":2*math.acos(min(1.,dot))})
    predicted=[root_delta[k]+sum(b["relative_com_contribution_delta_m"][k] for b in perbody) for k in range(3)]
    com={int(r["accepted_step"]):r for r in rows(paths["resting-com-momentum-diagnostic.csv"]) if selected(r)}
    if set(com)!=steps:raise ValueError("COM and body cadence disagree")
    observed=[f(com[last],"com_"+k+"_m")-f(com[first],"com_"+k+"_m") for k in "xyz"]
    residual=[a-b for a,b in zip(predicted,observed)]
    if max(abs(x) for x in residual)>1e-12:raise ValueError("body partition fails COM identity")
    impulses={(int(r["accepted_step"]),int(r["contact_index"])):f(r,"normal_impulse_ns")
              for r in rows(paths["resting-com-support-impulses.csv"]) if selected(r)}
    regions={}
    for r in rows(paths["resting-support-selected-skin-witness.csv"]):
        if not selected(r):continue
        step=int(r["accepted_step"]);reg=int(r["region"]);winner=int(r["selected_skin_vertex_local"])
        d=regions.setdefault(reg,{"count":0,"previous":None,"winners":set(),"sampled_winner_changes":0,
            "sampled_winner_changes_with_positive_normal_impulse":0,"positive_normal_count":0,"speed_sum":0.,"speed_max":0.})
        if d["previous"] is not None and winner!=d["previous"]:
            d["sampled_winner_changes"]+=1
            d["sampled_winner_changes_with_positive_normal_impulse"]+=impulses[(step,reg)]>0
        d["count"]+=1;d["previous"]=winner;d["winners"].add(winner)
        if any(int(r[k]) for k in ["region_error","global_error","nonfinite_jacobian_count"]):
            raise ValueError("selected support witness invalid")
    for r in rows(paths["resting-support-point-motion.csv"]):
        if not selected(r):continue
        step=int(r["accepted_step"]);reg=int(r["contact_index"]);d=regions[reg];bi=int(r["body_index"])
        d["static_seed_body_index"]=bi;d["static_seed_body_name"]=body_names[bi]
        if r["velocity_basis"]!="pre_step_point_J_times_stand_previous_velocity":raise ValueError("unexpected velocity basis")
        if impulses[(step,reg)]>0:
            speed=f(r,"tangential_speed_m_s");d["positive_normal_count"]+=1;d["speed_sum"]+=speed;d["speed_max"]=max(d["speed_max"],speed)
    for d in regions.values():
        if d["count"]!=len(steps):raise ValueError("support cadence mismatch")
        d["unique_selected_vertices"]=sorted(d.pop("winners"));d.pop("previous")
        d["positive_normal_pre_step_tangential_speed_mean_m_s"]=d.pop("speed_sum")/d["positive_normal_count"] if d["positive_normal_count"] else None
        d["positive_normal_pre_step_tangential_speed_max_m_s"]=d.pop("speed_max")
    if any(sha(paths[n])!=item["sha256"] for n,item in inputs.items()):raise ValueError("run inputs changed")
    report={"scope":"Descriptive attribution of accepted native observations; no dynamics update, causal proof, or equilibrium/clinical qualification.",
      "requested_accepted_steps":[first,last],"sample_count":len(steps),
      "actual_time_s":[f(root0,"time_s"),f(root1,"time_s")],"body_mass_kg":mass,
      "root_translation_delta_m":root_delta,"partition_total_com_delta_m":predicted,
      "owner_total_com_delta_m":observed,"partition_identity_residual_m":residual,
      "bodies":sorted(perbody,key=lambda b:abs(b["relative_com_contribution_delta_m"][0]),reverse=True),
      "support_regions":regions,"inputs":inputs,
      "body_name_source":{"path":str(manifest_path),"sha256":sha(manifest_path),"runtime_rigid_sha256":rigid_sha},
      "limitations":["Winner changes are sampled lower bounds; changes between exported steps are not counted.",
         "The contact seed body is metadata; the selected skin point uses weighted multi-body kinematics.",
         "Jv describes the final step's pre-dynamics basis, not a post-solve contact velocity.",
         "Positive normal impulses and selector changes establish observations, not the cause of drift."]}
    out.mkdir(parents=True);(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["actual_time_s","root_translation_delta_m","owner_total_com_delta_m","partition_identity_residual_m"]}))
    print("report_sha256="+sha(out/"report.json"))

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene",required=True,type=Path);ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--first-step",required=True,type=int);ap.add_argument("--last-step",required=True,type=int)
    ap.add_argument("--out",required=True,type=Path);a=ap.parse_args()
    analyze(a.scene,a.manifest,a.first_step,a.last_step,a.out)
