from pathlib import Path
import json,hashlib
N=Path(__file__).resolve().parent;S=N.parent;base=S/"native-import-081/install-motion.plan.json"
def pin(p):
 p=Path(p)
 with p.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
 return {"path":str(p),"sha256":digest,"bytes":p.stat().st_size}
p=json.loads(base.read_text())
for domain in ["input_pins","role_input_pins"]:
 for raw,expected in p[domain].items():assert pin(raw)["sha256"]==expected,raw
assert pin(p["native_binary"]["path"])["sha256"]==p["native_binary"]["sha256"]
for raw,v in p["runtime_dependencies"].items():assert pin(raw)["sha256"]==v["sha256"],raw
def rewrite(x):
 if isinstance(x,str):return x.replace(str(S/"native-import-081"),str(N))
 if isinstance(x,list):return [rewrite(v) for v in x]
 if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
 return x
stage=rewrite(p["stages"][0]);a=stage["argv"];a[a.index("--muscle-step-count")+1]="1";a[a.index("--muscle-step-seconds")+1]="0.000015625";stage["environment"]["NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT"]="0";stage["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"]="0,1";stage["expected_accepted_steps"]=1;stage["environment"]["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP"]="1";p["stages"]=[stage]
p["status"]="PREPARED_SINGLE_ORIGINAL_TISSUE_MICROSTEP_NOT_YET_LAUNCHED";p["measurement_scope"]="one complete-scene tissue microstep; same tissue dt as081, finer body/controller cadence; no sustained-motion or timestep qualification";p["source_qualifier"]=pin(Path(__file__));p["retained_prior_plan"]=pin(base);p["diagnostic_change"]={"physical_frame_dt_s":.000015625,"maximum_rate_exponent":0,"frames":1,"tissue_microsteps":1,"retained_trial081_tissue_microstep_dt_s":.000015625,"physical_equations_changed":False,"forces_disabled":False,"acceptance_gates_changed":False}
assert not (N/"install-motion").exists();(N/"install-motion.plan.json").write_text(json.dumps(p,indent=2)+"\n");print("prepared and input pins verified",N)
