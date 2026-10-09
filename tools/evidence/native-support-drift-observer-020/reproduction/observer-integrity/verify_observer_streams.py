#!/usr/bin/env python3
"""Fail-closed integrity check for the completed accepted-motion observer streams."""
from pathlib import Path
import csv,hashlib,json,math,re,struct,datetime
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020")
RUN=BASE/"native-run"
MANIFEST=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json")
OUT=Path(__file__).resolve().parent/"report.json"
DT=struct.unpack("<f",struct.pack("<f",0.002))[0]
FIRST,LAST,STRIDE=8,155000,8
EXPECTED_STEPS=list(range(FIRST,LAST+1,STRIDE))
BODY_COUNT=157
REGION_COUNT=32
FILES=("resting-body-motion-partition.csv","resting-support-point-motion.csv","resting-support-selected-skin-witness.csv")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def csv_rows(path):
 with Path(path).open(newline="") as f:
  r=csv.DictReader(f)
  if not r.fieldnames or len(set(r.fieldnames))!=len(r.fieldnames):raise ValueError("bad/duplicate header: "+str(path))
  for n,row in enumerate(r,2):
   if None in row or any(v is None for v in row.values()):raise ValueError(f"malformed CSV row {n}: {path}")
   yield row

def finite_columns(row, skip=()):
 for k,v in row.items():
  if k in skip:continue
  try:x=float(v)
  except Exception as e:raise ValueError(f"nonnumeric {k}={v!r}") from e
  if not math.isfinite(x):raise ValueError(f"nonfinite {k}={v!r}")
def validate_time(row,field,step):
 expected=float(step)*DT
 if float(row[field])!=expected:raise ValueError(f"time mismatch {field} step={step}: {row[field]} != {expected}")
def check_body(path):
 count=0;step_index=0;body_next=0;mass={};names={};rootvals=None;step_total_mass=0.;min_mass=math.inf;max_mass=-math.inf
 for r in csv_rows(path):
  step=int(r["accepted_step"]);bi=int(r["body_index"])
  if step_index>=len(EXPECTED_STEPS):raise ValueError("extra body rows")
  want=EXPECTED_STEPS[step_index]
  if step!=want or bi!=body_next:raise ValueError(f"body ordering/cadence mismatch at row {count+2}: {(step,bi)} expected {(want,body_next)}")
  validate_time(r,"time_s",step)
  finite_columns(r,skip=("accepted_step","body_index","body_name"))
  m=float(r["mass_kg"])
  if bi in mass and mass[bi]!=m:raise ValueError(f"body mass changed index {bi}")
  mass[bi]=m
  if bi in names and names[bi]!=r["body_name"]:raise ValueError(f"body label changed index {bi}")
  names[bi]=r["body_name"]
  if bi==0:
   rootvals=tuple(float(r["root_"+a+"_m"]) for a in "xyz")
   step_total_mass=0.
  else:
   if tuple(float(r["root_"+a+"_m"]) for a in "xyz")!=rootvals:raise ValueError(f"root translation inconsistent within step {step}")
  step_total_mass+=m
  count+=1;body_next+=1
  if body_next==BODY_COUNT:
   if not math.isclose(step_total_mass,72.0,rel_tol=0,abs_tol=2e-6):raise ValueError(f"body mass sum drift at {step}: {step_total_mass}")
   min_mass=min(min_mass,step_total_mass);max_mass=max(max_mass,step_total_mass)
   body_next=0;step_index+=1
 if count!=BODY_COUNT*len(EXPECTED_STEPS) or step_index!=len(EXPECTED_STEPS) or body_next!=0:raise ValueError(f"body rows {count} != expected {BODY_COUNT*len(EXPECTED_STEPS)}")
 if set(mass)!=set(range(BODY_COUNT)):raise ValueError("incomplete body index set")
 return {"data_rows":count,"accepted_steps":len(EXPECTED_STEPS),"body_count":BODY_COUNT,"body_indices":[min(mass),max(mass)],"mass_sum_min_max_kg":[min_mass,max_mass],"runtime_body_labels_unique":len(set(names.values()))}
def check_contact_stream(path,kind):
 count=0;step_index=0;region_next=0;static={};last_step=None
 for r in csv_rows(path):
  step=int(r["accepted_step"])
  if step_index>=len(EXPECTED_STEPS):raise ValueError("extra contact rows")
  want=EXPECTED_STEPS[step_index]
  reg=int(r["contact_index"] if kind=="point" else r["region"])
  if step!=want or reg!=region_next:raise ValueError(f"{kind} order/cadence mismatch row {count+2}: {(step,reg)} expected {(want,region_next)}")
  if kind=="point":
   pre=int(r["pre_step_index"]);tf="accepted_time_s";pretf="pre_step_time_s"
   if pre!=step-1:raise ValueError(f"point velocity pre-step index mismatch {step}/{pre}")
   validate_time(r,tf,step);validate_time(r,pretf,pre)
   if r["velocity_basis"]!="pre_step_point_J_times_stand_previous_velocity":raise ValueError("unexpected point velocity basis")
   fp=int(r["pre_step_velocity_fingerprint_fnv64"])
   if fp<0 or fp>=2**64:raise ValueError("pre-step velocity fingerprint outside uint64")
   static_key=(int(r["source_geometry_index"]),int(r["body_index"]),int(r["point_query_index"]))
  else:
   pre=int(r["pre_dynamics_step_index"])
   if pre!=step-1:raise ValueError(f"skin selected pre-step mismatch {step}/{pre}")
   validate_time(r,"accepted_time_s",step);validate_time(r,"pre_dynamics_time_s",pre)
   if any(int(r[k])!=0 for k in ("region_error","global_error","nonfinite_jacobian_count")):raise ValueError(f"skin selector invalid at step {step} region {reg}")
   if not r["selected_skin_vertex_local"] or not r["selected_vertex_map_index"]:raise ValueError("missing selected vertex/map index")
   if int(r["selected_skin_vertex_local"])<0 or int(r["selected_vertex_map_index"])<0:raise ValueError("negative selected vertex/map index")
   static_key=(int(r["source_geometry_index"]),int(r["point_query_index"]))
  finite_columns(r,skip=("accepted_step","pre_step_index","pre_dynamics_step_index","contact_index","region","source_geometry_index","body_index","point_query_index","pre_step_velocity_fingerprint_fnv64","selected_skin_vertex_local","selected_vertex_map_index","region_error","global_error","nonfinite_jacobian_count","velocity_basis"))
  if reg in static and static[reg]!=static_key:raise ValueError(f"static contact/region mapping changed region {reg}")
  static[reg]=static_key
  count+=1;region_next+=1;last_step=step
  if region_next==REGION_COUNT:region_next=0;step_index+=1
 if count!=REGION_COUNT*len(EXPECTED_STEPS) or step_index!=len(EXPECTED_STEPS) or region_next!=0:raise ValueError(f"{kind} rows {count} != expected {REGION_COUNT*len(EXPECTED_STEPS)}")
 return {"data_rows":count,"accepted_steps":len(EXPECTED_STEPS),"regions_per_step":REGION_COUNT,"regions":[min(static),max(static)],"static_region_mappings":len(static),"last_step":last_step}
def main():
 if OUT.exists():raise ValueError("refuse existing output")
 manifest=json.loads(MANIFEST.read_text());names=manifest["core_tree"]["body_order"]
 if len(names)!=BODY_COUNT:raise ValueError("body manifest count changed")
 inv=json.loads((RUN/"invocation.json").read_text());rigid=MANIFEST.parent/manifest["payloads"]["rigid"]["file"]
 if sha(rigid)!=manifest["payloads"]["rigid"]["sha256"] or inv["asset_sha256"].get(str(rigid))!=sha(rigid):raise ValueError("source body manifest not bound to runtime rigid payload")
 paths={n:RUN/n for n in FILES}
 hashes={n:sha(p) for n,p in paths.items()}
 results={"body_motion_partition":check_body(paths[FILES[0]]),"support_point_motion":check_contact_stream(paths[FILES[1]],"point"),"selected_skin_witness":check_contact_stream(paths[FILES[2]],"skin")}
 if results["support_point_motion"]["static_region_mappings"]!=REGION_COUNT or results["selected_skin_witness"]["static_region_mappings"]!=REGION_COUNT:raise ValueError("missing support region")
 # Every support point query/source identity must align with the corresponding selected skin witness region.
 point={}
 for r in csv_rows(paths[FILES[1]]):point[int(r["contact_index"])]=(int(r["source_geometry_index"]),int(r["point_query_index"]))
 skin={}
 for r in csv_rows(paths[FILES[2]]):skin[int(r["region"])]=(int(r["source_geometry_index"]),int(r["point_query_index"]))
 if point!=skin:raise ValueError("support point and skin-witness region identities differ")
 for n,p in paths.items():
  if sha(p)!=hashes[n]:raise ValueError("observer stream changed during verification")
 # Verify the emitted profile accounts for its listed sub-buckets.
 log=(RUN/"native.log").read_text(errors="strict")
 matches=re.findall(r"resting_integrated_observer_profile accepted_callbacks=(\d+) total_ms=([0-9.eE+-]+) mean_ms=([0-9.eE+-]+) respiration_trace_csv_ms=([0-9.eE+-]+) q_integration_csv_ms=([0-9.eE+-]+) com_cpu_kinematics_ms=([0-9.eE+-]+) body_motion_csv_ms=([0-9.eE+-]+) com_momentum_csv_ms=([0-9.eE+-]+) support_impulse_csv_ms=([0-9.eE+-]+) csv_flush_ms=([0-9.eE+-]+) presentation_ms=([0-9.eE+-]+) observer_other_ms=([0-9.eE+-]+)",log)
 if len(matches)!=1:raise ValueError("observer profile line missing or duplicated")
 z=matches[0];cb=int(z[0]);total=float(z[1]);mean=float(z[2]);subs=list(map(float,z[3:11]));other=float(z[11]);subtotal=sum(subs)
 if cb!=len(EXPECTED_STEPS) or min([total,mean,other,*subs])<0 or not all(math.isfinite(x) for x in [total,mean,other,*subs]) or abs(total-subtotal-other)>1e-3 or abs(mean-total/cb)>1e-6:raise ValueError("observer profile accounting inconsistency")
 report={"schema":"numi.human.accepted-support-motion-observer-integrity.v1","status":"passed","scope":"Full 310 s opt-in observer output integrity only; this verifies complete finite accepted-state streams, matching region maps, profile arithmetic, and source-bound 157-body inventory. It does not establish drift cause or a physical/anatomical threshold.","run":str(RUN),"accepted_steps":{"first":FIRST,"last":LAST,"stride":STRIDE,"count":len(EXPECTED_STEPS),"actual_terminal_time_s":LAST*DT},"body_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"body_count":BODY_COUNT,"runtime_rigid_path":str(rigid),"runtime_rigid_sha256":sha(rigid)},"streams":{n:{"path":str(paths[n]),"sha256":hashes[n]} for n in FILES},"counts":results,"profile":{"accepted_callbacks":cb,"total_ms":total,"mean_ms":mean,"subbucket_sum_ms":subtotal,"observer_other_ms":other,"accounting_residual_ms":total-subtotal-other},"run_metadata_sha256":sha(RUN/"run-metadata.json"),"invocation_sha256":sha(RUN/"invocation.json")}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":report["status"],"report":str(OUT),"counts":results,"profile":report["profile"]},indent=2))
 print("report_sha256="+sha(OUT))
if __name__=="__main__":main()
