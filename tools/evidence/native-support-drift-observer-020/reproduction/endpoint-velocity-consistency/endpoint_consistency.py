#!/usr/bin/env python3
"""Read-only comparison of sampled endpoint velocities, trapezoid integrals, and finite displacement."""
from pathlib import Path
import csv, hashlib, json, math, datetime
RUN=Path("/Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020/native-run")
OUT=Path(__file__).resolve().parent/"report.json"
WINDOWS=((5000,155000),(125000,155000))
FILES=("resting-com-momentum-diagnostic.csv","resting-body-motion-partition.csv")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
 with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def f(r,k):
 x=float(r[k]);
 if not math.isfinite(x):raise ValueError(f"nonfinite {k}")
 return x
def main():
 compath=RUN/FILES[0];bodypath=RUN/FILES[1]
 com=read(compath);body=read(bodypath)
 cm={int(r["accepted_step"]):r for r in com}
 br={int(r["accepted_step"]):r for r in body if int(r["body_index"])==0}
 out={"schema":"numi.human.support-drift-endpoint-velocity-consistency.v1","scope":"Sampled root and mass-COM endpoint velocities and trapezoid integrals compared with finite accepted-position displacement. Trapezoid covers only cadence-8 samples and is not a continuous exact integral or a drift-cause test.","run":str(RUN),"inputs":{str(p):sha(p) for p in (RUN/"run-metadata.json",RUN/"invocation.json",compath,bodypath)},"windows":[]}
 for first,last in WINDOWS:
  ids=[s for s in sorted(cm) if first<=s<=last]
  if not ids or ids[0]!=first or ids[-1]!=last or any(b-a!=8 for a,b in zip(ids,ids[1:])):raise ValueError("missing endpoint or cadence mismatch")
  if not all(s in br for s in (first,last)):raise ValueError("root endpoint missing")
  a,z=cm[first],cm[last];ra,rz=br[first],br[last]
  dt=f(z,"time_s")-f(a,"time_s")
  if dt<=0:raise ValueError("nonpositive window")
  row={"accepted_steps":[first,last],"accepted_time_s":[f(a,"time_s"),f(z,"time_s")],"sample_count":len(ids),"cadence_steps":8,"elapsed_s":dt}
  for kind,posprefix,velprefix in (("mass_com","com_","com_v"),("root_carrier","root_","root_v")):
   poskeys=[posprefix+k+"_m" for k in "xyz"]
   velkeys=[velprefix+k+"_m_s" for k in "xyz"]
   if kind=="mass_com":start,end=a,z
   else:start,end=ra,rz
   displacement=[f(end,k)-f(start,k) for k in poskeys]
   finite_mean=[d/dt for d in displacement]
   va=[f(a,k) for k in velkeys];vz=[f(z,k) for k in velkeys]
   integ=[]
   for key in velkeys:
    total=0.0
    for s0,s1 in zip(ids,ids[1:]):
     r0,r1=cm[s0],cm[s1]
     total+=(f(r0,key)+f(r1,key))*0.5*(f(r1,"time_s")-f(r0,"time_s"))
    integ.append(total)
   row[kind]={"position_delta_m":displacement,"finite_displacement_mean_velocity_m_s":finite_mean,"sampled_endpoint_velocity_m_s":{"start":va,"end":vz},"cadence8_trapezoid_velocity_integral_m":integ,"position_minus_trapezoid_residual_m":[x-y for x,y in zip(displacement,integ)],"endpoint_velocity_minus_finite_mean_m_s":{"start":[x-y for x,y in zip(va,finite_mean)],"end":[x-y for x,y in zip(vz,finite_mean)]}}
  out["windows"].append(row)
 OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(OUT),"windows":[{"steps":x["accepted_steps"],"mass_com_delta":x["mass_com"]["position_delta_m"],"mass_com_trap_residual":x["mass_com"]["position_minus_trapezoid_residual_m"],"root_delta":x["root_carrier"]["position_delta_m"],"root_trap_residual":x["root_carrier"]["position_minus_trapezoid_residual_m"]} for x in out["windows"]]},indent=2))
 print("report_sha256="+sha(OUT))
if __name__=="__main__":main()
