#!/usr/bin/env python3
import csv,hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent
RUN=Path("/Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020/native-run")
BODY=RUN/"resting-body-motion-partition.csv"
COM=RUN/"resting-com-momentum-diagnostic.csv"
STEPS={5000,125000,155000}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def mat(q):
 x,y,z,w=q
 n=math.sqrt(x*x+y*y+z*z+w*w)
 if abs(n-1)>1e-5: raise ValueError(f"non-unit quaternion norm {n}")
 x,y,z,w=(x/n,y/n,z/n,w/n)
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
def tr(m):return [list(x) for x in zip(*m)]
def mv(m,v):return [sum(m[i][j]*v[j] for j in range(3)) for i in range(3)]
def mmv(a,b,v):return [sum((a[i][j]-b[i][j])*v[j] for j in range(3)) for i in range(3)]
def sub(a,b):return [x-y for x,y in zip(a,b)]
def add(*vs):return [sum(v[i] for v in vs) for i in range(3)]
def norm(v):return math.sqrt(sum(x*x for x in v))
def one(start,end,root,com):
 r0=root[start]["r"];r1=root[end]["r"];q0=root[start]["q"];q1=root[end]["q"]
 R0=mat(q0);R1=mat(q1);R0T=tr(R0);R1T=tr(R1)
 C0=com[start];C1=com[end]
 c0=mv(R0T,sub(C0,r0));c1=mv(R1T,sub(C1,r1))
 trans=sub(r1,r0);rot=mmv(R1,R0,c0);art=mv(R1,sub(c1,c0));delta=sub(C1,C0)
 dot=abs(sum(a*b for a,b in zip(q0,q1)));angle=2*math.acos(min(1.,dot))
 reconstructed=add(trans,rot,art)
 return {"steps":[start,end],"root_quaternion_xyzw":[q0,q1],"root_rotation_angle_rad":angle,
         "world_com_delta_m":delta,"root_translation_delta_m":trans,
         "root_rotation_orbit_term_m":rot,"root_frame_articulated_com_term_m":art,
         "decomposition_sum_m":reconstructed,"decomposition_residual_m":sub(delta,reconstructed),
         "root_frame_com_endpoints_m":[c0,c1],
         "magnitudes_mm":{"world_com":norm(delta)*1000,"translation":norm(trans)*1000,
                           "rotation_orbit":norm(rot)*1000,"articulated_relative":norm(art)*1000}}
root={}
with BODY.open(newline="") as f:
 reader=csv.DictReader(f)
 for row in reader:
  s=int(row["accepted_step"])
  if s in STEPS and int(row["body_index"])==0:
   if s in root:raise ValueError(f"duplicate root row {s}")
   root[s]={"r":[float(row[f"root_{a}_m"]) for a in "xyz"],
            "q":[float(row[f"orientation_{a}"]) for a in ("x","y","z","w")],
            "name":row["body_name"],"mass":float(row["mass_kg"])}
  if len(root)==len(STEPS):break
if set(root)!=STEPS:raise ValueError(f"missing root rows {set(STEPS)-set(root)}")
com={}
with COM.open(newline="") as f:
 for row in csv.DictReader(f):
  s=int(row["accepted_step"])
  if s in STEPS:
   if s in com:raise ValueError(f"duplicate COM row {s}")
   com[s]=[float(row[f"com_{a}_m"]) for a in "xyz"]
if set(com)!=STEPS:raise ValueError(f"missing COM rows {set(STEPS)-set(com)}")
for s,r in root.items():
 if r["mass"]!=0.0 or r["name"]!="unnamed":raise ValueError(f"unexpected massless root record at {s}: {r}")
report={"schema":"numi.human.root-frame-com-drift-decomposition.v1",
 "status":"measured_descriptive_identity",
 "scope":"Exact endpoint coordinate decomposition from accepted observer fields. q is the root body-to-world quaternion (xyzw); the rotation matrix maps root-frame coordinates to world. This partitions measured COM displacement into carrier translation, orbit from root rotation acting on initial root-frame COM, and change in root-frame COM transformed at the endpoint. It is not a force attribution or cause proof.",
 "inputs":[{"path":str(p),"sha256":sha(p)} for p in (BODY,COM)],
 "root_records":{str(k):v for k,v in root.items()},
 "com_records_m":{str(k):v for k,v in com.items()},
 "decompositions":{"10s_to_terminal":one(5000,155000,root,com),
                   "250s_to_terminal":one(125000,155000,root,com)}}
out=OUT/"report.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report["decompositions"].items()},indent=2))
print("report_sha256="+sha(out))
