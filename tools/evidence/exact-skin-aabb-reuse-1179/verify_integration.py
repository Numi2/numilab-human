from pathlib import Path
import gc,hashlib,json,os,resource,sys,time
import numpy as np
W=Path("/Users/n/numi-human-free-apex-publication-1159")
sys.path.insert(0,str(W/"src"))
from numilab_human import common_atlas_skin_clearance as c
from numilab_human import cardiac_cavity_intersections as ci
E=Path("/Users/n/numi-human-resting-evidence-20261005")
R=Path("/Users/n/numi-human-retained-delivery-20261009")
OUT=R/"exact-skin-aabb-reuse-1179"
PACK=R/"native-integrated-resting-study-1170/trials/resting-baseline/output/scene/accepted-geometry/step-47519.mrvpack"
BASE=E/"native-lung-late-skin-audit-runner-1172/closed-baseline-1170/full-attempt-002"
INV=E/"native-complete-skin-containment-audit-890/pair-summary-v3.csv"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
inputs=[PACK,INV,BASE/"step-47519.targets.jsonl",BASE/"step-47519.crossing-witnesses.jsonl",W/"src/numilab_human/cardiac_cavity_intersections.py",W/"src/numilab_human/common_atlas_skin_clearance.py"]
pins={str(p):sha(p) for p in inputs}
keys,*_=c._load_target_inventory(INV)
t=time.perf_counter()
positions,surfaces,shape=c._pack_surfaces(PACK,keys)
skinfaces=surfaces[(51007,1)]["faces"]
unique=np.unique(skinfaces)
skin= c._exact_surface_records(positions[unique].astype("<f4").astype(np.float64),np.searchsorted(unique,skinfaces))
setup=time.perf_counter()-t
expected={}
for line in (BASE/"step-47519.targets.jsonl").read_text().splitlines():
 r=json.loads(line);key=tuple(r["surface"])
 expected[key]={"count":r["intersecting_triangle_pairs"],"aabb_candidate_pairs":r["aabb_candidate_pairs"],"triangle_pairs":[]}
for line in (BASE/"step-47519.crossing-witnesses.jsonl").read_text().splitlines():
 r=json.loads(line);key=tuple(r["target_surface"])
 expected[key]["triangle_pairs"].append([r["skin_source_face_row"],r["target_surface_face_row"]])
for r in expected.values():r["triangle_pairs"].sort()
t=time.perf_counter()
rows=c._target_intersection_audit(skin,{k:surfaces[k]["faces"] for k in keys},positions)
assert len(rows)==859
for key,row in rows.items():
 assert row==expected[tuple(map(int,key.split(":")))],key
payload=json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
assert {str(p):sha(p) for p in inputs}==pins
report={"status":"pass","input_pins":pins,"target_count":len(rows),"rows_sha256":hashlib.sha256(payload).hexdigest(),"wall_s":time.perf_counter()-t,"rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"scope":"existing common skin audit integration, complete859targets at baseline acceptedstep47519, exact lists/counts vs sealed audit"}
(OUT/"integration-verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
print(json.dumps(report),flush=True)
