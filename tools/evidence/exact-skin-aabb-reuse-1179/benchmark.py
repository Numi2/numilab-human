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
results={}
for mode in ["original","prepared_first"]:
 timer=time.perf_counter();tree_time=0.;record_time=0.;audit_time=0.;rows={}
 if mode=="prepared_first":
  t=time.perf_counter();index=ci._prepare_surface_aabb(skin);tree_time=time.perf_counter()-t
 for n,key in enumerate(sorted(keys),1):
  t=time.perf_counter()
  f=surfaces[key]["faces"];u=np.unique(f)
  records=c._exact_surface_records(positions[u].astype("<f4").astype(np.float64),np.searchsorted(u,f))
  record_time+=time.perf_counter()-t
  t=time.perf_counter()
  r=ci._audit_pair(skin,records,same_surface=False) if mode=="original" else ci._audit_pair_prepared_first(index,records)
  audit_time+=time.perf_counter()-t
  compact={k:r[k] for k in ("count","aabb_candidate_pairs","triangle_pairs")}
  assert compact==expected[key],(mode,key,compact,expected[key])
  rows[f"{key[0]}:{key[1]}"]=compact
  del records
  if n%100==0 or n==len(keys):
   print(json.dumps({"mode":mode,"targets":n,"wall_s":time.perf_counter()-timer,"record_s":record_time,"audit_s":audit_time,"rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
 payload=json.dumps(rows,sort_keys=True,separators=(",",":")).encode()
 results[mode]={"wall_s":time.perf_counter()-timer,"record_s":record_time,"audit_s":audit_time,"prepared_tree_s":tree_time,"rows_sha256":hashlib.sha256(payload).hexdigest(),"target_count":len(rows),"exact_pairs":sum(r["count"] for r in rows.values()),"aabb_candidates":sum(r["aabb_candidate_pairs"] for r in rows.values())}
 if mode=="prepared_first": del index
 gc.collect()
assert results["original"]["rows_sha256"]==results["prepared_first"]["rows_sha256"]
assert {str(p):sha(p) for p in inputs}==pins
report={"status":"pass","input_pins":pins,"setup_s":setup,"results":results,"speedup_including_record_build":results["original"]["wall_s"]/results["prepared_first"]["wall_s"],"speedup_audit_only":results["original"]["audit_s"]/(results["prepared_first"]["audit_s"]+results["prepared_first"]["prepared_tree_s"]),"rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"scope":"one actual accepted baseline pose, all859 targets; exact full pair lists and AABB counts equal to retained closed audit and both implementations; shared-host CPU fit/lung diagnosis/native GPU run overlap"}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report),flush=True)
