from pathlib import Path
import sys,json,hashlib,csv,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
N=A/"ecu-apl-native-composition-005/baseline/native-run"
B=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011/baseline/native-run"
O=A/"ecu-apl-native-composition-005/native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),N/"invocation.json",N/"run-metadata.json",N/"resting-coupled.csv",B/"resting-coupled.csv",Path(ca.__file__),Path(ci.__file__)]}
trace=list(csv.DictReader((N/"resting-coupled.csv").open()))
base=list(csv.DictReader((B/"resting-coupled.csv").open()))
assert len(trace)==1000
mismatch=[i for i,(x,y) in enumerate(zip(trace,base)) if x!=y]
result={"scope":"Actual native capture self/topology validation for four repaired passive surfaces, and complete short coupled-trace prefix comparison with unchanged baseline011. Not whole-anatomy or five-minute qualification.","pins":pins,"trace_rows":len(trace),"trace_columns":len(trace[0]),"all_coupled_trace_rows_exact_baseline_prefix":not mismatch,"mismatching_rows":mismatch,"captures":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
for receipt in sorted((N/"accepted-geometry").glob("*.receipt.json"),key=lambda p:int(p.name.split("-")[1].split(".")[0])):
 rc=json.loads(receipt.read_text());pack=receipt.with_name(receipt.name.replace(".receipt.json",".mrvpack"))
 pins[str(receipt)]=sha(receipt);pins[str(pack)]=sha(pack)
 assert rc["pack_file_sha256"]==pins[str(pack)] and rc["physical_endpoint"]=="accepted"
 keys={(51005,s) for s in [115,116,147,148]}
 positions,surfaces,_=ca._pack_surfaces(pack,keys)
 cr={"accepted_step":rc["accepted_step"],"rows":[]};result["captures"].append(cr)
 for _,sid in sorted(keys):
  f=surfaces[(51005,sid)]["faces"];used=np.unique(f);p=positions[used];local=np.searchsorted(used,f)
  u,iv=np.unique(p,axis=0,return_inverse=True);qf=iv[local]
  t=analyze_topology(u.astype(float).tolist(),qf.tolist());rec=ca._exact_surface_records(u,qf)
  audit=ci._audit_pair(rec,rec,same_surface=True)
  cr["rows"].append({"stable_id":sid,"self_intersection_count":audit["count"],"topology":{k:t[k] for k in ["closed_oriented_manifold_candidate","face_component_count","euler_characteristic","degenerate_face_ids","duplicate_face_ids","vertex_manifold_defect_ids"]}})
 save()
assert len(result["captures"])==8
result["all_native_repaired_surface_checks_pass"]=all(r["self_intersection_count"]==0 and r["topology"]["closed_oriented_manifold_candidate"] for c in result["captures"] for r in c["rows"])
result["complete"]=True;result["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()
print(json.dumps({k:result[k] for k in ["trace_rows","trace_columns","all_coupled_trace_rows_exact_baseline_prefix","all_native_repaired_surface_checks_pass","complete","inputs_unchanged"]}),flush=True)
