"""Bounded source-ancestry-derived spatial mass transport experiment.
The distance envelope is precomputed from declared child/source-cell centroids;
it is a reference-remapping neighborhood, not a changed material/physics limit.
"""
from pathlib import Path
import argparse,csv,json,hashlib,subprocess,time,math
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--volume",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
def rows(p):
 with p.open(newline="") as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pr=rows(CAP/"compiled-parent-cell-mass.csv.nodes.csv");tr=rows(CAP/"compiled-parent-cell-mass.csv.tets.csv");lr=rows(CAP/"compiled-parent-cell-mass.csv.labels.csv")
pp=np.array([[float(r["current_"+k+"_m"]) for k in "xyz"] for r in pr]);pt=np.array([[int(r["parent_node"+str(k)]) for k in range(4)] for r in tr])
with np.load(a.candidate,allow_pickle=False) as z:
 cp=z["child_parent_tet_row"].astype(np.int64);ct=z["volume_tetrahedra"].astype(np.int64);p=z["points_reference_world_m_f64"].astype(np.float32).astype(np.float64)
 for key in ["material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"]:
  if not np.array_equal(z[key],np.array([int(r[key]) for r in lr])[cp]):raise ValueError("source cell identity mismatch "+key)
# Reject a folded connected cell complex before invoking conservative flow.
from audit_coarse_material_bone_volume_427 import exterior,volumes
try:
 if not np.isfinite(p).all() or ct.min()<0 or ct.max()>=len(p) or np.any(volumes(p,ct)<=0):
  raise ValueError("nonpositive or invalid canonical child cells")
 exterior(ct)
except ValueError as exc:
 report={"stage":"connected_cell_preparation","status":"FAIL","reason":str(exc),"candidate_sha256":sha(a.candidate),"physical_admission":False,"mass_mapper_started":False}
 (a.out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report));raise SystemExit(7)
d=np.linalg.norm(p[ct].mean(1)-pp[pt].mean(1)[cp],axis=1)
cap=max(.0002,math.ceil(float(d.max())/1e-6)*1e-6+1e-6)
cli=Path("/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-physical-skin-child-mass-mapping-cli")
base=[str(cli),"--parent-mass-csv",str(CAP/"compiled-parent-cell-mass.csv"),"--labels-csv",str(CAP/"compiled-parent-cell-mass.csv.labels.csv"),"--child-node-file",str(a.volume/"mesh.node"),"--child-ele-file",str(a.volume/"mesh.ele"),"--child-labels-csv",str(a.volume/"child.labels.csv"),"--coordinate-frame-sha256","0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a","--source-body-count","157"]
plan={"script_sha256":sha(Path(__file__)),"topology_gate_sha256":sha(P/"audit_coarse_material_bone_volume_427.py"),"connected_cell_preparation":"PASS","scope":"bounded conservative reference mass transport with declared source-cell ancestry; no native motion admission","candidate_sha256":sha(a.candidate),"binary_sha256":sha(cli),"source_parent_packet_sha256":sha(CAP/"compiled-parent-cell-mass.csv"),"child_volume_sha256":{f:sha(a.volume/f) for f in ["mesh.node","mesh.ele","child.labels.csv"]},"maximum_declared_source_centroid_distance_m":float(d.max()),"predeclared_transport_envelope_m":cap,"rule":"ceil(max declared child-to-original-source-parent centroid displacement / 1um)*1um +1um, floor200um; no further enlargement after results","sensitivity_factors":[.5,1.0],"density_bounds_unchanged":True,"physical_admission":False}
(a.out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");results=[]
for factor in [.5,1.0]:
 bound=cap*factor;cmd=base+["--maximum-transport-distance-m",format(bound,".17g")];t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True);tag="sensitivity-half" if factor==.5 else "declared-envelope"
 (a.out/(tag+".stdout")).write_text(r.stdout);(a.out/(tag+".stderr")).write_text(r.stderr)
 results.append(dict(factor=factor,bound_m=bound,command=cmd,exit_code=r.returncode,wall_s=time.monotonic()-t,stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr.encode()).hexdigest()))
 (a.out/"result.json").write_text(json.dumps({"plan":plan,"trials":results},indent=2)+"\n")
print(json.dumps({"maximum_declared_source_centroid_distance_m":float(d.max()),"envelope_m":cap,"trials":[{k:r[k] for k in ["factor","exit_code","wall_s"]} for r in results]}))
