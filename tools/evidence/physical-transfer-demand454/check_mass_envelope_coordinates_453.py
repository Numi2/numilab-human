from pathlib import Path
import csv,numpy as np,json,math,hashlib
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209");C=P/"native-physical-parent-mass-330/native-source-identity-control-003"
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
n=rows(C/"compiled-parent-cell-mass.csv.nodes.csv");r=rows(C/"compiled-parent-cell-mass.csv.tets.csv")
t=np.array([[int(x["parent_node"+str(i)]) for i in range(4)] for x in r])
current=np.array([[float(x["current_"+k+"_m"]) for k in "xyz"] for x in n]);rest=np.array([[float(x["rest_"+k+"_m"]) for k in "xyz"] for x in n])
checks=[]
for mass,geometry in [("interior-refinement-mass-436","refined-native-producer-433/interior-refinement-control-434/candidate.npz"),("interface-refinement-mass-446","refined-native-producer-433/interface-refinement-control-442-retry-001/candidate.npz"),("hand-cavity-mass-451","reduced-physical-boundary-400/attempt-008-local-cavity-tetgen/pilot-004/candidate-vertexstar-tetgen.npz")]:
 plan=json.loads((P/"refined-native-producer-433"/mass/"plan.json").read_text());path=P/geometry
 assert hashlib.sha256(path.read_bytes()).hexdigest()==plan["candidate_sha256"]
 z=np.load(path);p=z["points_reference_world_m_f64"].astype(np.float32).astype(float);ct=z["volume_tetrahedra"];cp=z["child_parent_tet_row"]
 old=np.linalg.norm(p[ct].mean(1)-current[t].mean(1)[cp],axis=1).max()
 new=np.linalg.norm(p[ct].mean(1)-rest[t].mean(1)[cp],axis=1).max()
 cap=max(.0002,math.ceil(float(new)/1e-6)*1e-6+1e-6)
 assert cap==plan["predeclared_transport_envelope_m"]
 checks.append(dict(case=mass,old_current_centroid_max_m=float(old),native_rest_centroid_max_m=float(new),declared_envelope_m=cap,envelope_unchanged=True))
out={"scope":"offline envelope-coordinate correction; authoritative native mass mapper uses rest_xyz_m, original experiments retained; no physical admission","max_parent_current_rest_difference_m":float(np.linalg.norm(rest-current,axis=1).max()),"checks":checks}
(P/"mass-cut-witness-452/native-coordinate-envelope-parity.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
