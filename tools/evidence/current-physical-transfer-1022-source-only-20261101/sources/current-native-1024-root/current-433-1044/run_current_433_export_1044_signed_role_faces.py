from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
NATIVE=ROOT/"current-native-1024-root"
TASK=NATIVE/"current-433-1044"
PY=Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")
WRAPPER=ROOT.parent/"run_geometry_serial.py"
ADAPTER=ROOT/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat/current1020-433-adapter-1044/export_refined_coarse_registered_volume_433_p1_anchor_1044_signed_role_faces.py"
ADAPTER_SHA="5c4a78a1310e466741341504db1800a23f728567a609367d51842dea63ccecbc"
CROSS=ROOT/"current-role-p1-support-938-1025-active-source-map/actual-001/217-handoff/skin-fat-combined-registered-surface-node-map.npz"
CROSS_SHA="4b98c0d437f9369b1ed07b2ef69138b3f0ccb5946631e4e0518e3bddfde986fa"
HANDOFF=CROSS.parent
CAPTURE=ROOT.parent/"native-physical-parent-mass-330/native-source-identity-control-003"
BUILDER=NATIVE/"current-433-1037"
CANDIDATE=BUILDER/"candidate-current-typed-volume.npz"
CANDIDATE_SHA="5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372"
NODEMAP=BUILDER/"candidate-node-map.csv"
NODEMAP_SHA="d71ede83bb69ea20304d561c658aa96050175bdf68688075777ea1f611e56bf9"
ANCESTRY=BUILDER/"inserted-node-ancestry.csv"
ANCESTRY_SHA="fbf7481c31f856cfc5f732bfce0b0363ab12141fdb18176e42a9786683b56a09"
BUILDER_RECEIPT=BUILDER/"receipt.json"
BUILDER_RECEIPT_SHA="a8a1f368c1859065bd175b11977c8edc0e708a6d9495b052cd88be552f9d7f5e"
P1=ROOT/"current-role-p1-support-938-1028-active-node-domain/actual-001/p1-anchor-export-937-active-node-domain"
CHILD=ROOT/"current-role-p1-support-938-1025-active-source-map/actual-001/217-handoff"
REG=ROOT/"current-native-1024-root/current-217-1036/actual-013/launch-inputs.json"
OUT=TASK/"volume-export"
CANONICAL_NODE=NATIVE/"current-433-1043/candidate-canonical-f32.node"
CANONICAL_NODE_SHA="8f47890b86b3a71ce247caf69eff6e72cf4a7080af28a287b8f4cd40efa27424"
CANONICAL_NODE_RECEIPT=NATIVE/"current-433-1043/canonical-node-receipt.json"
CANONICAL_NODE_RECEIPT_SHA="f60d61657f8e17ae982286258296bacffac0abc942ac8c20164b0b9ad5fddb3c"
SIGNED_FACE_AUDIT=TASK/"1022-role-surface-orientation-audit.json"
SIGNED_FACE_AUDIT_SHA="cad9e484357d9340f2ca43638a3e83874156a5693ddb9192f919e5c91ee8ebed"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p,h):
 p=Path(p)
 if not p.is_file():raise SystemExit(f"PIN_MISSING {p}")
 actual=sha(p)
 if actual!=h:raise SystemExit(f"PIN_MISMATCH {p} expected={h} actual={actual}")
for p,h in ((ADAPTER,ADAPTER_SHA),(CROSS,CROSS_SHA),(CANDIDATE,CANDIDATE_SHA),(NODEMAP,NODEMAP_SHA),(ANCESTRY,ANCESTRY_SHA),(BUILDER_RECEIPT,BUILDER_RECEIPT_SHA),(CANONICAL_NODE,CANONICAL_NODE_SHA),(CANONICAL_NODE_RECEIPT,CANONICAL_NODE_RECEIPT_SHA),(SIGNED_FACE_AUDIT,SIGNED_FACE_AUDIT_SHA)):
 pin(p,h)
launch217=json.loads(REG.read_text()); role_pins={}
for i,x in enumerate(launch217["argv"]):
 if x=="--current-typed-role":role_pins[launch217["argv"][i+1]]=(launch217["argv"][i+2],launch217["argv"][i+3])
if set(role_pins)!={"O","I","D"}:raise SystemExit("217 launcher role pin set mismatch")
for p,h in role_pins.values():pin(p,h)
for p,h in ((CHILD/"skin-fat-combined.node","f58cb0299e7a9c806ec141a051ca6b304fd05938d8168e2169655a4be708571e"),(P1/"child.labels-850.csv","c48c944a0f4080ecc3df3d6430bf6f8ec8abca00c90c8b990ffbfdd72dc2156d"),(P1/"child.labels.csv.source.csv","c168ec4d6558bb0a8d63c2957159bca58e38e988321ba21bea0d14855064b7af"),(P1/"p1-anchor.npz","bb2dff5a3efed948136d81b3f28cba0445cbf8fcd38f7e819d864b11b074da0f"),(P1/"receipt.json","9cefb9344c23ad21df230de340fd834d00e3387dba405e69ca43ebb8b78a9b41"),(P1/"p1-anchor-projection.csv","812c6f402c25f2a156c811d2354f599a73d08f70d3f516d935e0e6144872875d")):
 pin(p,h)
if OUT.exists():raise SystemExit(f"export output must be fresh: {OUT}")
argv=[str(PY),str(WRAPPER),str(PY),str(ADAPTER),"--current-surface-node-map",str(CROSS),CROSS_SHA,
 "--signed-role-face-audit",str(SIGNED_FACE_AUDIT),SIGNED_FACE_AUDIT_SHA,
 "--candidate-npz",str(CANDIDATE),"--candidate-node",str(CANONICAL_NODE),"--candidate-node-map",str(NODEMAP),"--inserted-node-ancestry",str(ANCESTRY),
 "--capture",str(CAPTURE),"--out",str(OUT),"--candidate-operational-reference","--current-typed-volume-candidate"]
for role in ("O","I","D"):
 p,h=role_pins[role];argv += ["--current-typed-role",role,p,h]
launch={"schema":"numi.h706.current-433-signed-face-volume-export-launch.v1","started_unix":time.time(),"adapter":{"path":str(ADAPTER),"sha256":ADAPTER_SHA},"candidate_npz":{"path":str(CANDIDATE),"sha256":CANDIDATE_SHA},"candidate_node_map":{"path":str(NODEMAP),"sha256":NODEMAP_SHA},"inserted_node_ancestry":{"path":str(ANCESTRY),"sha256":ANCESTRY_SHA},"canonical_candidate_node":{"path":str(CANONICAL_NODE),"sha256":CANONICAL_NODE_SHA,"receipt_path":str(CANONICAL_NODE_RECEIPT),"receipt_sha256":CANONICAL_NODE_RECEIPT_SHA},"signed_role_face_audit":{"path":str(SIGNED_FACE_AUDIT),"sha256":SIGNED_FACE_AUDIT_SHA},"builder_receipt":{"path":str(BUILDER_RECEIPT),"sha256":BUILDER_RECEIPT_SHA},"crosswalk":{"path":str(CROSS),"sha256":CROSS_SHA},"role_pins":{r:{"path":p,"sha256":h} for r,(p,h) in role_pins.items()},"argv":argv,"physical_admission":False,"native_steps":0}
(TASK/"launch-inputs.json").write_text(json.dumps(launch,sort_keys=True,indent=2)+"\n")
proc=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
(TASK/"stdout.log").write_text(proc.stdout);(TASK/"stderr.log").write_text(proc.stderr)
summary={"schema":"numi.h706.current-433-volume-export-execution.v1","returncode":proc.returncode,"ended_unix":time.time(),"stdout_sha256":sha(TASK/"stdout.log"),"stderr_sha256":sha(TASK/"stderr.log"),"output_dir":str(OUT),"outputs":{},"physical_admission":False,"native_steps":0}
if OUT.is_dir():
 for name in ("receipt.json","mesh.node","mesh.ele","candidate-current-typed-volume.npz","candidate-node-map.csv"):
  p=OUT/name
  if p.is_file():summary["outputs"][name]={"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
(TASK/"execution.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
print(json.dumps(summary,sort_keys=True))
if proc.returncode:print(proc.stderr,file=sys.stderr)
raise SystemExit(proc.returncode)
