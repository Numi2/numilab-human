"""Actual paired-retirement plus final singleton D row-map control."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
HERE=Path(__file__).resolve().parent
AUDIT=H/"combined-surface-diff-audit-270/audit_combined_current_surface_diff.py"
spec=importlib.util.spec_from_file_location("final_singleton_D_transition_control",AUDIT)
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def entry(p):return {"path":str(p.resolve()),"sha256":sha(p)}
root=H/"paired-inferred-envelope-resolution-reduction-293-root"
single=H/"native-apex-D-singleton-after-paired-293-root/run-001"
base=H/"current-volume-boundary-remesh-032/derived-D-pY-edge-split-001/candidate-D-pY-boundary-split.npz"
parent=H/"native-apex-D-ears-282-root/run-001/typed-001/candidate-D-typed-current.npz"
pair=root/"candidate-D-paired-resolution-reduced.npz"
final=single/"typed-001/candidate-D-typed-current.npz"
typed=single/"typed-001/receipt.json"
operation=root/"operation.json";receipt=root/"receipt.json";maps=root/"paired-retirement-maps.npz"
auth={"schema":"h706.paired-D-closed-component-retirement-auth.v1","operation":entry(operation),"receipt":entry(receipt),"maps":entry(maps),"parent_candidate":entry(parent),"paired_candidate":entry(pair),"final_typed_receipt":entry(typed)}
with np.load(base,allow_pickle=False) as z:b={k:np.array(z[k],copy=True) for k in z.files}
with np.load(final,allow_pickle=False) as z:c={k:np.array(z[k],copy=True) for k in z.files}
final_to_baseline,result=owner.validate_paired_d_retirement(auth,b,c,entry(final))
if len(final_to_baseline)!=len(c["points_world_m_f32"]):raise AssertionError("point map output domain mismatch")
bad=dict(auth);bad["maps"]={"path":str(maps.resolve()),"sha256":"0"*64}
try:owner.validate_paired_d_retirement(bad,b,c,entry(final))
except ValueError as e:negative={"rejected":True,"reason":str(e)}
else:raise AssertionError("tampered map digest unexpectedly accepted")
out={"schema":"h706.test-only-final-singleton-D-transition-control.v1","status":"PASS_FINAL_SINGLETON_FACE_ONLY_PARENT_AND_POINT_MAPS","physical_admission":False,"native_steps":0,"audit_owner":{"path":str(AUDIT),"sha256":sha(AUDIT)},"baseline_D":{"path":str(base.resolve()),"sha256":sha(base)},"parent_D282":entry(parent),"paired_candidate":entry(pair),"final_typed_candidate":entry(final),"final_typed_receipt":entry(typed),"composed_maps":{"path":str(single/"composed-row-maps-001.npz"),"sha256":"1f05fde555a310ac7d7ea16ee153a68df739a57448a1f57ed8b4e4330c77670a"},"operation":entry(operation),"receipt":entry(receipt),"paired_maps":entry(maps),"point_map_count":len(final_to_baseline),"retired_baseline_points":result["retired_baseline_point_rows"],"retired_baseline_faces":result["retired_baseline_face_rows"],"singleton_transition":result["final_singleton_transition"],"negative_control":negative}
report=HERE/"final-singleton-d-transition-control.json"
report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":out["status"],"final_points":len(final_to_baseline),"retired_points":len(out["retired_baseline_points"]),"retired_faces":len(out["retired_baseline_faces"]),"report":str(report),"report_sha256":sha(report)},sort_keys=True))
