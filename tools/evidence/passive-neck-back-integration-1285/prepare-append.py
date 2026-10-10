from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import passive_attachment_composition as pc
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
P=A/"twenty-two-surface-native-composition-002"
S=R/"passive-neck-back-coverage-1281/candidate-005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
O=A/"neck-back-append-preparation-002"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=P/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
assert sha(T)=="decb87319f01e7315c26c4d527a40e3fb0c2e434cbed1d154b742ec925978e3b"
assert sha(S)=="08c534229c322391ce5c7fb972810ace13bc07cf9d95ff18825561ac2f792ada"
manifest=json.loads(S.with_suffix(".manifest.json").read_text())
assert sha(S.with_suffix(".manifest.json"))=="ec3c9efeb55b827eca485480a5e336a2b41fe3672f1ad094ad1438b5b7e2b890"
assert manifest["status"]=="passive_registered_support_visual_input_not_collision_or_physics"
proof=pc.append_passive_surfaces(T,S,O,stable_ids=tuple(range(151,159)))
bound=pc.bind_anatomy_receipt(P/"resting-anatomy-receipt.json",O/T.name,O/"resting-anatomy-receipt.json")
old=json.loads((P/"resting-anatomy-receipt.json").read_text())
assert bound["mass_geometry_accounting"]==old["mass_geometry_accounting"]
report={"scope":"Unadmitted 158-row asset integration preparation. Pose diagnostics retain source/self/external defects. No native launch or anatomy qualification.",
 "composer_source_sha256":sha(Path(pc.__file__)),"script_sha256":sha(Path(__file__)),
 "proof":proof,"source_receipt_sha256":sha(P/"resting-anatomy-receipt.json"),
 "receipt_sha256":sha(O/"resting-anatomy-receipt.json"),
 "mass_geometry_accounting_unchanged":True,"new_surface_count":158}
(O/"integration-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
