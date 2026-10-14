import copy,hashlib,json,sys,tempfile,time
from pathlib import Path
R=Path("/Users/n/numi-human-final-integration-publish-1397");sys.path.insert(0,str(R/"src"))
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
D=A/"full-anatomy-delivery-audit-1378";B=D/"current-manifest-recovery-004"
OUT=B/"registered-resting-scene-source-preparation-002"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
inputs={
 "source_receipt":(D/"intercostal-parent-face-retirement-compose-002/index-only-respiratory-rebind-001/resting-anatomy-receipt.json","ac539607f2c1ddef19ada42bd248e4403e40a7d04f2ce6ce623cce21a286926c"),
 "body_scene":(A/"back412-skin-reference-native-021-003/resting-supine-scene.manifest.json","d4711f4bd686703b4d8cfc707dbd2228cf0150bcdf37f09342dc528e4b971382"),
 "anatomy_proof":(B/"intercostal-core-local-registration-b42-source-001/source-preparation.json","c565676a85241b10747cf3a408e9ded7797dc049fb1776efaa734f063f0d0de5"),
 "muscle_preflight":(B/"accepted-registration-binding-retarget-006/preflight.json","ee038df0ed00162e9a3388a251ce41a3ba31f1eaa023b89228db5dd6c19546c0"),
 "muscle_result":(B/"accepted-registration-binding-retarget-006/result.json","88952ef42ec411ab3d90e567404bae8fc3366b2b943b27cefd477adc4284ab21"),
 "carrier_proof":(B/"eight-parent-carrier-registration-reemission-001/verification-002.json","e2c33644c5b4d6ea2f7c825eb55daa12b33320e690d5308a991a6b80ca581c1c"),
 "parent_extension":(B/"eight-parent-carrier-registration-reemission-001/passive-tissue-parent-bindings.json","994514d561c7bea8e1e4c63168c3fbf6d3bbe93dbfb5882da855a18022ca1f0e"),
 "bone_proof":(B/"bilateral-b42-native-bone-owner-001/regional-row-byte-proof-002.json","49c70d36068b162e90175636c1e0b2e4f92723833e314d86b86a2aecb1c4065f"),
 "tendon_proof":(B/"bilateral-b42-native-tendon-owner-001/source-endpoint-invariants-001.json","a72bd673f2c86325e1a9b760619260eac2b42997c940e07b02d2fe4089f55162"),
}
for label,(p,h) in inputs.items():assert p.is_file() and sha(p)==h,label
assert not OUT.exists()
src=read(inputs["source_receipt"][0]);scene=read(inputs["body_scene"][0])
np=read(inputs["anatomy_proof"][0]);mp=read(inputs["muscle_preflight"][0]);mr=read(inputs["muscle_result"][0])
cp=read(inputs["carrier_proof"][0]);ext=read(inputs["parent_extension"][0])
pins=mp["pins"]; op=mr["operation"]
for label, record in pins.items():
 if label=="composer_source":
  import subprocess
  historical=subprocess.check_output(["git","-C",str(R),"show","69e54d57:src/numilab_human/passive_attachment_composition.py"])
  assert hashlib.sha256(historical).hexdigest()==record["sha256"]
  retained=B/"accepted-registration-retarget-owner-69e54d57.py"
  if not retained.exists():retained.write_bytes(historical)
  assert sha(retained)==record["sha256"]
 else:assert sha(record["path"])==record["sha256"],record["path"]
p=Path(mr["output_payload_path"]);manifest_path=Path(mr["output_manifest_path"])
assert sha(p)==mr["output_payload_sha256"]=="cf7d35ae92f00773163f24b1e48d9f6762082f6bbcd6805d85b05ea00e1647b5"
assert sha(manifest_path)==mr["output_manifest_sha256"]=="7617a9873b32b85e92cbaa9eb010851993eec1c03ae99aa4916ba34f7f3f565d"
assert ext["child_payload_sha256"]==mr["output_payload_sha256"]
for key in ("surfaces","anatomy_surfaces"):assert ext[key]==src["passive_tissue_parent_bindings"][key]
for i,proof in enumerate(cp["carrier_reports"]):
 assert sha(proof["source_payload_path"])==proof["source_payload_sha256"]==src["passive_tissue_parent_bindings"]["source_payloads"][i]["sha256"]
 assert sha(proof["output_payload_path"])==proof["output_payload_sha256"]==ext["source_payloads"][i]["sha256"]
assert cp["actual_rigid_payload_sha256"]==scene["source"]["rigid"]["sha256"]
assert cp["registration_sha256"]==np["registration_sha256"]==pins["source_registration"]["sha256"]
# Reconstruct the exact accepted-child operation using the existing owner.
proof={"schema":"numi.human.passive-registration-source-evolution-proof.v1","compatibility_report_path":pins["compatibility_report"]["path"],"compatibility_report_sha256":pins["compatibility_report"]["sha256"],"accepted_surface_map_path":pins["old_surface_map"]["path"],"owner_surface_map_path":pins["current_surface_map"]["path"],"accepted_binding_metadata_recoveries":[mp["recovery"]]}
from numilab_human.passive_attachment_composition import rebind_registration_bindings
with tempfile.TemporaryDirectory(prefix="registered-scene-replay-",dir=B) as tmp:
 emitted=rebind_registration_bindings(Path(op["accepted_payload_path"]),Path(op["owner_base_payload_path"]),Path(op["registration_path"]),Path(tmp)/"replay",tuple(op["selected_stable_ids"]),source_evolution_proof=proof)
 assert sha(emitted["output_payload_path"])==sha(p)
 assert read(emitted["output_manifest_path"])==read(manifest_path)
# Validate the anatomy owner result and source-body records independently.
ap=Path(np["output_payload_path"]);old_ap=Path(np["source_payload_path"])
assert sha(ap)==np["output_payload_sha256"]=="1eb42c05a618d597797a411796e2066a67884531ad8946c1183ec8ed1167adfb"
assert sha(old_ap)==src["payload"]["sha256"]==np["source_payload_sha256"]
from numilab_human.resting_anatomy import HEADER,_copy_checked_sidecar
assert ap.read_bytes()[HEADER.size:]==old_ap.read_bytes()[HEADER.size:]
new_fp=cp["registration_sha256"][:8];assert new_fp=="31c77660"
assert HEADER.unpack_from(ap.read_bytes())[5]==int(new_fp,16)
bone=B/"bilateral-b42-native-bone-owner-001/owner-base/bodyparts3d-myosim-major-bones.nhbones"
tendon=B/"bilateral-b42-native-tendon-owner-001/owner-base/numi-human-tendon-attachments.nhtendon"
skin=D/"skin-canonical-registration-b42-source-preparation-001/bodyparts3d-myosim-skinned-shell.nhskin"
assert sha(bone)=="4d74bc786176c1d4479e3be8ebb7c97797a0d55feca687f83c89136e7c6635b0"
assert sha(tendon)=="09ff6ae15c22a2b8e5639b9683c1471722fd77af7742ba5920ad7328987595a9"
assert sha(skin)=="b949ff00f5a445d3889a0f3e530e10fb276ca44fe152945bf321199f7ce73fc8"
old_skin=Path(scene["source"]["skin"]["path"]);assert sha(old_skin)==src["mass_geometry_accounting"]["skin_payload_sha256"]
before,after=old_skin.read_bytes(),skin.read_bytes()
assert before[:24]+before[28:]==after[:24]+after[28:]
assert len(before)==len(after)
# Freeze complete JSON output before creating the receipt directory.
doc=copy.deepcopy(src)
new_sha=sha(ap)
doc["payload"].update(path=str(ap),sha256=new_sha,registration_fingerprint32=new_fp)
doc["functional_bindings"].update(anatomy_payload_sha256=new_sha,bones_payload_sha256=sha(bone))
doc["provenance"].update(bones_payload=str(bone),bones_payload_sha256=sha(bone),bodyparts_registration_sha256=cp["registration_sha256"])
ms=doc["provenance"]["native_muscle_surfaces"]
ms.update(payload_path=str(p),sha256=sha(p),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),registration_fingerprint32=new_fp)
doc["passive_tissue_parent_bindings"]=copy.deepcopy(ext)
doc["passive_tissue_parent_bindings"]["anatomy_payload_sha256"]=new_sha
common=doc["provenance"]["cardiac_geometry_binding"]["common_field"]
common["anatomy_payload_sha256"]=new_sha
mass=doc["mass_geometry_accounting"];mass.update(skin_payload_path=str(skin),skin_payload_sha256=sha(skin))
scene["source"]["skin"].update(path=str(skin),sha256=sha(skin),registration_fingerprint32=new_fp)
scene["qualification"]="Source-only coherent registration preparation. Native initialization, full anatomy and motion acceptance remain pending; original real flat-bed contact retained."
doc["qualification"]="Source-only coherent registration preparation. Forearm/head/vascular clearance and native qualification pending; no anatomical admission."
record={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"registered_resting_scene_source_prepared_native_admission_pending","admission":False,"source_inputs":{k:{"path":str(p),"sha256":h} for k,(p,h) in inputs.items()},"owner_binding_replay_exact":True,"all_nha_core_local_row_bytes_exact":True,"skin_geometry_and_bindings_exact":True,"parent_terms_and_float64_coefficients_exact":True,"physical_mass_and_functional_parameter_values_unchanged":True,"tendon_payload_path":str(tendon),"tendon_payload_sha256":sha(tendon),"pending":"Candidate006 head row323 not yet merged; this scene is for authoritative initialization diagnostic, not final acceptance."}
doc["provenance"]["registered_resting_scene_source_preparation"]={"report_path":str(OUT/"report.json"),**record}
# Assert only declared functional hash identities changed.
f=copy.deepcopy(doc["functional_bindings"]);f.update(anatomy_payload_sha256=src["functional_bindings"]["anatomy_payload_sha256"],bones_payload_sha256=src["functional_bindings"]["bones_payload_sha256"]);assert f==src["functional_bindings"]
m=copy.deepcopy(mass);m.update(skin_payload_path=src["mass_geometry_accounting"]["skin_payload_path"],skin_payload_sha256=src["mass_geometry_accounting"]["skin_payload_sha256"]);assert m==src["mass_geometry_accounting"]
assert scene["bed"]==read(inputs["body_scene"][0])["bed"] and scene["outputs"]==read(inputs["body_scene"][0])["outputs"]
for obj in (doc,scene,record):json.dumps(obj,allow_nan=False)
OUT.mkdir()
sidecars=[]
for owner in ("map","polynomials","domain_boxes"):
 desc=common[owner];path=Path(desc["path"])
 if not path.is_absolute():path=inputs["source_receipt"][0].parent/path
 target=OUT/path.name;_copy_checked_sidecar(path,target,desc["sha256"])
 sidecars.append({"owner":owner,"sha256":sha(target),"path":str(target)})
 desc["path"]=target.name
rp=OUT/"resting-anatomy-receipt.json";sp=OUT/"resting-supine-scene.manifest.json"
rp.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n");sp.write_text(json.dumps(scene,indent=2,sort_keys=True,allow_nan=False)+"\n")
record.update(anatomy_receipt_path=str(rp),anatomy_receipt_sha256=sha(rp),body_scene_path=str(sp),body_scene_sha256=sha(sp),sidecars_exact=sidecars,finished_unix=time.time())
(OUT/"report.json").write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"status":record["status"],"anatomy_receipt_sha256":sha(rp),"body_scene_sha256":sha(sp),"source_report_sha256":sha(OUT/"report.json")}))
