import json,hashlib,sys,struct,time
from pathlib import Path
R=Path("/Users/n/numi-human-final-integration-publish-1397")
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
B=D/"current-manifest-recovery-004";OUT=B/"eight-parent-carrier-registration-reemission-001"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=D/"forearm492-head-cord-vessel-seam-integrated-source-004/resting-anatomy-receipt.json"
old=D.parent/"input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json"
new=B/"bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
ext=json.loads(receipt.read_text())["passive_tissue_parent_bindings"]
assert isinstance(ext["provenance"],dict) and len(ext["source_payloads"])==8
assert sha(receipt)=="98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d"
assert sha(new)=="31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
json.dumps(ext,sort_keys=True)
sys.path.insert(0,str(R/"src"))
from numilab_human.model import _bodyparts_visual_local_pose,_bodyparts_runtime_bindings
from numilab_human.passive_tissue_parent_carriers import _retarget_carrier_binding_bytes
od=json.loads(old.read_text());nd=json.loads(new.read_text())
oc,ob=_bodyparts_runtime_bindings(od,B/"recovered-artifact");nc,nb=_bodyparts_runtime_bindings(nd,B/"recovered-artifact")
assert oc["rigid"]==nc["rigid"] and ob==nb
def bm(doc):
 out={}
 for a in doc["anchors"]:
  core=a["target"]["core_body_index"]
  tr,qu,sc=_bodyparts_visual_local_pose(a["registration"]["source_obj_mm_to_core_inertial_body_m"],"carrier verification")
  val=struct.pack("<I8f",core,*tr,*qu,sc)
  assert core not in out or out[core]==val;out[core]=val
 return out
oldmap,newmap=bm(od),bm(nd);proofs=[]
for i,src in enumerate(ext["source_payloads"]):
 p=Path(src["path"]);assert sha(p)==src["sha256"]
 raw,proof=_retarget_carrier_binding_bytes(p.read_bytes(),previous_fingerprint=int(sha(old)[:8],16),fingerprint=int(sha(new)[:8],16),source_digest=bytes.fromhex(nc["source_archive_sha256"]),previous_bindings=oldmap,bindings=newmap,coordinate_frame="core_inertial_local" if i in (4,6,7) else "registered_source_atlas")
 out=OUT/f"assets/parent-{i}.nhtissue"
 assert out.read_bytes()==raw
 proof.update(source_payload_path=str(p),source_payload_sha256=sha(p),output_payload_path=str(out),output_payload_sha256=sha(out))
 proofs.append(proof);ext["source_payloads"][i]={"path":str(out),"sha256":sha(out)}
ext["child_payload_sha256"]="cf7d35ae92f00773163f24b1e48d9f6762082f6bbcd6805d85b05ea00e1647b5"
ext["provenance"]["registered_carrier_owner_reemission"]={"report_path":str(OUT/"verification-002.json"),"scope":"Source preparation only; source terms/order/Float64 coefficients unchanged. Anatomy receipt rebind and native child-rest reconstruction pending."}
ep=OUT/"passive-tissue-parent-bindings.json";assert not ep.exists();ep.write_text(json.dumps(ext,indent=2,sort_keys=True)+"\n")
r={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"eight_parent_carriers_verified_source_preparation_only","admission":False,"source_receipt_sha256":sha(receipt),"previous_registration_sha256":sha(old),"registration_sha256":sha(new),"owner_source_sha256":sha(R/"src/numilab_human/passive_tissue_parent_carriers.py"),"actual_rigid_payload_sha256":nc["rigid"]["sha256"],"carrier_reports":proofs,"parent_extension_path":str(ep),"parent_extension_sha256":sha(ep),"preparation_erratum":"Attempt001 re-emitted all8payloads but finalizer incorrectly treated provenance dict as list. Failed report retained; this successor verifies existing payloads against exact owner result without re-emission.","finished_unix":time.time()}
rp=OUT/"verification-002.json";rp.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
print(json.dumps({"extension_sha256":sha(ep),"report_sha256":sha(rp),"changed":[len(x["changed_binding_records"]) for x in proofs],"carrier_sha256":[x["output_payload_sha256"] for x in proofs]}))
