import hashlib,json,sys,time,traceback
from pathlib import Path
R=Path("/Users/n/numi-human-final-integration-publish-1397")
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
B=D/"current-manifest-recovery-004"
OUT=B/"eight-parent-carrier-registration-reemission-001";OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
receipt=D/"forearm492-head-cord-vessel-seam-integrated-source-004/resting-anatomy-receipt.json"
old=D.parent/"input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json"
new=B/"bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
child=B/"accepted-registration-binding-retarget-006"
result=load(child/"result.json")
print("child keys",result.keys())
assert sha(receipt)=="98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d"
assert sha(new)=="31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
assert sha(old).startswith("b1b410ad")
ext=load(receipt)["passive_tissue_parent_bindings"]
assert ext["child_payload_sha256"]=="c9d5c8de77d398b047075cc222cdb2bfd7db08d37dbd1a52d0bc802664805ca2"
assert len(ext["source_payloads"])==8
record={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"preparing","started_unix":time.time(),"admission":False,"source_receipt_sha256":sha(receipt),"source_extension_sha256":hashlib.sha256(json.dumps(ext,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"previous_registration_sha256":sha(old),"registration_sha256":sha(new)}
def save():(OUT/"report.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
save()
sys.path.insert(0,str(R/"src"))
from numilab_human.passive_tissue_parent_carriers import reemit_registered_carrier
try:
 reports=[]
 for i,src in enumerate(ext["source_payloads"]):
  p=Path(src["path"]);assert sha(p)==src["sha256"]
  out=OUT/f"assets/parent-{i}.nhtissue"
  proof=reemit_registered_carrier(p,old,new,B/"recovered-artifact",out,coordinate_frame="core_inertial_local" if i in (4,6,7) else "registered_source_atlas")
  reports.append(proof)
  ext["source_payloads"][i]={"path":str(out),"sha256":sha(out)}
 # This remains a parent extension preparation, not a rebound anatomy receipt.
 ext["child_payload_sha256"]="cf7d35ae92f00773163f24b1e48d9f6762082f6bbcd6805d85b05ea00e1647b5"
 ext["provenance"].append({"operation":"registered_carrier_owner_reemission","report_path":str(OUT/"report.json"),"scope":"Registration source preparation only. Source terms/order/Float64 coefficients unchanged; native child-rest reconstruction and anatomy receipt rebind pending."})
 ep=OUT/"passive-tissue-parent-bindings.json";ep.write_text(json.dumps(ext,indent=2,sort_keys=True)+"\n")
 record.update(status="eight_parent_carriers_reemitted_native_reconstruction_pending",finished_unix=time.time(),carrier_reports=reports,parent_extension_path=str(ep),parent_extension_sha256=sha(ep))
 save()
 print(json.dumps({"status":record["status"],"extension_sha256":sha(ep),"changed_counts":[len(x["changed_binding_records"]) for x in reports]}))
except BaseException as e:
 record.update(status="failed",error=repr(e),traceback=traceback.format_exc());save();raise
