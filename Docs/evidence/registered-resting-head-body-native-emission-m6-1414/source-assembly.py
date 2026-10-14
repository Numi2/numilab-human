#!/usr/bin/env python3
"""Join existing passive head repair with proof-bound rest-conforming body receipt."""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
ROOT=Path("/Users/n/numi-human-final-integration-publish-1397")
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
sys.path.insert(0,str(ROOT/"src"))
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows
from numilab_human.resting_anatomy import _copy_checked_sidecar
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--rest-receipt",type=Path,required=True);p.add_argument("--rest-sha",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 assert sha(a.rest_receipt)==a.rest_sha
 source=json.loads(a.rest_receipt.read_text());old=Path(source["payload"]["path"])
 ms=source["provenance"]["native_muscle_surfaces"];assert source["passive_tissue_parent_bindings"]["child_payload_sha256"]==ms["sha256"]
 assert sha(ms["payload_path"])==ms["sha256"] and sha(ms["manifest_path"])==ms["manifest_sha256"]
 assert sha(old)==source["payload"]["sha256"]=="1eb42c05a618d597797a411796e2066a67884531ad8946c1183ec8ed1167adfb"
 proofpath=D/"head-cord-abi5-compose-001/source-composition-report.json"
 assert sha(proofpath)=="e56da4a0885abd3770ad94f668767f3b821d7110ccc2c74620ffdeb9d4e959b8"
 proof=json.loads(proofpath.read_text());desc=proof["current_registration_reemission"];new=Path(desc["payload_path"])
 assert sha(new)==desc["payload_sha256"]=="1e6da4fa8c550334f31a6b74fda7aed6f26b0c436bff0b9ad168fe97d41cbb23"
 reportpath=Path(desc["registration_report_path"]);assert sha(reportpath)==desc["registration_report_sha256"]
 assert source["provenance"]["bodyparts_registration_sha256"]=="31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
 module=Path(sys.modules[rebind_passive_rows.__module__].__file__)
 assert sha(module)=="d8566e09d27c1b2827f03d10d8e4786bce8ececd91c709646154ea47e97e0221"
 doc,audit=rebind_passive_rows(source,old.read_bytes(),new.read_bytes(),changed_ids=(323,),output_path=str(new))
 doc["qualification"]="Unadmitted source preparation: registered body rest join with passive head006 and retained back412. Exact anatomy, motion and native qualification pending."
 record={"schema":"numi.human.upper-limb-full-owner-base-compile.v1","status":"registered_resting_head_body_source_prepared_native_admission_pending","admission":False,"source_inputs":{str(a.rest_receipt):a.rest_sha,str(proofpath):sha(proofpath),str(reportpath):sha(reportpath),str(new):sha(new),str(module):sha(module),str(Path(__file__)):sha(Path(__file__))},"existing_passive_rebind_owner":audit,"physical_mass_functional_parameters_and_parent_terms_unchanged":True,"source_only":True,"native_or_gpu_work_performed":False}
 doc["provenance"]["registered_resting_head_body_source_preparation"]=record
 # Match the functional owners exactly, restoring only the declared payload hash.
 functional=copy.deepcopy(doc["functional_bindings"]);functional["anatomy_payload_sha256"]=source["functional_bindings"]["anatomy_payload_sha256"];assert functional==source["functional_bindings"]
 assert doc["mass_geometry_accounting"]==source["mass_geometry_accounting"]
 assert doc["provenance"]["native_muscle_surfaces"]==source["provenance"]["native_muscle_surfaces"]
 ext=copy.deepcopy(doc["passive_tissue_parent_bindings"]);ext["anatomy_payload_sha256"]=source["passive_tissue_parent_bindings"]["anatomy_payload_sha256"];assert ext==source["passive_tissue_parent_bindings"]
 assert not a.out.exists();a.out.mkdir()
 common=doc["provenance"]["cardiac_geometry_binding"]["common_field"];sidecars=[]
 for owner in ("map","polynomials","domain_boxes"):
  descriptor=common[owner];path=Path(descriptor["path"])
  if not path.is_absolute():path=a.rest_receipt.parent/path
  destination=a.out/path.name;_copy_checked_sidecar(path,destination,descriptor["sha256"]);descriptor["path"]=destination.name
  sidecars.append({"owner":owner,"path":str(destination),"sha256":sha(destination)})
 out=a.out/"resting-anatomy-receipt.json";out.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+"\n")
 record.update(output_receipt_path=str(out),output_receipt_sha256=sha(out),sidecars_exact=sidecars,finished_unix=time.time())
 report=a.out/"report.json";report.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":record["status"],"receipt_path":str(out),"receipt_sha256":sha(out),"report_sha256":sha(report)}))
if __name__=="__main__":main()
