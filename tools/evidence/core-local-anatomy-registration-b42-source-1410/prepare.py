import hashlib,json,sys,time
from pathlib import Path
R=Path("/Users/n/numi-human-final-integration-publish-1397");sys.path.insert(0,str(R/"src"))
D=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
B=D/"current-manifest-recovery-004"
p=D/"intercostal-parent-face-retirement-compose-002/index-only-respiratory-rebind-001/resting-thorax.nhanatomy"
print("payload exists",p.exists())
if not p.exists():
 d=json.loads((p.parent/"resting-anatomy-receipt.json").read_text());p=Path(d["payload"]["path"])
assert hashlib.sha256(p.read_bytes()).hexdigest()=="5613eb15de08a7dfe38f583f24c67c8cd4848f7f809c7e4783d27b3ccf326104"
out=B/"intercostal-core-local-registration-b42-source-001";out.mkdir(exist_ok=False)
from numilab_human.resting_anatomy import reemit_core_local_registration
proof=reemit_core_local_registration(p,D.parent/"input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json",B/"bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json",B/"recovered-artifact",out/"resting-thorax.nhanatomy")
rp=out/"source-preparation.json";rp.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
print(json.dumps({"payload_sha256":proof["output_payload_sha256"],"report_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),"row_count":proof["surface_count"]}))
