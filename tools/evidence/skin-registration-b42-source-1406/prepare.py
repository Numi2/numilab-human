from pathlib import Path
import sys,json,hashlib
root=Path("/Users/n/numi-human-final-integration-publish-1397")
sys.path.insert(0,str(root/"src"))
from numilab_human.skin_source_payload_preflight import reemit_canonical_skin_registration
a=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
d=a/"full-anatomy-delivery-audit-1378"
payload=a/"back412-skin-reference-native-021-003/assets/bodyparts3d-myosim-skinned-shell.nhskin"
old=a/"input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json"
new=d/"current-manifest-recovery-004/bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
assert hashlib.sha256(payload.read_bytes()).hexdigest()=="b01028f163bde0d46c9605d730f2e104713a1c490be3c718e77ff9cebb865710"
assert hashlib.sha256(old.read_bytes()).hexdigest()=="b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"
assert hashlib.sha256(new.read_bytes()).hexdigest()=="31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
report=reemit_canonical_skin_registration(payload,old,new,
Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact"),
d/"skin-canonical-registration-b42-source-preparation-001")
print(json.dumps(report,indent=2))
