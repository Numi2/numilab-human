from pathlib import Path
import sys,json,time,hashlib
E=Path('/Users/n/numi-human-resting-evidence-20261005')
D=E/'choroid-lung-disjoint-composition-918'
H=Path('/Users/n/numi-human-lung-source-publication-902')
sys.path.insert(0,str(H/'src'))
from numilab_human import right_choroid_laterality_repair as c
m=json.loads((E/'right-choroid-laterality-current-897-v3/right-choroid-current-897-candidate.manifest.json').read_text())
p=Path(m['parent_payload']['path']);r=Path(m['parent_receipt']['path'])
sources=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/anatomy-complete-20261005/inputs/Sources')
assert sources.is_dir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
module=H/'src/numilab_human/right_choroid_laterality_repair.py';before=sha(module);start=time.monotonic()
result=c.compose_after_lung_correction(p,r,sources,D/'unchanged-parent-regression',reference_payload=p,reference_receipt=r,expected_parent_sha256=m['parent_payload']['sha256'],expected_parent_receipt_sha256=m['parent_receipt']['sha256'])
actual=sha(D/'unchanged-parent-regression/resting-thorax.nhanatomy')
expected='91f27d73d3ffa2df59ba13efd512471de4123ec1472f20cd4a731a6205547b09'
assert actual==expected and sha(module)==before
report={'scope':'Unchanged-parent regression of disjoint lung/choroid composition; not a new lung candidate or anatomical admission','source_module_sha256':before,'native_payload_byte_identical_to_existing_897_v3':True,'payload_sha256':actual,'elapsed_seconds':time.monotonic()-start,'changed_parent_rows':result['disjoint_lung_composition']['changed_parent_rows']}
(D/'unchanged-parent-regression.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
