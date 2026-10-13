#!/usr/bin/env python3
"""Compile immutable vascular fields through the existing native parent ABI."""
from pathlib import Path
import hashlib, json, subprocess, sys
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
F=Path('/Users/numi/numi-human-fds132-m6-1350/attempt-347-shared-proximal-branch-field-002')
O=A/'forearm-vascular-carrier-cpu-048-001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fields=[F/f'stable-{s}-shared-proximal-field.npz' for s in (483,485,487,489)]
original=A/'axillary-parent-original-owner-control-1384-001/original-body91-field.npz'
constructor=A/'prepare_combined_anatomy_native_047.py'
oldverifier=A/'verify_vascular_carriers_035.py'
parent=A/'liver-island-native-027-002'
prep=A/'vascular-source-no-op-control-038-001/prepared'
pins={str(p.resolve()):sha(p) for p in fields+[original,constructor,oldverifier,Path(__file__),prep/'preparation.json',prep/'resting-thorax.nhanatomy']}
assert not O.exists()
O.mkdir()
z=np.load(original,allow_pickle=False)
data={k:z[k].copy() for k in z.files}
data['source_local_positions']=data['source_positions'].copy()
data['source_local_normals']=data['source_normals'].copy()
outfield=O/'unchanged-477-native-owner-local-schema.npz'
np.savez(outfield,**data)
for k in z.files: assert np.array_equal(np.load(outfield)[k],z[k])
fields=[outfield]+fields
args=[sys.executable,str(constructor),'--parent',str(parent),'--source-preparation',str(prep),'--output',str(O/'prepared')]
for p in fields: args+=['--field',str(p)]
with (O/'constructor.log').open('w') as log:
    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
text=oldverifier.read_text()
old="Load-time constructor test only. Candidate015 is rejected for geometry and self folds; source475 is unchanged body91 compatibility control. No accepted anatomy claim."
new="Load-time constructor test only. Fields347 remain unadmitted pending surrounding anatomy, late poses, and native replay. Source477 is the unchanged native body91 compatibility control. No accepted anatomy claim."
assert old in text
verifier=O/'verify_vascular_carriers_045.py'
verifier.write_text(text.replace(old,new))
args=[sys.executable,str(verifier),'--preparation',str(O/'prepared')]
for p in fields: args+=['--field',str(p)]
with (O/'verification.log').open('w') as log:
    subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
metadata=json.loads((O/'prepared/preparation.json').read_text())
assert metadata['source_geometry_unchanged'] is True
assert all(row['native_float32_gate_failures']==0 and row['max_um']<=2 for row in metadata['vessel_rest_rows'].values())
assert all(sha(p)==h for p,h in pins.items())
report={'status':'existing_native_carrier_constructor_pass', 'native_run_performed':False,'anatomical_admission':False,
'input_sha256':pins,'field_sha256':{str(p.resolve()):sha(p) for p in fields},
'normalized_original_field':{'coordinates_normals_and_original_keys_unchanged':True,'only_added_local_schema_aliases':True},
'verifier_sha256':sha(verifier),'constructor_output':str((O/'prepared').resolve()),
'proof_sha256':sha(O/'prepared/exact-carrier-map-proof.json'),
'qualification':'Serialization/rest preflight only. No all-target, native anatomical, or integrated physiology qualification.'}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'report':str(O/'report.json'),'proof':report['proof_sha256']}))
