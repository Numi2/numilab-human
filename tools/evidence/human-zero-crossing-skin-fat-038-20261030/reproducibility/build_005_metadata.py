from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TASK=BASE/'quality-warning-connected-union-038-typed-metadata-005-001'
WRITER=BASE/'quality-warning-connected-union-038-typed-metadata-001/prepare_typed_connected_union.py'
PYTHON=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
role=sys.argv[1] if len(sys.argv)>1 else ''
if role not in ('I','D'):raise SystemExit('usage: build_005_metadata.py I|D')
role_pin=json.loads((BASE/'quality-warning-connected-union-038-typed-metadata-001/parent-037-pins.json').read_text())['roles'][role]
parent_typed=BASE/role_pin['typed_parent']['path']; parent_maps=BASE/role_pin['typed_parent_maps']['path']; parent_raw=BASE/role_pin['raw_parent']['path']
if role=='I':
 proposal=BASE/'quality-warning-connected-union-038/run005-source-precision-restoration-001/I-source-precision-restored.npz'
 expected='f4bc1b96aeef6991f25c1aa91740b92f95c61ad6e7bfd5324900c9107326c437'
 outdir=TASK/'I-run-002'
else:
 proposal=BASE/'quality-warning-connected-union-038/run005-source-dtype-restoration-002/D-source-dtype-restored.npz'
 expected='f729d6fd09c76b6c30048cbfa9a713a458157f07704910f01de3494b39497d0d'
 outdir=TASK/'D-run-001'
assert sha(proposal)==expected,(proposal,sha(proposal))
assert sha(parent_typed)==role_pin['typed_parent']['sha256']
assert sha(parent_maps)==role_pin['typed_parent_maps']['sha256']
assert sha(parent_raw)==role_pin['raw_parent']['sha256']
assert sha(BASE/'microedge-collar-014-saved-op-replay/receipt.json')=='a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01'
assert json.loads((TASK/'independent-005-source-correction-verification.json').read_text())['status']=='PASS_515_OF_515_FINALIZATION_AND_IMMEDIATE_037_PLUS_ULTIMATE_015_ROW_MAPS'
assert json.loads((TASK/'input-derivative-verification.json').read_text())['status']=='PASS_I_SOURCE_F64_AND_D_SOURCE_F32_DERIVATIVE_EQUIVALENCE'
geo=BASE/'quality-warning-connected-union-038/combined004O005ID/independent-verification-002/report.json'
assert sha(geo)=='73fc9db2f2a39734059424088005a6f3be6b3e2ffe8376bb0fc917784f19bd59'
assert json.loads(geo.read_text())['status']=='PASS_FIXED_COORDINATE_GEOMETRY_DELTA'
assert not outdir.exists(),f'refusing existing output {outdir}'
cmd=[str(PYTHON),str(WRITER),'--role',role,'--parent-typed',str(parent_typed),'--parent-maps',str(parent_maps),'--parent-raw',str(parent_raw),'--proposal',str(proposal),'--operation-dir',str(BASE/'quality-warning-connected-union-038/run-005'),'--composed-map',str(proposal),'--protected-receipt',str(BASE/'microedge-collar-014-saved-op-replay/receipt.json'),'--outdir',str(outdir),'--generation-label','038','--operation-glob','038-{role}-union-*.json','--operation-retired-face-key','retired_input_face_rows','--operation-new-point-key','new_faces_input_point_rows','--proposal-parent-face-map-key','face_output_to_parent_rows','--proposal-parent-point-map-key','point_output_to_parent_rows']
proc=subprocess.run(cmd,text=True,capture_output=True)
print(proc.stdout,end='')
if proc.returncode:
 print(proc.stderr,end='',file=sys.stderr);raise SystemExit(proc.returncode)
inv={'schema':'quality-warning-connected-union-005-typed-metadata-invocation.v1','role':role,'argv':cmd,'writer_sha256':sha(WRITER),'proposal_sha256':sha(proposal),'parent_typed_sha256':sha(parent_typed),'parent_maps_sha256':sha(parent_maps),'parent_raw_sha256':sha(parent_raw),'protected_receipt_sha256':sha(BASE/'microedge-collar-014-saved-op-replay/receipt.json'),'independent_correction_receipt_sha256':sha(TASK/'independent-005-source-correction-verification.json'),'input_derivative_verification_sha256':sha(TASK/'input-derivative-verification.json'),'geometry_delta_report_sha256':sha(geo),'writer_result':json.loads(proc.stdout.strip().splitlines()[-1])}
path=TASK/f'{role}-build-invocation.json';path.write_text(json.dumps(inv,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'PASS_BUILD_INVOCATION_PINNED','role':role,'invocation':str(path),'sha256':sha(path)}))
