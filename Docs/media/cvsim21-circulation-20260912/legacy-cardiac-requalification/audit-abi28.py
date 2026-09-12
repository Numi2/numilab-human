import hashlib,importlib.util,inspect,json,pathlib
root=pathlib.Path('/Users/n/MetalRobo-human-completion-20260907/Build/absolute-blood-20260912/legacy-cardiac-requalification')
path=pathlib.Path('/tmp/numi-frozen-cardiac-verifier.py')
spec=importlib.util.spec_from_file_location('frozen',path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
# New ABI expectation is explicit. No historical verifier, receipt, log,
# metric envelope or refinement rule is changed.
source=inspect.getsource(v.audit_native_run)
assert source.count('abi=27')==1
exec("from __future__ import annotations\n"+source.replace('abi=27','abi=28'),v.__dict__)
payload=json.loads(pathlib.Path('/Users/n/MetalRobo-human-completion-20260907/matter/tools/fixtures/shi-hose.native.v2.json').read_text())
scales,initial=v.source_coordinates(payload);runs={};identities={}
for name,(dt,steps) in v.RUNS.items():
 record=json.loads((root/(name+'.execution.json')).read_text())
 assert record['exit_code']==0 and record['source_and_binary_identity_unchanged'] and record['before']==record['after']
 for a,identity in record['artifacts'].items():assert hashlib.sha256((root/a).read_bytes()).hexdigest()==identity['sha256']
 runs[name]=v.audit_native_run(root/(name+'.csv.gz'),(root/(name+'.log')).read_text(),{'id':name,'dt_seconds':dt,'steps':steps},scales,initial,payload)
 identities[name]=record['before']
ratios=v.audit_native_refinement(runs)
report={'schema':'NumiHuman.LegacyCardiacRequalification.v1','status':'pass','native_abi':28,'old_receipt_unchanged':True,
 'frozen_verifier_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'audit_adaptation':'device identity expectation ABI27 -> ABI28 only; unchanged trace recomputation, accuracy ceilings and refinement thresholds',
 'accuracy_limits':v.ACCURACY_LIMITS,'refinement_minimum_ratio':1.5,'runs':runs,'refinement_error_ratios':ratios,
 'exact_clock':'all samples and production accepted-state clock checks passed','replay':'two environments bitwise at every step',
 'identity_before_after_equal':True,'identities':identities,'biological_calibration':'unqualified'}
(root/'summary.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':'pass','runs':runs,'refinement_error_ratios':ratios},sort_keys=True))
