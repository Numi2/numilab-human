import hashlib,json,pathlib,subprocess,time
from fractions import Fraction
root=pathlib.Path('/Users/n/human-cardiac-material-attribution-20260912')
asset=pathlib.Path('/Users/n/human-cardiac-wall-anatomy-20260912/asset-final')
out=root/'cpp-evidence/full-source-attempt-001';out.mkdir(exist_ok=False)
paths=[asset/n for n in ['nodes.f64le','tetrahedra.u32le','labels.u32le']]+[
 root/'attribution/passive-material-classes.u32le',root/'attribution/manifest.json',
 root/'cardiac_material_attribution_check.cpp',root/'cardiac_material_attribution_check']
def info(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
before={str(p):info(p) for p in paths}
manifest=json.loads((root/'attribution/manifest.json').read_text())
argv=['/usr/bin/time','-l',str(root/'cardiac_material_attribution_check'),
 str(asset/'nodes.f64le'),str(asset/'tetrahedra.u32le'),str(asset/'labels.u32le'),
 str(root/'attribution/passive-material-classes.u32le'),'1470083']
start=time.monotonic()
with (out/'result.json').open('wb') as stdout,(out/'timing.log').open('wb') as stderr:
 proc=subprocess.run(argv,stdout=stdout,stderr=stderr)
elapsed=time.monotonic()-start
after={str(p):info(p) for p in paths}
record={'schema':'numilab-human.cardiac-attribution-cpp-execution.v1','argv':argv,
 'returncode':proc.returncode,'elapsed_seconds':elapsed,'before':before,'after':after,
 'inputs_unchanged':before==after,'physical_steps':0,'status':'fail'}
if proc.returncode==0:
 result=json.loads((out/'result.json').read_text())
 checks={'source_label_counts':result['counts_by_source_label']==manifest['counts_by_source_label'],
  'class_counts':all(result['counts_by_class'][str(i)]==manifest['counts_by_class'][str(i)] for i in range(5)),
  'unresolved_cells':result['unresolved_cells']==manifest['unresolved_cells'],
  'lv_cell_count':result['lv_geometric_mass']['cell_count']==manifest['lv_geometric_mass']['cell_count']}
 for name in ['volume_exact','mass_exact']:
  cpp=result['lv_geometric_mass'][name];py=manifest['lv_geometric_mass'][name]
  checks[name]=Fraction(int(cpp['numerator']),int(cpp['denominator']))==Fraction(int(py['numerator_hex'],16),int(py['denominator_hex'],16))
 record['independent_checks']=checks
 record['status']='pass' if before==after and all(checks.values()) else 'fail'
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'elapsed_seconds':elapsed,'checks':record.get('independent_checks',{})}))
raise SystemExit(0 if record['status']=='pass' else 1)
