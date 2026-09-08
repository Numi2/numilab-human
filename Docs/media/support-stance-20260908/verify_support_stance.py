#!/usr/bin/env python3
"""Verify retained bounded evidence, keeping unqualified gates false."""
from pathlib import Path
import hashlib,json,math,re,shlex,struct
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for name,sha in receipt['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
native=json.loads((root/'qualified-certificate/receipt.json').read_text())
assert native['status']=='wrench_only_passed' and not native['failures']
assert native['runtime_revision']==receipt['native_revision'] and native['runtime_worktree_status']==''
assert not native['standing_qualified'] and not native['walking_qualified']
lines=(root/'qualified-certificate/stdout.log').read_text().splitlines()
summary=[l for l in lines if l.startswith('numi_human_whole_body_support_wrench=')];assert len(summary)==1
m=dict(token.split('=',1) for token in shlex.split(summary[0]));assert m['replay']=='bitwise' and m['internal_balanced']=='false'
forces=[float(m[f'contact_{i}_normal_force_n']) for i in range(10)]
assert all(math.isfinite(f) and f>=0 for f in forces)
assert abs(sum(forces)-float(m['expected_weight_n']))<1e-4
assert float(m['max_root_force_residual'])<1e-3
geometry=[l for l in lines if l.startswith('compiled_support_geometry=')];assert len(geometry)==2
for line in geometry:
 g=dict(token.split('=',1) for token in shlex.split(line))
 assert g['compiled_support_geometry']=='admissible'
 assert float(g['compiled_support_min_gap_m'])>=-float(g['compiled_support_gap_tolerance_m'])
 assert float(g['compiled_support_max_separated_force_n'])==0
assert 'placement omitted the exact source equality tangent' in (root/'tangent-mutant.log').read_text()
assert 'numi_human_static_support_test=passed' in (root/'tangent-restored.log').read_text()
assert 'stance source payload drift' in (root/'source-drift-negative.log').read_text()
assert 'Ran 185 tests' in (root/'human-python-final.log').read_text()
assert 'OK (skipped=7)' in (root/'human-python-final.log').read_text()
assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 2', (root/'ctest-final.log').read_text())
manifest=json.loads((root/'source-manifest.json').read_text())
def state(name):
 rows=[json.loads(l.split('=',1)[1]) for l in (root/name).read_text().splitlines() if l.startswith('stand_terminal_state=')]
 assert len(rows)==1
 s=rows[0];assert not s['root_assistance']
 for k,n in [('q',129),('initial_q',129),('v',128),('initial_v',128)]:
  assert len(s[k])==n and all(math.isfinite(x) for x in s[k])
 return s
runs=json.loads((root/'qualified-native.json').read_text());assert len(runs)==2
limits=[]
for run in runs:
 assert run['runtime_revision']==receipt['native_revision'] and run['exit_code']==0
 assert run['metrics']['stand_deterministic_replay']=='bitwise'
 assert run['metrics']['compiled_stand_balanced']=='false'
 s=state(run['label']+'.log')
 violations=[]
 for joint in manifest['core_tree']['source_joint_map']:
  if joint['source_limited']:
   lo,hi=[struct.unpack('<f',struct.pack('<f',x))[0] for x in joint['source_range']]
   value=s['q'][joint['core_q_index']]
   violations.append(max(lo-value,value-hi,0))
 limits.append(max(violations))
assert limits[-1]>1e-4
oracle=json.loads((root/'source-oracle.json').read_text())
assert oracle['version']=='3.12.0' and oracle['archive_source_files_verified']>10
assert oracle['witness_geometry_within_native_tolerance'] and oracle['external_wrench_within_1e_minus3']
assert not oracle['primitive_geometry_within_native_tolerance']
assert min(row['primitive_gap_m'] for row in oracle['contacts']) < -1e-6
assert min(row['witness_gap_m'] for row in oracle['contacts']) >= -1e-6
assert max(abs(x) for x in oracle['world_force_residual_n'])<1e-3
assert max(abs(x) for x in oracle['world_moment_residual_nm'])<1e-3
assert not any(receipt['qualification'][key] for key in ['full_primitive_geometry','internal_equilibrium','source_compliant_sustained_behavior','standing','walking','experimental_calibration','performance','full_release'])
print(json.dumps({'format':'numi-human-support-stance-verification-v1','witness_geometry':True,'static_gravity_wrench':True,'source_witness_oracle':True,'negative_controls_detected':True,'native_replay':True,'minimum_source_capsule_gap_m':oracle['minimum_primitive_gap_m'],'maximum_terminal_joint_limit_violation_rad':limits[-1],'full_primitive_geometry':False,'internal_equilibrium':False,'standing':False,'walking':False,'full_release':False},indent=2))
