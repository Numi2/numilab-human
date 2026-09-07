from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess
base=Path('/Users/n/human-completion-20260907')
brain=Path('/Users/n/numi-brain-human-completion-20260907')
runtime=Path('/Users/n/MetalRobo-human-completion-20260907')
build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
binpath=brain/'.build/arm64-apple-macosx/debug'
def command(*args): return subprocess.check_output(args,text=True).strip()
def asset(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
repos={name:{'commit':command('git','-C',str(path),'rev-parse','HEAD'),
 'status':command('git','-C',str(path),'status','--porcelain=v1'), 'path':str(path)}
 for name,path in [('numi-lab',runtime),('numi-brain',brain)]}
assert all(not value['status'] for value in repos.values()),repos
assets=[build/'bin/metalrobo_numanx_fullbody_bridge_probe',build/'lib/libmetalrobo.dylib',
 build/'shaders/MetalRobo.metallib', build/'matter/shaders/NumiMatter.metallib',
 binpath/'numi-brain-gate-c',binpath/'NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests',
 binpath/'mlx.metallib',brain/'Package.resolved',build/'CMakeCache.txt',
 base/'input/myosim-fullbody-core-reference.nhrigid',base/'input/myosim-fullbody-muscle-reference.nhmyo',
 base/'input/myosim-fullbody-support-contact.nhcnt',base/'authored-world-fixture/authored-100us.nmatterpack',
 Path('/Users/n/Numan-X-neuron-v1/assets/numanx-head-vision-profile.v1.json'),
 Path('/Users/n/NumiHumanCurrent/anterior-thorax-smoke/myosim-fullbody-articulated-markers-muscle-driven-selected-actuators-source-support-contact.mrvpack')]
assets += sorted(p for p in (binpath/'NumiBrain_NumiBrainMetal.bundle').rglob('*') if p.is_file())
assets += [base/name for name in ['authored-world-native-tests-final.log','authored-world-swift-e2e-final.log',
 'authored-world-swift-e2e-launch.json','authored-world-cpu-tests-final.log','run_swift_e2e.py']]
launch=json.loads((base/'authored-world-swift-e2e-launch.json').read_text())
assert launch['returncode']==0
native=(base/'authored-world-native-tests-final.log').read_text()
assert '100% tests passed out of 2' in native and 'negative_cases=12' in native
swift=(base/'authored-world-swift-e2e-final.log').read_text()
assert 'Executed 2 tests, with 0 failures' in swift
cpu=(base/'authored-world-cpu-tests-final.log').read_text()
assert 'Executed 29 tests, with 0 failures' in cpu
receipt={'schema':'numi.human.authored-world-qualification.v1',
 'recorded_at':datetime.now(timezone.utc).isoformat(), 'repositories':repos,
 'hardware':{'cpu':command('sysctl','-n','machdep.cpu.brand_string'),
 'memory_bytes':int(command('sysctl','-n','hw.memsize')),'os':command('sw_vers')},
 'toolchain':{'swift':command('swift','--version'),'clang':command('clang','--version')},
 'verification':{'native_tests_passed':2,'negative_admission_cases':12,'swift_e2e_tests_passed':2,
 'cpu_tests_passed':29,'accepted_roots_per_e2e_case':8,'timestep_microseconds':100,
 'accepted_physical_seconds_per_e2e_case':0.0008,'authored_fixture_objects':3,
 'authored_fixture_attachments':12,'rejected_candidate_replayed_exactly':True,
 'standing_qualified':False,'walking_qualified':False,'anatomical_tissue_qualified':False,
 'performance_qualified':False,'whole_human_complete':False},
 'assets':[asset(p) for p in assets],
 'limitations':['The authored world contains tiny pelvis-attached execution fixtures.',
 'No joint-equality coupled operator, source mass partition, active-force replacement map or standing controller is supplied.',
 'Test wall time includes setup, MLX and assertions; it is not control latency or throughput.']}
(base/'authored-world-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'repositories':repos,'verification':receipt['verification'],'asset_count':len(assets)},indent=2))
