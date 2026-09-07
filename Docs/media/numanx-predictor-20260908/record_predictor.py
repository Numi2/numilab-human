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
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
repos={name:{'commit':command('git','-C',str(path),'rev-parse','HEAD'),
    'status':command('git','-C',str(path),'status','--porcelain=v1'), 'path':str(path)}
    for name,path in [('numi-lab',runtime),('numi-brain',brain)]}
assert all(not r['status'] for r in repos.values()),repos
assert repos['numi-lab']['commit']=='87f67836b1552ff7c110c4a6faacd32f85b0d58f'
native=(base/'predictor-native-final.log').read_text()
matter=(base/'predictor-matter-regressions.log').read_text()
swift=(base/'predictor-swift-e2e.log').read_text()
launch=json.loads((base/'predictor-swift-e2e-launch.json').read_text())
assert '100% tests passed out of 7' in native and 'negative_cases=12' in native
assert 'q_closure=0 v_closure=0 legacy_q_mismatch=9.89437e-06' in native
assert '100% tests passed out of 2' in matter
assert 'Executed 2 tests, with 0 failures' in swift and launch['returncode']==0
assets=[build/'bin'/name for name in (
    'metalrobo_numanx_human_matter_owner_probe','metalrobo_numanx_human_matter_v4_probe',
    'metalrobo_numanx_human_matter_candidate_probe','metalrobo_numanx_human_matter_adapter_probe',
    'metalrobo_numanx_matter_attachment_runtime_probe','metalrobo_numanx_fullbody_bridge_probe',
    'metalrobo_matter_physics_probe')]
assets += [build/'lib/libmetalrobo.dylib', build/'shaders/MetalRobo.metallib',
    build/'matter/shaders/NumiMatter.metallib', build/'CMakeCache.txt',
    binpath/'NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests',
    binpath/'mlx.metallib', brain/'Package.resolved']
assets += sorted(p for p in (binpath/'NumiBrain_NumiBrainMetal.bundle').rglob('*') if p.is_file())
assets += [Path(launch['environment'][name]) for name in (
    'NUMANX_FULLBODY_RIGID','NUMANX_FULLBODY_MUSCLE','NUMANX_FULLBODY_CONTACT',
    'NUMANX_FULLBODY_VISUAL_PACK','NUMANX_FULLBODY_VISION_PROFILE',
    'NUMANX_MATTER_MATERIAL','NUMANX_MATTER_WORLD_PACKAGE')]
assets += [base/name for name in (
    'predictor-native-final.log','predictor-matter-regressions.log',
    'predictor-swift-e2e.log','predictor-swift-e2e-launch.json',
    'predictor-native-tests.log','predictor-physical-diagnostic.log',
    'predictor-support-tests.log','predictor-convergence-tests.log',
    'run_predictor_swift.py','record_predictor.py')]
receipt={
    'schema':'numi.human.predictor-closure-qualification.v1',
    'recorded_at':datetime.now(timezone.utc).isoformat(), 'repositories':repos,
    'hardware':{'cpu':command('sysctl','-n','machdep.cpu.brand_string'),
        'memory_bytes':int(command('sysctl','-n','hw.memsize')), 'os':command('sw_vers')},
    'toolchain':{'swift':command('swift','--version'),'clang':command('clang','--version')},
    'verification':{'native_numanx_tests_passed':7,'native_matter_tests_passed':2,
        'swift_e2e_tests_passed':2,'negative_admission_cases':12,
        'candidate_probe_dofs':160,'candidate_probe_timestep_seconds':0.001,
        'candidate_position_max_abs_error':0,'candidate_velocity_max_abs_error':0,
        'legacy_position_negative_control_error_m':9.89437e-6,
        'support_tangent_fd_abs_error':1.90735e-5,
        'accepted_roots_per_e2e_case':8,'timestep_microseconds':100,
        'accepted_physical_seconds_per_e2e_case':0.0008,
        'rejected_candidate_replayed_exactly':True,
        'publication_tolerances_changed':False,
        'standing_qualified':False,'walking_qualified':False,
        'anatomical_tissue_qualified':False,'performance_qualified':False,
        'whole_human_complete':False},
    'assets':[asset(p) for p in assets],
    'limitations':[
        'Full-body joint equalities are not integrated in the coupled path.',
        'The authored Matter world contains three tiny pelvis-attached execution fixtures.',
        'Standing, walking, mass/active-force replacement and calibrated anatomy remain open.',
        'The 160-DoF zero-reaction probe checks candidate/Stand closure, not a behavioral outcome.',
        'Wall times include setup and assertions and are not performance qualification.']}
(base/'predictor-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'repositories':repos,'verification':receipt['verification'],
    'asset_count':len(assets)},indent=2))
