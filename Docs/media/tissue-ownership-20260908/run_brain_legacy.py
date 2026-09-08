from pathlib import Path
import hashlib, json, os, signal, subprocess, time
base=Path('/Users/n/human-completion-20260907')
out=Path('/Users/n/numi-human-tissue-ownership-20260908')
root=Path('/Users/n/MetalRobo-human-completion-20260907')
build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
brain=Path('/Users/n/numi-brain-human-completion-20260907')
def fnv(path):
    value=0xcbf29ce484222325
    for byte in path.read_bytes(): value=((value^byte)*0x100000001b3)&0xffffffffffffffff
    return f'{value:016x}'
identity=json.loads((base/'authored-world-fixture/authored-100us.json').read_text())
world=json.loads((out/'mass-compilation-metal-final.json').read_text())
equality=base/'input/myosim-fullbody-joint-equalities-source-compliance.nheq'
selected={
 'NUMANX_METALROBO_LIBRARY':str(build/'lib/libmetalrobo.dylib'),
 'NUMANX_FULLBODY_RIGID':str(base/'input/myosim-fullbody-core-reference.nhrigid'),
 'NUMANX_FULLBODY_MUSCLE':str(base/'input/myosim-fullbody-muscle-reference.nhmyo'),
 'NUMANX_FULLBODY_CONTACT':str(base/'input/myosim-fullbody-support-contact.nhcnt'),
 'NUMANX_METALROBO_METALLIB':str(build/'shaders/MetalRobo.metallib'),
 'NUMANX_MATTER_METALLIB':str(build/'matter/shaders/NumiMatter.metallib'),
 'NUMANX_FULLBODY_VISUAL_PACK':'/Users/n/NumiHumanCurrent/anterior-thorax-smoke/myosim-fullbody-articulated-markers-muscle-driven-selected-actuators-source-support-contact.mrvpack',
 'NUMANX_FULLBODY_VISION_PROFILE':'/Users/n/Numan-X-neuron-v1/assets/numanx-head-vision-profile.v1.json',
 'NUMANX_MATTER_MATERIAL':str(root/'matter/materials/silicone.nmatter'),
 'NUMANX_MATTER_WORLD_PACKAGE':str(out/'costal-rebased.nmatterpack'),
 'NUMANX_HUMAN_SOURCE_FP':identity['human_source_fp'],
 'NUMANX_MATTER_WORLD_FP':f"{world['matter_world_fingerprint']:016x}",
 'NUMANX_JOINT_EQUALITIES':str(equality),
 'NUMANX_JOINT_EQUALITY_FP':fnv(equality),
 'NUMANX_COSTAL_CARTILAGE':str(out/'bodyparts3d-costal-cartilage.nhcartilage'),
 'NUMANX_COSTAL_BINDING':str(out/'costal-tissue.nhtbind'),
 'NUMANX_COSTAL_BINDING_FP':fnv(out/'costal-tissue.nhtbind'),
 'NUMANX_CONSTRAINED_HUMAN_SOURCE_FP':'4b26cde26f8a228a',
 'NUMANX_GATE_B_EVIDENCE':'1',
 'MTL_DEBUG_LAYER':'1',
 'MRNX_RUNTIME_DIAGNOSTICS':'1',
}
for key in ('NUMANX_COSTAL_CARTILAGE','NUMANX_COSTAL_BINDING','NUMANX_COSTAL_BINDING_FP','NUMANX_CONSTRAINED_HUMAN_SOURCE_FP'):
    selected.pop(key)
selected['NUMANX_MATTER_WORLD_PACKAGE']=str(base/'authored-world-fixture/authored-100us.nmatterpack')
selected['NUMANX_MATTER_WORLD_FP']=identity['matter_world_fp']
command=['swift','test','-c','release','--skip-build','--filter','testRealFullBodyBrainProposalApplyAndJointPublication|testAuthoredMatterBrainProposalApplyAndJointPublication']
environment=dict(os.environ,**selected)
environment['PATH']='/opt/homebrew/bin:'+environment.get('PATH','')
source_state={name:{'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True)} for name,path in [('numi-brain',brain),('numi-lab',root)]}
extra_artifacts={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
    'mlx_metallib':brain/'.build/arm64-apple-macosx/release/mlx.metallib',
    'test_executable':brain/'.build/arm64-apple-macosx/release/NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests',
    'package_resolved':brain/'Package.resolved',
}.items()}
started=time.monotonic()
with (out/'brain-legacy-e2e.log').open('w') as log:
    process=subprocess.Popen(command,cwd=brain,env=environment,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    try: code=process.wait(timeout=480)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
        code=124
(out/'brain-legacy-e2e-launch.json').write_text(json.dumps({'command':command,'cwd':str(brain),'environment':selected,'elapsed_seconds':time.monotonic()-started,'returncode':code,'source_state':source_state,'extra_artifact_sha256':extra_artifacts,'artifact_sha256':{key:hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in selected.items() if Path(value).is_file()}},indent=2)+'\n')
raise SystemExit(code)
