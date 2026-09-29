from pathlib import Path
import subprocess,json,hashlib,os,time
root=Path(__file__).resolve().parent;repo=root.parents[1];dest=root/'abi2-checks';dest.mkdir(exist_ok=True)
cmd=[str(repo/'.venv-mujoco312/bin/python'),'-m','pytest','-q','tests/test_torso_anatomy_source_audit.py::test_native_lung_payload_rejects_unsupported_abi_layers_and_counts','--basetemp',str(dest/'work')]
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1',
 'NUMILAB_HUMAN_NATIVE_VISUAL_PROBE':str(root/'native/myosim-visual-probe'),
 'NUMILAB_HUMAN_NATIVE_BONE_PAYLOAD':str(repo/'Build/knee-parity-registration-20260929/bones.v6/payload/bodyparts3d-myosim-major-bones.nhbones'),
 'NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT':str(repo/'Build/myosim-fullbody'),
 'NUMILAB_HUMAN_MOTION_SOURCES':str(repo/'Sources'),
 'NUMILAB_HUMAN_MOTION_REPAIRED':str(repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json')}
(dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(dest/'executed-source.json').write_text(json.dumps({p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in ['tests/test_torso_anatomy_source_audit.py','src/numilab_human/model.py']},indent=2)+'\n')
start=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True)
for file,data in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-start)+'\n')]:
 (dest/file).write_text(data)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
