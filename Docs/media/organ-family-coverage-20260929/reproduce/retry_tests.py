from pathlib import Path
import subprocess,json,hashlib,os,time
root=Path(__file__).resolve().parent;repo=root.parents[1];dest=root/'tests.verified';dest.mkdir(exist_ok=True)
nodes=['tests/test_organ_family_geometry.py::test_declared_families_are_complete_and_spaces_are_distinct',
 'tests/test_organ_family_geometry.py::test_rehashed_native_data_cannot_forge_source_geometry[owner]',
 'tests/test_organ_family_geometry.py::test_rehashed_payload_and_receipt_cannot_forge_source_proof[owner]',
 'tests/test_organ_family_geometry.py::test_old_abi_native_geometry_is_unchanged',
 'tests/test_organ_family_geometry.py::test_same_size_changed_archive_fails_before_source_selection']
cmd=[str(repo/'.venv-mujoco312/bin/python'),'-m','pytest','-q',*nodes,'--basetemp',str(dest/'work')]
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1',
 'NUMILAB_HUMAN_ORGAN_FAMILY_EVIDENCE':str(root),'NUMILAB_HUMAN_LUNG_ENVELOPE_EVIDENCE':str(root.parent/'lung-envelope-20260929')}
(dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
files=json.loads((root/'tests.final/executed-source.json').read_text()).keys()
(dest/'executed-source.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
start=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True)
for file,data in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-start)+'\n')]:
 (dest/file).write_text(data)
print(r.stdout,r.stderr);raise SystemExit(r.returncode)
