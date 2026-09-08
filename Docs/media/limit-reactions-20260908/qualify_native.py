import hashlib, json, pathlib, subprocess, time
root = pathlib.Path('/Users/n/human-limit-reactions-20260908')
repo = pathlib.Path('/Users/n/MetalRobo-human-completion-20260907')
build = pathlib.Path('/Users/n/MetalRobo-human-completion-build-20260907')
revision = '117aa90b5c28a5448d845cbde45f3de85c481cb2'
def git(*args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
def fingerprint(path):
    path = pathlib.Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
assert git('rev-parse', 'HEAD') == revision and not git('status', '--porcelain')
base = json.loads((root/'candidate-command.json').read_text())
record = {'schema': 'numi.human.offline-native-qualification.v1', 'native_commit': revision,
          'working_tree_clean': True, 'physics_owner': 'offline C++ compiler; Metal runtime unchanged',
          'performance_qualification': False,
          'host': {'architecture': subprocess.check_output(['uname','-m'], text=True).strip(),
                   'processor': subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True).strip(),
                   'os': subprocess.check_output(['sw_vers'], text=True).strip()},
          'binaries': [fingerprint(base[0]), fingerprint(build/'lib/libmetalrobo.dylib')],
          'inputs': [fingerprint(base[i]) for i in (1,2,3)] + [fingerprint(base[base.index(flag)+1]) for flag in ('--support-contact-payload','--joint-equality-payload')], 'runs': []}
for sweeps in (240,1024):
    assert git('rev-parse','HEAD') == revision and not git('status','--porcelain')
    label = f'qualified-{sweeps}'
    command = base.copy(); command[4] = str(root/label)
    command += ['--whole-body-activation-sweeps',str(sweeps)]
    (root/f'{label}-command.json').write_text(json.dumps(command,indent=2)+'\n')
    started=time.monotonic()
    with (root/f'{label}-stdout.log').open('w') as out, (root/f'{label}-stderr.log').open('w') as err:
        result=subprocess.run(command,stdout=out,stderr=err)
    run={'label':label, 'returncode':result.returncode, 'elapsed_seconds':time.monotonic()-started,
         'sweeps':sweeps,'stdout':fingerprint(root/f'{label}-stdout.log'), 'stderr':fingerprint(root/f'{label}-stderr.log')}
    record['runs'].append(run)
    (root/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'label':label,'returncode':result.returncode,'elapsed_seconds':run['elapsed_seconds']}),flush=True)
    if result.returncode != 0: raise SystemExit(result.returncode)
assert git('rev-parse','HEAD') == revision and not git('status','--porcelain')
record['working_tree_clean_after'] = True
(root/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
