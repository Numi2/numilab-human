"""Captured offline attribution and byte-preserving repeat; zero physical steps."""
import datetime, hashlib, json, os, resource, subprocess, time
from pathlib import Path
BASE=Path('/Users/n/human-cardiac-material-attribution-20260912')
SOURCE=BASE/'source'
ASSET=Path('/Users/n/human-cardiac-wall-anatomy-20260912/asset-final')
PYTHON='/Users/n/human-cardiac-partition-20260912/venv/bin/python'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while data:=stream.read(1<<20):h.update(data)
    return h.hexdigest()
def sources():
    return {str(p.relative_to(SOURCE)):digest(p) for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}
def inputs():
    names=['manifest.json',*sorted(json.loads((ASSET/'manifest.json').read_bytes())['buffers'])]
    return {n:digest(ASSET/n) for n in names}
env={**os.environ,'PYTHONPATH':str(SOURCE/'src')}
commands=[
    ('tests-attempt-001',[PYTHON,'-m','unittest','tests.test_cardiac_material_attribution','tests.test_cardiac_material_frames','tests.test_cardiac_wall_source','-v']),
    ('full-source-attempt-001',[PYTHON,'-m','numilab_human.cardiac_material_attribution','--asset',str(ASSET),'--output',str(BASE/'attribution')]),
    ('repeat-source-attempt-001',[PYTHON,'-m','numilab_human.cardiac_material_attribution','--asset',str(ASSET),'--output',str(BASE/'attribution')])]
for name,command in commands:
    record={'schema':'HumanPack.cardiac-material-attribution-execution.v1','command':command,'cwd':str(SOURCE),
        'environment':{'PYTHONPATH':env['PYTHONPATH']},'source_before_sha256':sources(),'input_before_sha256':inputs(),
        'runner_sha256':digest(Path(__file__)),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'physical_steps':0,'status':'running'}
    if name.startswith('repeat'):
        record['output_before']={p.name:{'sha256':digest(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size}
                                 for p in sorted((BASE/'attribution').iterdir())}
    def save():
        (BASE/(name+'.execution.json')).write_text(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n')
    save();start=time.monotonic()
    with (BASE/(name+'.log')).open('wb') as log:
        result=subprocess.run(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        source_after_sha256=sources(),input_after_sha256=inputs(),log_sha256=digest(BASE/(name+'.log')))
    record['inputs_unchanged']=record['input_before_sha256']==record['input_after_sha256']
    record['implementation_unchanged']=record['source_before_sha256']==record['source_after_sha256']
    if (BASE/'attribution').exists():
        record['output_after']={p.name:{'sha256':digest(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size}
                                 for p in sorted((BASE/'attribution').iterdir())}
    record['output_unchanged']=record.get('output_before',record.get('output_after'))==record.get('output_after')
    record['status']='pass' if result.returncode==0 and record['inputs_unchanged'] and record['implementation_unchanged'] and record['output_unchanged'] else 'fail'
    save();print(json.dumps({k:record[k] for k in ['status','elapsed_seconds','peak_child_rss_bytes','returncode']}),flush=True)
    if record['status']!='pass':raise SystemExit(1)
