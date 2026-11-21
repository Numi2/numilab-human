from pathlib import Path
import hashlib,json,os,subprocess,sys
base,cwd,out=map(Path,sys.argv[1:])
cfg=json.loads((base/'cardiac-targets-launch.json').read_text())
for path,sha in cfg['pins'].items():
    with open(path,'rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==sha,path
args=cfg['argv'];assert args[4:]==['--steps','8000','--dt','0.002','--vascular-dense45']
args[3]=str(out/'forecast.csv');args[5]='30000'
assert cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1'
(out/'forecast-launch.json').write_text(json.dumps(dict(argv=args,cwd=str(cwd),environment=cfg['environment'],pins=cfg['pins']),indent=2)+'\n')
base_env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
subprocess.run(args,env=base_env|cfg['environment'],cwd=cwd,check=True)
