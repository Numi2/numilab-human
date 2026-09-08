from pathlib import Path
import hashlib,json,os,subprocess,time,signal
out=Path('/Users/n/numi-human-active-locomotion-20260908')
prior=json.loads(Path('/Users/n/numi-human-tissue-ownership-20260908/brain-costal-e2e-launch.json').read_text())
brain=Path('/Users/n/numi-brain-human-completion-20260907')
env=dict(os.environ,**prior['environment']);env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
command=prior['command'];start=time.monotonic()
with (out/'costal-regression.log').open('w') as log:
 p=subprocess.Popen(command,cwd=brain,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=480)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
(out/'costal-regression-launch.json').write_text(json.dumps({'command':command,'environment':prior['environment'],'returncode':code,'elapsed_seconds':time.monotonic()-start,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=brain,text=True).strip(),'artifact_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in prior['environment'].items() if Path(v).is_file()}},indent=2)+'\n')
raise SystemExit(code)
