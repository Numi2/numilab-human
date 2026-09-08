from pathlib import Path
import json,os,subprocess,time,hashlib,signal
out=Path('/Users/n/numi-human-active-locomotion-20260908')
prior=json.loads(Path('/Users/n/numi-human-tissue-ownership-20260908/brain-costal-e2e-launch.json').read_text())
keys=['NUMANX_METALROBO_LIBRARY','NUMANX_FULLBODY_RIGID','NUMANX_FULLBODY_MUSCLE','NUMANX_FULLBODY_CONTACT','NUMANX_METALROBO_METALLIB','NUMANX_MATTER_METALLIB','NUMANX_FULLBODY_VISUAL_PACK','NUMANX_FULLBODY_VISION_PROFILE','NUMANX_MATTER_MATERIAL']
env={k:prior['environment'][k] for k in keys}
env.update(MTL_DEBUG_LAYER='1',NUMANX_GATE_B_EVIDENCE='1')
command=['swift','test','-c','release','--skip-build','--filter','testActiveMuscleLocomotorAcceptedRoots']
brain=Path('/Users/n/numi-brain-human-completion-20260907')
start=time.monotonic()
with (out/'active-native.log').open('w') as log:
 p=subprocess.Popen(command,cwd=brain,env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=360)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);code=124
(out/'active-native-launch.json').write_text(json.dumps({'command':command,'environment':env,'elapsed_seconds':time.monotonic()-start,'returncode':code,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=brain,text=True).strip(),'source_status':subprocess.check_output(['git','status','--porcelain'],cwd=brain,text=True),'artifact_sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in env.items() if Path(v).is_file()}},indent=2)+'\n')
raise SystemExit(code)
