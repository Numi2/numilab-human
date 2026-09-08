from pathlib import Path
import json,subprocess,time,os,hashlib
out=Path('/Users/n/numi-human-active-locomotion-20260908');src=Path('/Users/n/human-completion-20260907/input')
cmd=['/Users/n/MetalRobo-human-completion-build-20260907/bin/metalrobo_numilab_human_myosim_visual_probe',str(src/'myosim-fullbody-core-reference.nhrigid'),str(src/'myosim-fullbody-muscle-reference.nhmyo'),str(src/'bodyparts3d-myosim-major-bones.nhbones'),str(out/'standing-final'), '--muscle-step-seconds','0.0001','--muscle-step-count','8','--persistent-metal-stand','--stand-deterministic-replay','--support-contact-payload',str(src/'myosim-fullbody-support-contact.nhcnt'),'--tendon-payload',str(src/'numi-human-tendon-attachments.nhtendon'),'--joint-equality-payload',str(src/'myosim-fullbody-joint-equalities.nheq'),'--dimension','512']
start=time.monotonic()
with (out/'standing-final.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,MTL_DEBUG_LAYER='1'))
(out/'standing-final-launch.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode,'elapsed_seconds':time.monotonic()-start,'native_revision':subprocess.check_output(['git','-C','/Users/n/MetalRobo-human-completion-20260907','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest(),'boundary':'legacy NHEQ1 native recruitment baseline; not Brain or costal standing qualification'},indent=2)+'\n')
raise SystemExit(p.returncode)
