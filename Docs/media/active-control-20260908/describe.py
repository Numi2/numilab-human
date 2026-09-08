from pathlib import Path
import json,subprocess,time,os
out=Path('/Users/n/numi-human-active-locomotion-20260908')
env=json.loads(Path('/Users/n/numi-human-tissue-ownership-20260908/brain-costal-e2e-launch.json').read_text())['environment']
args={'--library':'NUMANX_METALROBO_LIBRARY','--rigid':'NUMANX_FULLBODY_RIGID','--muscle':'NUMANX_FULLBODY_MUSCLE','--contacts':'NUMANX_FULLBODY_CONTACT','--visual-pack':'NUMANX_FULLBODY_VISUAL_PACK','--vision-profile':'NUMANX_FULLBODY_VISION_PROFILE','--metalrobo-metallib':'NUMANX_METALROBO_METALLIB','--matter-metallib':'NUMANX_MATTER_METALLIB','--matter-world':'NUMANX_MATTER_WORLD_PACKAGE','--human-source-fp':'NUMANX_HUMAN_SOURCE_FP','--matter-world-fp':'NUMANX_MATTER_WORLD_FP','--joint-equalities':'NUMANX_JOINT_EQUALITIES','--joint-equality-fp':'NUMANX_JOINT_EQUALITY_FP','--costal-cartilage':'NUMANX_COSTAL_CARTILAGE','--costal-binding':'NUMANX_COSTAL_BINDING','--costal-binding-fp':'NUMANX_COSTAL_BINDING_FP'}
command=['/Users/n/numi-brain-human-completion-20260907/.build/arm64-apple-macosx/release/numi-brain-gate-c','describe-body']
for key,value in args.items():command +=[key,env[value]]
command+=['--timestep-microseconds','10']
(out/'costal-native-arguments.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic()
with (out/'costal-body.json').open('w') as stdout,(out/'costal-describe.log').open('w') as stderr:
 p=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=180,env=dict(os.environ,MTL_DEBUG_LAYER='1'))
print('describe_returncode',p.returncode,'elapsed',time.monotonic()-start)
raise SystemExit(p.returncode)
