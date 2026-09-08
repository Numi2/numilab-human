import json,subprocess,os,time,re
from pathlib import Path
out=Path('/Users/n/numi-human-stance-20260908')
base=json.loads(Path('/Users/n/numi-human-active-locomotion-20260908/standing-final-launch.json').read_text())['command']
profile=json.loads((out/'stance-launch.json').read_text())['command'];extras=profile[profile.index('--support-stance-dof'):]
rows=[]
for steps,dt in [(1,'0.0001'),(64,'0.0001'),(128,'0.00005'),(256,'0.000025'),(1024,'0.0001')]:
 label=f'native-{steps}-{dt}';cmd=list(base);cmd[4]=str(out/label);cmd[cmd.index('--muscle-step-count')+1]=str(steps);cmd[cmd.index('--muscle-step-seconds')+1]=dt;cmd+=extras+['--camera-index','0']
 workload=subprocess.run(['ps','-axo','pid,pcpu,etime,command'],capture_output=True,text=True).stdout
 (out/(label+'-workload.txt')).write_text('\n'.join(l for l in workload.splitlines() if 'numivivo md-benchmark' in l)+'\n')
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,MTL_DEBUG_LAYER='1'))
 kv=dict(re.findall(r'([a-zA-Z_0-9]+)=([^\s]+)',(out/(label+'.log')).read_text()))
 row={'label':label,'command':cmd,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start,'metrics':{k:v for k,v in kv.items() if k.startswith(('compiled_stand','source_support','persistent_stand','stand_','muscle_step_max'))}};rows.append(row);(out/'native.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(label,p.returncode,{k:v for k,v in row['metrics'].items() if k in ['compiled_stand_balanced','muscle_step_max_velocity_delta','muscle_step_max_configuration_delta','source_support_min_plane_gap_m']},flush=True)
 if p.returncode:break
