from pathlib import Path
import json,subprocess,os,time,re,hashlib
out=Path('/Users/n/numi-human-balance-solver-20260908');out.mkdir(exist_ok=True)
base=json.loads(Path('/Users/n/numi-human-active-locomotion-20260908/standing-final-launch.json').read_text())['command']
rows=[]
for steps,dt in [(1,'0.0001'),(2,'0.0001'),(4,'0.0001'),(6,'0.0001'),(60,'0.00001')]:
 label=f'geometry-{steps}-{dt}';cmd=list(base);cmd[4]=str(out/label);cmd[cmd.index('--muscle-step-count')+1]=str(steps);cmd[cmd.index('--muscle-step-seconds')+1]=dt
 cmd+=['--camera-index','0'];start=time.monotonic()
 with (out/(label+'.log')).open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,MTL_DEBUG_LAYER='1'))
 kv=dict(re.findall(r'([a-zA-Z_0-9]+)=([^\s]+)',(out/(label+'.log')).read_text()))
 row={'label':label,'command':cmd,'exit_code':p.returncode,'elapsed':time.monotonic()-start,'metrics':{k:v for k,v in kv.items() if k.startswith(('compiled_stand','source_support','persistent_stand','stand_')) or k.startswith('muscle_step_max')}}
 rows.append(row);(out/'geometry.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(label,p.returncode,{k:v for k,v in row['metrics'].items() if k in ['compiled_stand_balanced','muscle_step_max_velocity_delta','muscle_step_max_configuration_delta','source_support_min_plane_gap_m']},flush=True)
 if p.returncode:break
