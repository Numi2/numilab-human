from pathlib import Path
import json,subprocess
out=Path('/Users/n/numi-human-active-locomotion-20260908')
base=json.loads((out/'costal-active-launch.json').read_text())['command']
results=[]
for name,extra,expected in [('zero-roots',[],'--roots must be a positive integer'),('qualified-candidate',['--candidate-sha','a'*64],'cannot train, inherit qualification')]:
 command=list(base)
 if name=='zero-roots':command[command.index('--roots')+1]='0'
 command+=extra
 p=subprocess.run(command,capture_output=True,text=True,timeout=30)
 (out/('cli-'+name+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==64 and expected in p.stderr,(name,p.returncode,p.stderr[-1000:])
 results.append({'name':name,'returncode':p.returncode,'expected_error':expected,'command':command})
(out/'cli-rejections.json').write_text(json.dumps(results,indent=2)+'\n')
print('CLI invalid root count and inherited qualification reject')
