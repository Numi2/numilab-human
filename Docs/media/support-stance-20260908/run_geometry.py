import subprocess,json,os
from pathlib import Path
base=json.loads(Path("/Users/n/numi-human-active-locomotion-20260908/standing-final-launch.json").read_text())["command"]
cmd=base[:5]+["--whole-body-support-certificate","--muscle-step-seconds","0.0001"]
cmd[4]="/Users/n/numi-human-stance-20260908/geometry"
for opt in ["--support-contact-payload","--joint-equality-payload"]:cmd+=base[base.index(opt):base.index(opt)+2]
p=subprocess.run(cmd,env=dict(os.environ,NUMI_SUPPORT_STANCE_DIAGNOSTIC="1"),capture_output=True,text=True)
Path("/Users/n/numi-human-stance-20260908/geometry.log").write_text(p.stdout+p.stderr)
print(p.returncode,p.stdout[-9500:],p.stderr)
