import subprocess,json,os
from pathlib import Path
base=json.loads(Path("/Users/n/numi-human-active-locomotion-20260908/standing-final-launch.json").read_text())["command"]
cmd=base[:5]+["--whole-body-support-certificate","--muscle-step-seconds","0.0001"]
cmd[4]="/Users/n/numi-human-stance-20260908/geometry"
for opt in ["--support-contact-payload","--joint-equality-payload"]:cmd+=base[base.index(opt):base.index(opt)+2]
for dof,bound in [(2,0.02),(108,0.1),(109,0.1),(110,0.1),(122,0.1),(123,0.1),(124,0.1)]:cmd += ["--support-stance-dof",str(dof),str(bound)]
for contact in [2,3,4,5,6,7]:cmd += ["--support-stance-contact",str(contact)]
Path("/Users/n/numi-human-stance-20260908/stance-launch.json").write_text(json.dumps({"command":cmd},indent=2))
with Path("/Users/n/numi-human-stance-20260908/stance.log").open("w") as log:
 p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
print("exit_code",p.returncode)
raise SystemExit(p.returncode)
