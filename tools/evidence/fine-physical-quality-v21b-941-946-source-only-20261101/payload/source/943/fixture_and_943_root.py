import runpy,subprocess,sys,json,hashlib
from pathlib import Path
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
f=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v26c-expanded-joint-stars/test_v26c_expanded_joint_star_fixture.py"
assert hashlib.sha256(f.read_bytes()).hexdigest()=="aa788b4553bad4d89821a9bfcd0336ef0185d96141177c5b7ade33893370952e"
subprocess.run([sys.executable,str(f)],check=True)
runpy.run_path(str(H/"run_points_943_root.py"),run_name="__main__")
