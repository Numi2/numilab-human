"""Inspect skin and both sets of finger bones on exactly the same native receipt."""
import os,subprocess,sys
from pathlib import Path
from audit_native_finger_cortex import run
p,s=map(Path,sys.argv[1:]);w=Path(os.environ['W'])
from skin_pose_inputs import capture
capture(p,s.parent,s.parent,w)
subprocess.run([sys.executable,w/'resting-clearance-and-atrium-017/audit_native_skin_endpoint.py',p,s],check=True)
run(w,s.parent/'launch.json',p,s.parent/'terminal-cortical.json')
