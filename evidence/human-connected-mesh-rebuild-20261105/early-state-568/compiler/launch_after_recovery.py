from pathlib import Path
import json,time,sys,subprocess
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=Path(__file__).resolve().parent
p=W/'closed-mass-witness-recovery-573/human-closed495-541-mass-witnesses-573/human-closed495-541-mass-witnesses-573-verification.json';start=time.time()
while not p.exists():
 assert time.time()-start<900
 time.sleep(5)
assert json.loads(p.read_text())['status']=='PASS_INDEPENDENT_DOWNLOAD_ALL_MEMBERS_AND_RESTORE'
subprocess.run([sys.executable,str(O/'run_physical_transfer.py')],check=True)
