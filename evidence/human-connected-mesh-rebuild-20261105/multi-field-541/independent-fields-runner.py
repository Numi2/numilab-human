from pathlib import Path
import subprocess,sys,json
O=Path(__file__).resolve().parent
for field in ['step16','step58','step64']:
 d=O/field
 with (d/'stdout.log').open('w') as a,(d/'stderr.log').open('w') as b:r=subprocess.run([sys.executable,str(d/'audit_fixed_field.py')],stdout=a,stderr=b)
 if r.returncode:raise SystemExit(r.returncode)
 v=json.loads((d/'report.json').read_text());assert v['calibration_pass'] and v['force_failures']==v['energy_failures']==0,field
 print(field,'PASS',flush=True)
