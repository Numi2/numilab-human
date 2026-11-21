import csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from material_context import sha
from capture_posterior import audit_native
from preserve_posterior_packs import keep_delta
owner,configuration,w,cap,audit,exe,patches,reference,scratch,out=map(Path,sys.argv[1:])
run=scratch/'scene';out.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(owner/'run_scene.py'),str(configuration),str(run)],check=True)
for name in ('launch.json','result.json','stdout.log','stderr.log'):
    shutil.copyfile(run/name,out/name)
report=json.loads((out/'result.json').read_text())
if report['exit_code'] or report['trace_rows']!=8:raise RuntimeError(report)
for name in ('resting-coupled.csv','resting-com-support-impulses.csv','resting-body-motion-partition.csv','resting-surface-audit.csv'):
    shutil.copyfile(run/'native'/name,out/name)
ar=list(csv.DictReader((out/'resting-coupled.csv').open()));br=list(csv.DictReader((reference/'resting-coupled.csv').open()))
if len(ar)!=len(br) or ar[0].keys()!=br[0].keys():raise ValueError('Incomplete comparison')
report.update(exact_columns=sum(all(a[k]==b[k] for a,b in zip(ar,br,strict=True)) for k in ar[0]),different_columns=[k for k in ar[0] if any(a[k]!=b[k] for a,b in zip(ar,br,strict=True))],reference_launch_sha256=sha(reference/'launch.json'),accepted_time_s=float(ar[-1]['time_s']))
(out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
capdir=run/'native/accepted-geometry';r=next(capdir.glob('step-8.receipt.json'));shutil.copyfile(r,out/r.name)
rc=json.loads(r.read_text())
entry=run/'native/resting-human.mrvpack';end=Path(rc['accepted_pack_path'])
base=w/'qualified/native-coupled-006/native/resting-human.mrvpack'
provenance=keep_delta(base,entry,out/'entrance.xor.zst')
previous=next((w/'qualified/motion-008/capture').glob('step-8.mrvpack.zst'))
second=keep_delta(previous,end,out/'step-8.xor.zst')
(out/'geometry-recovery.json').write_text(json.dumps([provenance,second],indent=2)+'\n')
audit_native(w,cap,out/'launch.json',audit,exe,patches,run,out)
